"""Minimal Inference Pipeline: per-phase timing breakdown.

Strips ALL abstraction layers and instruments each phase of Boltz-2
inference with CUDA synchronization barriers for precise timing.

Phases measured:
1. Input processing (YAML parse + featurization + batch transfer)
2. Input embedding + init (input_embedder, s_init, z_init, rel_pos)
3. MSA module
4. Pairformer trunk
5. Diffusion conditioning
6. Diffusion sampling (ODE steps)
7. Confidence module
8. Output writing

Also applies proven optimizations:
- ODE-12 (gamma_0=0) for fast deterministic sampling
- TF32 matmul precision
- bf16 trunk (no .float() upcast in triangular_mult)
- DiffusionTransformer layer truncation (24 -> 8 layers)
- cuequivariance CUDA kernels

Usage:
    modal run orbits/minimal-inference/eval_minimal.py --mode eval
    modal run orbits/minimal-inference/eval_minimal.py --mode multi
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
    )
    .pip_install(
        "boltz==2.2.1",
    )
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
)

app = modal.App("boltz-eval-minimal-inference", image=boltz_image)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BEST_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "diffusion_samples": 1,
    "seed": 42,
    "truncate_diffusion_layers": 8,  # 24 -> 8 layers (proven safe)
}


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

def _patch_diffusion_params(gamma_0_val: float, noise_scale_val: float):
    import boltz.main as boltz_main
    _g0 = gamma_0_val
    _ns = noise_scale_val

    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = field(default_factory=lambda: _g0)
        gamma_min: float = 1.0
        noise_scale: float = field(default_factory=lambda: _ns)
        rho: float = 7
        step_scale: float = 1.5
        sigma_min: float = 0.0001
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True

    boltz_main.Boltz2DiffusionParams = PatchedBoltz2DiffusionParams
    return PatchedBoltz2DiffusionParams


def _patch_triangular_mult_bf16():
    import torch
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    def forward_outgoing_bf16(self, x, mask, use_kernels=False):
        if use_kernels:
            from boltz.model.layers.triangular_mult import kernel_triangular_mult
            return kernel_triangular_mult(
                x, direction="outgoing", mask=mask,
                norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight, g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight,
                eps=1e-5,
            )
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = torch.einsum("bikd,bjkd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    def forward_incoming_bf16(self, x, mask, use_kernels=False):
        if use_kernels:
            from boltz.model.layers.triangular_mult import kernel_triangular_mult
            return kernel_triangular_mult(
                x, direction="incoming", mask=mask,
                norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight, g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight,
                eps=1e-5,
            )
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = torch.einsum("bkid,bkjd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    TriangleMultiplicationOutgoing.forward = forward_outgoing_bf16
    TriangleMultiplicationIncoming.forward = forward_incoming_bf16


def _truncate_diffusion_transformer(model, num_layers: int):
    """Truncate DiffusionTransformer layers from 24 to num_layers.

    The DiffusionTransformer splits bias into per-layer chunks (D // L).
    When truncating, we need to adjust the to_keys projection that produces
    the bias tensor so it outputs the right dimension.

    Approach: just truncate the layer list. The bias is split at runtime
    in the forward method using D // L where L = len(self.layers), so
    truncating layers automatically adjusts the split. But we need to
    ensure the bias dimension (D) is divisible by the new layer count.

    Actually, looking at the code: bias is produced by diffusion_conditioning
    with a fixed dimension. The DiffusionTransformer.forward does:
        bias = bias.view(B, N, M, L, D // L)
    where L = len(self.layers). So if we change L, D // L changes.

    The bias dimension D is set at model init time. We need to check if
    D is divisible by the new layer count.

    Safest approach: truncate layers AND adjust the bias view in forward.
    We'll monkey-patch the forward to handle the dimension mismatch.
    """
    import torch
    from torch.nn import ModuleList

    score_model = model.structure_module.score_model
    token_transformer = score_model.token_transformer
    original_layers = len(token_transformer.layers)

    if num_layers >= original_layers:
        print(f"[minimal] DiffusionTransformer already has {original_layers} layers, "
              f"not truncating to {num_layers}")
        return

    # Truncate layers
    token_transformer.layers = ModuleList(list(token_transformer.layers)[:num_layers])

    # Monkey-patch forward to handle bias dimension mismatch.
    # The bias tensor has shape (B, N, M, D) where D = original_layers * per_layer_dim.
    # The forward does: bias.view(B, N, M, L, D // L) where L = len(self.layers).
    # After truncation L changed, so D // L is wrong. We need to use the original
    # per-layer dim and only take the first num_layers slices.

    def patched_forward(a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
        if token_transformer.pair_bias_attn and bias is not None:
            B, N, M, D = bias.shape
            L = len(token_transformer.layers)
            per_layer_dim = D // original_layers  # original per-layer dim

            # Take only the first num_layers * per_layer_dim dimensions
            bias_used = bias[:, :, :, :L * per_layer_dim]
            bias_used = bias_used.view(B, N, M, L, per_layer_dim)

            for i, layer in enumerate(token_transformer.layers):
                bias_l = bias_used[:, :, :, i]
                a = layer(a, s, bias_l, mask, to_keys, multiplicity)
            return a
        else:
            for layer in token_transformer.layers:
                a = layer(a, s, None, mask, to_keys, multiplicity)
            return a

    token_transformer.forward = patched_forward
    print(f"[minimal] DiffusionTransformer truncated: {original_layers} -> {num_layers} layers")


CPU_KEYS = frozenset([
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
    "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
    "record", "affinity_mw",
])


def _run_predict_step_phased(model, batch, device):
    """Run model forward with per-phase CUDA timing.

    Instead of calling model(batch) which lumps everything together,
    we replicate the forward() logic with timing barriers between phases.
    """
    import torch

    timings = {}

    def sync_and_time():
        torch.cuda.synchronize()
        return time.perf_counter()

    # ===== Phase 1: Input Embedding =====
    t0 = sync_and_time()

    feats = batch
    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)
    z_init = (
        model.z_init_1(s_inputs)[:, :, None]
        + model.z_init_2(s_inputs)[:, None, :]
    )
    relative_position_encoding = model.rel_pos(feats)
    z_init = z_init + relative_position_encoding
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    t1 = sync_and_time()
    timings["input_embedding_s"] = t1 - t0

    # ===== Phase 2: Recycling Init + MSA + Pairformer =====
    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    recycling_steps = model.predict_args["recycling_steps"]

    for i in range(recycling_steps + 1):
        # Recycling
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))

        # Templates
        if model.use_templates:
            template_module = model.template_module
            if hasattr(model, 'is_template_compiled') and model.is_template_compiled:
                template_module = model.template_module._orig_mod
            z = z + template_module(z, feats, pair_mask, use_kernels=model.use_kernels)

        # MSA
        t_msa_start = sync_and_time()
        msa_module = model.msa_module
        if hasattr(model, 'is_msa_compiled') and model.is_msa_compiled:
            msa_module = model.msa_module._orig_mod
        z = z + msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)
        t_msa_end = sync_and_time()

        # Pairformer
        t_pf_start = sync_and_time()
        pairformer_module = model.pairformer_module
        if hasattr(model, 'is_pairformer_compiled') and model.is_pairformer_compiled:
            pairformer_module = model.pairformer_module._orig_mod
        s, z = pairformer_module(
            s, z, mask=mask, pair_mask=pair_mask,
            use_kernels=model.use_kernels,
        )
        t_pf_end = sync_and_time()

    timings["msa_s"] = t_msa_end - t_msa_start
    timings["pairformer_s"] = t_pf_end - t_pf_start

    # ===== Phase 3: Distogram =====
    pdistogram = model.distogram_module(z)

    t2 = sync_and_time()
    timings["distogram_s"] = t2 - t_pf_end

    # ===== Phase 4: Diffusion Conditioning =====
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            s_trunk=s,
            z_trunk=z,
            relative_position_encoding=relative_position_encoding,
            feats=feats,
        )
    )
    diffusion_conditioning = {
        "q": q, "c": c, "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }

    t3 = sync_and_time()
    timings["diffusion_conditioning_s"] = t3 - t2

    # ===== Phase 5: Diffusion Sampling =====
    num_sampling_steps = model.predict_args["sampling_steps"]
    diffusion_samples = model.predict_args["diffusion_samples"]
    max_parallel_samples = model.predict_args.get("max_parallel_samples")

    with torch.autocast("cuda", enabled=False):
        struct_out = model.structure_module.sample(
            s_trunk=s.float(),
            s_inputs=s_inputs.float(),
            feats=feats,
            num_sampling_steps=num_sampling_steps,
            atom_mask=feats["atom_pad_mask"].float(),
            multiplicity=diffusion_samples,
            max_parallel_samples=max_parallel_samples,
            steering_args=model.steering_args,
            diffusion_conditioning=diffusion_conditioning,
        )

    t4 = sync_and_time()
    timings["diffusion_sampling_s"] = t4 - t3

    # ===== Phase 6: Confidence =====
    if model.confidence_prediction:
        conf_out = model.confidence_module(
            s_inputs=s_inputs.detach(),
            s=s.detach(),
            z=z.detach(),
            x_pred=struct_out["sample_atom_coords"].detach(),
            feats=feats,
            pred_distogram_logits=pdistogram[:, :, :, 0].detach(),
            multiplicity=diffusion_samples,
            run_sequentially=True,
            use_kernels=model.use_kernels,
        )

    t5 = sync_and_time()
    timings["confidence_s"] = t5 - t4

    # Build output dict (same as predict_step)
    dict_out = {
        "pdistogram": pdistogram,
        "s": s, "z": z,
    }
    dict_out.update(struct_out)
    if model.confidence_prediction:
        dict_out.update(conf_out)

    pred_dict = {"exception": False}
    if "keys_dict_batch" in model.predict_args:
        for key in model.predict_args["keys_dict_batch"]:
            pred_dict[key] = batch[key]

    pred_dict["masks"] = batch["atom_pad_mask"]
    pred_dict["token_masks"] = batch["token_pad_mask"]
    pred_dict["s"] = s
    pred_dict["z"] = z

    if "keys_dict_out" in model.predict_args:
        for key in model.predict_args["keys_dict_out"]:
            pred_dict[key] = dict_out[key]
    pred_dict["coords"] = struct_out["sample_atom_coords"]

    if model.confidence_prediction:
        pred_dict["pde"] = conf_out["pde"]
        pred_dict["plddt"] = conf_out["plddt"]
        pred_dict["confidence_score"] = (
            4 * conf_out["complex_plddt"]
            + (
                conf_out["iptm"]
                if not torch.allclose(conf_out["iptm"], torch.zeros_like(conf_out["iptm"]))
                else conf_out["ptm"]
            )
        ) / 5
        pred_dict["complex_plddt"] = conf_out["complex_plddt"]
        pred_dict["complex_iplddt"] = conf_out["complex_iplddt"]
        pred_dict["complex_pde"] = conf_out["complex_pde"]
        pred_dict["complex_ipde"] = conf_out["complex_ipde"]
        if model.alpha_pae > 0:
            pred_dict["pae"] = conf_out["pae"]
            pred_dict["ptm"] = conf_out["ptm"]
            pred_dict["iptm"] = conf_out["iptm"]
            pred_dict["ligand_iptm"] = conf_out["ligand_iptm"]
            pred_dict["protein_iptm"] = conf_out["protein_iptm"]
            pred_dict["pair_chains_iptm"] = conf_out["pair_chains_iptm"]

    timings["total_gpu_s"] = t5 - t0

    return pred_dict, timings


def _parse_confidence_from_dir(results_dir: Path) -> dict[str, Any]:
    quality: dict[str, Any] = {}
    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON files found"}
    with confidence_files[0].open() as f:
        conf = json.load(f)
    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]
    return quality


def _load_config_yaml() -> dict:
    import yaml
    with Path("/eval/config.yaml").open() as f:
        return yaml.safe_load(f)


def _compute_aggregates(results: dict, eval_config: dict) -> dict:
    successful = [
        r for r in results["per_complex"]
        if r["error"] is None and r["wall_time_s"] is not None
    ]
    test_cases = eval_config.get("test_cases", [])

    if len(successful) < len(test_cases):
        failed_names = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        return {
            "error": f"Not all test cases succeeded. Failed: {failed_names}",
            "num_successful": len(successful),
            "num_total": len(test_cases),
            "speedup": 0,
            "passes_quality_gate": False,
        }

    if not successful:
        return {"error": "No successful test cases"}

    total_time = sum(r["wall_time_s"] for r in successful)
    mean_time = total_time / len(successful)
    plddts_raw = [
        r["quality"]["complex_plddt"]
        for r in successful
        if "complex_plddt" in r["quality"]
    ]
    plddts = [
        p for p in plddts_raw
        if p is not None and isinstance(p, (int, float))
        and not math.isnan(p) and not math.isinf(p) and 0.0 <= p <= 1.0
    ]

    agg = {
        "num_successful": len(successful),
        "num_total": len(test_cases),
        "total_wall_time_s": total_time,
        "mean_wall_time_s": mean_time,
        "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
    }

    baseline = eval_config.get("baseline")
    if baseline is not None and baseline:
        baseline_time = baseline.get("mean_wall_time_s")
        baseline_plddt = baseline.get("mean_plddt")
        if baseline_time and mean_time > 0:
            agg["speedup"] = baseline_time / mean_time
        if baseline_plddt is not None and plddts:
            mean_plddt = sum(plddts) / len(plddts)
            agg["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
            regression = (baseline_plddt - mean_plddt) * 100.0
            agg["passes_quality_gate"] = regression <= 2.0

            if baseline.get("per_complex"):
                baseline_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                per_complex_violations = {}
                for r in successful:
                    bl_case = baseline_by_name.get(r["name"])
                    if bl_case and bl_case.get("complex_plddt") is not None:
                        case_plddt = r["quality"].get("complex_plddt")
                        if case_plddt is None:
                            agg["passes_quality_gate"] = False
                            per_complex_violations[r["name"]] = "missing pLDDT"
                        else:
                            case_regression = (bl_case["complex_plddt"] - case_plddt) * 100.0
                            if case_regression > 5.0:
                                agg["passes_quality_gate"] = False
                                per_complex_violations[r["name"]] = f"-{case_regression:.1f}pp"
                if per_complex_violations:
                    agg["per_complex_regression"] = per_complex_violations

    return agg


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
)
def evaluate_minimal(config_json: str) -> str:
    """Run all test cases with minimal inference pipeline + per-phase timing."""
    import torch
    from pytorch_lightning import seed_everything
    from rdkit import Chem

    config = json.loads(config_json)
    seed = config.get("seed", 42)

    torch.set_float32_matmul_precision(config.get("matmul_precision", "highest"))
    torch.set_grad_enabled(False)
    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    if seed is not None:
        seed_everything(seed)

    import boltz.main as boltz_main
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.module.inferencev2 import PredictionDataset, collate
    from boltz.data.write.writer import BoltzWriter
    from boltz.data.types import Manifest

    gamma_0 = config.get("gamma_0", 0.8)
    noise_scale = config.get("noise_scale", 1.003)
    PatchedParams = _patch_diffusion_params(gamma_0, noise_scale)

    if config.get("bf16_trunk", False):
        _patch_triangular_mult_bf16()
        print("[minimal] bf16 trunk patch applied")

    use_kernels = config.get("enable_kernels", True)
    try:
        import cuequivariance_torch
        print(f"[minimal] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        use_kernels = False

    cache = Path("~/.boltz").expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Download model
    t_download_start = time.perf_counter()
    boltz_main.download_boltz2(cache)
    t_download_end = time.perf_counter()
    download_time = t_download_end - t_download_start
    print(f"[minimal] Download/cache check: {download_time:.1f}s")

    # Load model to GPU
    diffusion_params = PatchedParams()
    step_scale = config.get("step_scale", 1.5)
    diffusion_params.step_scale = step_scale
    pairformer_args = boltz_main.PairformerArgsV2()
    msa_args = boltz_main.MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True,
    )
    steering_args = boltz_main.BoltzSteeringParams()

    predict_args = {
        "recycling_steps": config.get("recycling_steps", 3),
        "sampling_steps": config.get("sampling_steps", 200),
        "diffusion_samples": config.get("diffusion_samples", 1),
        "max_parallel_samples": config.get("max_parallel_samples", None),
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    t_load_start = time.perf_counter()
    checkpoint = cache / "boltz2_conf.ckpt"
    device = torch.device("cuda")

    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=use_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    model = model.to(device)

    # Apply DiffusionTransformer layer truncation
    truncate_layers = config.get("truncate_diffusion_layers")
    if truncate_layers is not None:
        _truncate_diffusion_transformer(model, truncate_layers)

    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start
    print(f"[minimal] Model loaded to GPU: {load_time:.1f}s")

    # Run test cases
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    mol_dir = cache / "mols"

    results: dict[str, Any] = {
        "config": config,
        "download_time_s": download_time,
        "load_time_s": load_time,
        "per_complex": [],
        "aggregate": {},
        "phase_timings": [],  # Per-complex phase breakdowns
    }

    results["env"] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"Test case YAML not found: {tc_yaml}",
                "wall_time_s": None,
                "quality": {},
            })
            continue

        work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[minimal] Running {tc_name}")

        try:
            t_start = time.perf_counter()

            # Input processing (YAML parse, featurization)
            t_input_start = time.perf_counter()
            data = boltz_main.check_inputs(tc_yaml)
            out_dir = work_dir / f"boltz_results_{tc_yaml.stem}"
            out_dir.mkdir(parents=True, exist_ok=True)

            boltz_main.process_inputs(
                data=data, out_dir=out_dir, ccd_path=cache / "ccd.pkl",
                mol_dir=mol_dir, use_msa_server=True,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy", boltz2=True,
                preprocessing_threads=1, max_msa_seqs=8192,
            )

            manifest = Manifest.load(out_dir / "processed" / "manifest.json")
            filtered_manifest = boltz_main.filter_inputs_structure(
                manifest=manifest, outdir=out_dir, override=True,
            )

            if not filtered_manifest.records:
                results["per_complex"].append({
                    "name": tc_name, "error": "No records",
                    "wall_time_s": None, "quality": {},
                })
                continue

            processed_dir = out_dir / "processed"
            targets_dir = processed_dir / "structures"
            msa_dir_local = processed_dir / "msa"
            constraints_dir = (processed_dir / "constraints") if (processed_dir / "constraints").exists() else None
            template_dir = (processed_dir / "templates") if (processed_dir / "templates").exists() else None
            extra_mols_dir = (processed_dir / "mols") if (processed_dir / "mols").exists() else None

            dataset = PredictionDataset(
                manifest=filtered_manifest,
                target_dir=targets_dir, msa_dir=msa_dir_local,
                mol_dir=mol_dir, constraints_dir=constraints_dir,
                template_dir=template_dir, extra_mols_dir=extra_mols_dir,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=0,
                pin_memory=True, shuffle=False, collate_fn=collate,
            )

            pred_writer = BoltzWriter(
                data_dir=targets_dir,
                output_dir=out_dir / "predictions",
                output_format="mmcif", boltz2=True,
            )

            t_input_end = time.perf_counter()
            input_processing_time = t_input_end - t_input_start

            # GPU inference with per-phase timing
            for batch_idx, batch in enumerate(dataloader):
                # Transfer to device
                t_transfer_start = time.perf_counter()
                batch = {
                    k: (v.to(device, non_blocking=True) if k not in CPU_KEYS else v)
                    for k, v in batch.items()
                }
                torch.cuda.synchronize()
                t_transfer_end = time.perf_counter()

                # Run phased inference
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_dict, phase_timings = _run_predict_step_phased(model, batch, device)

                phase_timings["batch_transfer_s"] = t_transfer_end - t_transfer_start
                phase_timings["input_processing_s"] = input_processing_time

                # Write output
                t_write_start = time.perf_counter()
                pred_writer.write_on_batch_end(
                    trainer=None, pl_module=None,
                    prediction=pred_dict, batch_indices=[batch_idx],
                    batch=batch, batch_idx=batch_idx, dataloader_idx=0,
                )
                t_write_end = time.perf_counter()
                phase_timings["output_writing_s"] = t_write_end - t_write_start

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            wall_time = t_end - t_start

            pred_dir = out_dir / "predictions"
            quality = {}
            if pred_dir.exists():
                for subdir in pred_dir.iterdir():
                    if subdir.is_dir():
                        quality = _parse_confidence_from_dir(subdir)
                        break

            entry = {
                "name": tc_name,
                "wall_time_s": wall_time,
                "gpu_time_s": phase_timings.get("total_gpu_s"),
                "quality": quality,
                "error": None,
                "phase_timings": phase_timings,
            }

            # Print phase breakdown
            print(f"[minimal] {tc_name}: total={wall_time:.1f}s, "
                  f"gpu={phase_timings.get('total_gpu_s', 0):.1f}s, "
                  f"pLDDT={quality.get('complex_plddt', 'N/A')}")
            print(f"  Phase breakdown:")
            print(f"    Input processing:       {phase_timings.get('input_processing_s', 0):.2f}s")
            print(f"    Batch transfer:         {phase_timings.get('batch_transfer_s', 0):.3f}s")
            print(f"    Input embedding:        {phase_timings.get('input_embedding_s', 0):.3f}s")
            print(f"    MSA module:             {phase_timings.get('msa_s', 0):.3f}s")
            print(f"    Pairformer:             {phase_timings.get('pairformer_s', 0):.2f}s")
            print(f"    Distogram:              {phase_timings.get('distogram_s', 0):.3f}s")
            print(f"    Diffusion conditioning: {phase_timings.get('diffusion_conditioning_s', 0):.3f}s")
            print(f"    Diffusion sampling:     {phase_timings.get('diffusion_sampling_s', 0):.2f}s")
            print(f"    Confidence:             {phase_timings.get('confidence_s', 0):.2f}s")
            print(f"    Output writing:         {phase_timings.get('output_writing_s', 0):.3f}s")

        except Exception as exc:
            import traceback
            entry = {
                "name": tc_name,
                "wall_time_s": None,
                "gpu_time_s": None,
                "quality": {},
                "error": f"{exc}\n{traceback.format_exc()[-1000:]}",
                "phase_timings": {},
            }
            print(f"[minimal] {tc_name}: ERROR - {exc}")

        results["per_complex"].append(entry)
        results["phase_timings"].append({
            "name": tc_name,
            "timings": entry.get("phase_timings", {}),
        })

    results["aggregate"] = _compute_aggregates(results, eval_config)

    # Additional metrics
    successful = [r for r in results["per_complex"] if r["error"] is None and r["wall_time_s"] is not None]
    if successful:
        n = len(successful)
        total_inference = sum(r["wall_time_s"] for r in successful)
        mean_inference = total_inference / n
        total_gpu = sum(r.get("gpu_time_s", 0) for r in successful if r.get("gpu_time_s"))
        mean_gpu = total_gpu / n if total_gpu else None

        mean_with_load = mean_inference + load_time / n

        results["aggregate"]["mean_time_no_load"] = mean_inference
        results["aggregate"]["mean_time_with_load"] = mean_with_load
        if mean_gpu:
            results["aggregate"]["mean_gpu_time"] = mean_gpu

        baseline = eval_config.get("baseline", {})
        baseline_time = baseline.get("mean_wall_time_s")
        if baseline_time:
            results["aggregate"]["speedup_no_load"] = baseline_time / mean_inference
            results["aggregate"]["speedup_with_load"] = baseline_time / mean_with_load
            if mean_gpu:
                results["aggregate"]["speedup_gpu_only"] = baseline_time / mean_gpu

        # Compute mean phase timings
        all_phases = [r.get("phase_timings", {}) for r in successful if r.get("phase_timings")]
        if all_phases:
            phase_keys = [
                "input_processing_s", "batch_transfer_s", "input_embedding_s",
                "msa_s", "pairformer_s", "distogram_s",
                "diffusion_conditioning_s", "diffusion_sampling_s",
                "confidence_s", "output_writing_s", "total_gpu_s",
            ]
            mean_phases = {}
            for pk in phase_keys:
                vals = [p.get(pk, 0) for p in all_phases if pk in p]
                if vals:
                    mean_phases[f"mean_{pk}"] = sum(vals) / len(vals)
            results["aggregate"]["mean_phase_timings"] = mean_phases

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    config: str = "",
    seed: int = -1,
):
    """Minimal inference pipeline evaluation."""
    if mode == "eval":
        cfg = json.loads(config) if config else dict(BEST_CONFIG)
        if seed >= 0:
            cfg["seed"] = seed
        print(f"[minimal] Evaluating: {json.dumps(cfg)}")
        result_json = evaluate_minimal.remote(json.dumps(cfg))
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "multi":
        seeds = [42, 123, 7]
        cfg_base = json.loads(config) if config else dict(BEST_CONFIG)

        config_jsons = []
        for s in seeds:
            c = dict(cfg_base)
            c["seed"] = s
            config_jsons.append(json.dumps(c))

        print(f"[minimal] Running {len(seeds)} seeds in parallel: {seeds}")
        all_results = []
        for s, result_json in zip(seeds, evaluate_minimal.map(config_jsons)):
            result = json.loads(result_json)
            all_results.append(result)
            agg = result.get("aggregate", {})
            t_no = agg.get("mean_time_no_load", 0)
            t_with = agg.get("mean_time_with_load", 0)
            t_gpu = agg.get("mean_gpu_time", 0)
            p = agg.get("mean_plddt", 0)
            s_no = agg.get("speedup_no_load", 0)
            s_with = agg.get("speedup_with_load", 0)
            dl = result.get("download_time_s", 0)
            ld = result.get("load_time_s", 0)
            print(f"  Seed {s}: no_load={t_no:.1f}s ({s_no:.2f}x), "
                  f"with_load={t_with:.1f}s ({s_with:.2f}x), "
                  f"gpu={t_gpu:.1f}s, pLDDT={p:.4f}, "
                  f"download={dl:.1f}s, load={ld:.1f}s")

        _print_multi_summary(all_results, seeds)

    elif mode == "no-truncate":
        # Same as eval but without DiffusionTransformer truncation
        cfg = json.loads(config) if config else dict(BEST_CONFIG)
        cfg["truncate_diffusion_layers"] = None  # disable truncation
        if seed >= 0:
            cfg["seed"] = seed
        print(f"[minimal] Evaluating (no truncation): {json.dumps(cfg)}")
        result_json = evaluate_minimal.remote(json.dumps(cfg))
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "ablation":
        # Compare: with vs without DiffusionTransformer truncation
        cfg_base = json.loads(config) if config else dict(BEST_CONFIG)

        configs = [
            {"label": "minimal+truncate8", **cfg_base},
            {"label": "minimal-no-truncate", **{**cfg_base, "truncate_diffusion_layers": None}},
        ]

        config_jsons = []
        for c in configs:
            label = c.pop("label")
            config_jsons.append(json.dumps(c))

        labels = ["minimal+truncate8", "minimal-no-truncate"]
        print(f"[minimal] Running ablation: {labels}")
        for label, result_json in zip(labels, evaluate_minimal.map(config_jsons)):
            result = json.loads(result_json)
            agg = result.get("aggregate", {})
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            _print_summary(result)


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    dl = result.get("download_time_s")
    ld = result.get("load_time_s")

    if dl is not None:
        print(f"  Download time:    {dl:.1f}s")
    if ld is not None:
        print(f"  Model load time:  {ld:.1f}s")

    for key, label in [
        ("mean_time_no_load", "Mean time (no load)"),
        ("mean_time_with_load", "Mean time (with load)"),
        ("mean_gpu_time", "Mean GPU time"),
        ("mean_plddt", "Mean pLDDT"),
        ("speedup_no_load", "Speedup (no load)"),
        ("speedup_with_load", "Speedup (with load)"),
        ("speedup_gpu_only", "Speedup (GPU only)"),
        ("plddt_delta_pp", "pLDDT delta"),
        ("passes_quality_gate", "Quality gate"),
    ]:
        val = agg.get(key)
        if val is not None:
            if key == "passes_quality_gate":
                print(f"  {label}: {'PASS' if val else 'FAIL'}")
            elif "plddt" in key and "delta" not in key:
                print(f"  {label}: {val:.4f}")
            elif "delta" in key:
                print(f"  {label}: {val:+.2f} pp")
            elif "speedup" in key:
                print(f"  {label}: {val:.2f}x")
            else:
                print(f"  {label}: {val:.1f}s")

    # Phase timings
    mean_phases = agg.get("mean_phase_timings", {})
    if mean_phases:
        print(f"\n  Mean phase breakdown:")
        for pk, pv in mean_phases.items():
            label = pk.replace("mean_", "").replace("_s", "").replace("_", " ")
            print(f"    {label:30s}: {pv:.3f}s")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:100]}")
        else:
            t = pc.get("wall_time_s")
            g = pc.get("gpu_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            pstr = f"{p:.4f}" if p is not None else "N/A"
            tstr = f"{t:.1f}s" if t is not None else "N/A"
            gstr = f"{g:.1f}s" if g is not None else "N/A"
            print(f"  {pc['name']}: total={tstr}, gpu={gstr}, pLDDT={pstr}")


def _print_multi_summary(all_results, seeds):
    def safe_mean(vals):
        return sum(vals) / len(vals) if vals else 0
    def safe_std(vals):
        if len(vals) < 2: return 0
        m = sum(vals) / len(vals)
        return (sum((v - m)**2 for v in vals) / len(vals))**0.5

    s_no_vals = [r["aggregate"].get("speedup_no_load", 0) for r in all_results]
    s_with_vals = [r["aggregate"].get("speedup_with_load", 0) for r in all_results]
    s_gpu_vals = [r["aggregate"].get("speedup_gpu_only", 0) for r in all_results]
    plddt_vals = [r["aggregate"].get("mean_plddt", 0) for r in all_results]
    load_vals = [r.get("load_time_s", 0) for r in all_results]

    print(f"\n{'='*60}")
    print(f"MULTI-SEED SUMMARY (n={len(seeds)})")
    print(f"{'='*60}")
    print(f"  Speedup (no load):    {safe_mean(s_no_vals):.2f}x +/- {safe_std(s_no_vals):.2f}")
    print(f"  Speedup (with load):  {safe_mean(s_with_vals):.2f}x +/- {safe_std(s_with_vals):.2f}")
    if any(s_gpu_vals):
        print(f"  Speedup (GPU only):   {safe_mean(s_gpu_vals):.2f}x +/- {safe_std(s_gpu_vals):.2f}")
    print(f"  Mean pLDDT:           {safe_mean(plddt_vals):.4f}")
    print(f"  Model load time:      {safe_mean(load_vals):.1f}s")

    # Mean phase timings across seeds
    all_phase_timings = []
    for r in all_results:
        mean_phases = r.get("aggregate", {}).get("mean_phase_timings", {})
        if mean_phases:
            all_phase_timings.append(mean_phases)

    if all_phase_timings:
        print(f"\n  Mean phase breakdown (across seeds):")
        all_keys = list(all_phase_timings[0].keys())
        for pk in all_keys:
            vals = [pt.get(pk, 0) for pt in all_phase_timings]
            label = pk.replace("mean_", "").replace("_s", "").replace("_", " ")
            print(f"    {label:30s}: {safe_mean(vals):.3f}s +/- {safe_std(vals):.3f}")

    # Per-complex
    print(f"\n{'Complex':<20} ", end="")
    for s in seeds:
        print(f"{'Seed '+str(s)+' (total)':>16} {'(gpu)':>8} ", end="")
    print()
    print("-" * (20 + 25 * len(seeds)))

    for tc_idx, tc_name in enumerate(["small_complex", "medium_complex", "large_complex"]):
        print(f"{tc_name:<20} ", end="")
        for r in all_results:
            pc = r["per_complex"][tc_idx] if tc_idx < len(r["per_complex"]) else {}
            t = pc.get("wall_time_s")
            g = pc.get("gpu_time_s")
            tstr = f"{t:.1f}s" if t is not None else "ERR"
            gstr = f"{g:.1f}s" if g is not None else "ERR"
            print(f"{tstr:>16} {gstr:>8} ", end="")
        print()

    print("\n--- FULL RESULTS ---")
    print(json.dumps(all_results, indent=2))
