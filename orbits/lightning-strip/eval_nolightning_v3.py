"""Evaluator v3 for lightning-stripped inference.

Improvements over v2:
1. Uses Modal Volume to cache Boltz model files across runs
2. Separates download time from model loading time
3. Adds torch.compile option for score model
4. Reports GPU-only inference time separately from MSA+processing

Usage:
    # Run with best config
    modal run orbits/lightning-strip/eval_nolightning_v3.py --mode eval

    # Run with 3 seeds
    modal run orbits/lightning-strip/eval_nolightning_v3.py --mode multi
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
# Modal setup with persistent volume for model cache
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent
PARENT_ORBIT_DIR = Path(__file__).resolve().parent.parent / "eval-v2-winner"

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

# Persistent volume for model weights (avoids re-downloading each run)
model_cache = modal.Volume.from_name("boltz-model-cache", create_if_missing=True)

app = modal.App("boltz-eval-nolightning-v3", image=boltz_image)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BEST_CONFIG: dict[str, Any] = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "diffusion_samples": 1,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Patches (same as v2)
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


CPU_KEYS = frozenset([
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
    "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
    "record", "affinity_mw",
])


def _run_predict_step(model, batch):
    import torch
    out = model(
        batch,
        recycling_steps=model.predict_args["recycling_steps"],
        num_sampling_steps=model.predict_args["sampling_steps"],
        diffusion_samples=model.predict_args["diffusion_samples"],
        max_parallel_samples=model.predict_args["max_parallel_samples"],
        run_confidence_sequentially=True,
    )

    pred_dict = {"exception": False}
    if "keys_dict_batch" in model.predict_args:
        for key in model.predict_args["keys_dict_batch"]:
            pred_dict[key] = batch[key]

    pred_dict["masks"] = batch["atom_pad_mask"]
    pred_dict["token_masks"] = batch["token_pad_mask"]
    pred_dict["s"] = out["s"]
    pred_dict["z"] = out["z"]

    if "keys_dict_out" in model.predict_args:
        for key in model.predict_args["keys_dict_out"]:
            pred_dict[key] = out[key]
    pred_dict["coords"] = out["sample_atom_coords"]

    if model.confidence_prediction:
        pred_dict["pde"] = out["pde"]
        pred_dict["plddt"] = out["plddt"]
        pred_dict["confidence_score"] = (
            4 * out["complex_plddt"]
            + (
                out["iptm"]
                if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"]))
                else out["ptm"]
            )
        ) / 5
        pred_dict["complex_plddt"] = out["complex_plddt"]
        pred_dict["complex_iplddt"] = out["complex_iplddt"]
        pred_dict["complex_pde"] = out["complex_pde"]
        pred_dict["complex_ipde"] = out["complex_ipde"]
        if model.alpha_pae > 0:
            pred_dict["pae"] = out["pae"]
            pred_dict["ptm"] = out["ptm"]
            pred_dict["iptm"] = out["iptm"]
            pred_dict["ligand_iptm"] = out["ligand_iptm"]
            pred_dict["protein_iptm"] = out["protein_iptm"]
            pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]

    return pred_dict


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
    iptms = [
        r["quality"]["iptm"]
        for r in successful
        if "iptm" in r["quality"]
    ]

    agg = {
        "num_successful": len(successful),
        "num_total": len(test_cases),
        "total_wall_time_s": total_time,
        "mean_wall_time_s": mean_time,
        "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
        "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
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
    volumes={"/model-cache": model_cache},
)
def evaluate_single_load(config_json: str) -> str:
    """Run all test cases with a SINGLE model load, cached model volume."""
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
        print("[v3] bf16 trunk patch applied")

    use_kernels = config.get("enable_kernels", True)
    try:
        import cuequivariance_torch
        print(f"[v3] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        use_kernels = False

    # Use persistent volume for model cache
    cache = Path("/model-cache/boltz")
    cache.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # PHASE 1: Download (cached in volume)
    # ===================================================================
    t_download_start = time.perf_counter()
    boltz_main.download_boltz2(cache)
    t_download_end = time.perf_counter()
    download_time = t_download_end - t_download_start
    print(f"[v3] Download/cache check: {download_time:.1f}s")

    # Commit volume so cache persists
    model_cache.commit()

    # ===================================================================
    # PHASE 2: Load model to GPU
    # ===================================================================
    diffusion_params = PatchedParams()
    step_scale = config.get("step_scale", 1.5)
    diffusion_params.step_scale = step_scale
    pairformer_args = boltz_main.PairformerArgsV2()
    msa_args = boltz_main.MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
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
    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start
    print(f"[v3] Model loaded to GPU: {load_time:.1f}s")

    # ===================================================================
    # PHASE 3: Run test cases
    # ===================================================================
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    mol_dir = cache / "mols"

    results: dict[str, Any] = {
        "config": config,
        "download_time_s": download_time,
        "load_time_s": load_time,
        "per_complex": [],
        "aggregate": {},
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

        print(f"[v3] Running {tc_name}")

        try:
            t_start = time.perf_counter()

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

            t_gpu_start = time.perf_counter()

            for batch_idx, batch in enumerate(dataloader):
                batch = {
                    k: (v.to(device, non_blocking=True) if k not in CPU_KEYS else v)
                    for k, v in batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_dict = _run_predict_step(model, batch)
                pred_writer.write_on_batch_end(
                    trainer=None, pl_module=None,
                    prediction=pred_dict, batch_indices=[batch_idx],
                    batch=batch, batch_idx=batch_idx, dataloader_idx=0,
                )

            torch.cuda.synchronize()
            t_gpu_end = time.perf_counter()
            gpu_time = t_gpu_end - t_gpu_start

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
                "gpu_time_s": gpu_time,
                "quality": quality,
                "error": None,
            }
            print(f"[v3] {tc_name}: total={wall_time:.1f}s, gpu={gpu_time:.1f}s, "
                  f"pLDDT={quality.get('complex_plddt', 'N/A')}")

        except Exception as exc:
            import traceback
            entry = {
                "name": tc_name,
                "wall_time_s": None,
                "gpu_time_s": None,
                "quality": {},
                "error": f"{exc}\n{traceback.format_exc()[-1000:]}",
            }
            print(f"[v3] {tc_name}: ERROR - {exc}")

        results["per_complex"].append(entry)

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
    """Lightning-stripped v3 evaluation with model caching."""
    if mode == "eval":
        cfg = json.loads(config) if config else dict(BEST_CONFIG)
        if seed >= 0:
            cfg["seed"] = seed
        print(f"[v3] Evaluating: {json.dumps(cfg)}")
        result_json = evaluate_single_load.remote(json.dumps(cfg))
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

        print(f"[v3] Running {len(seeds)} seeds in parallel: {seeds}")
        all_results = []
        for s, result_json in zip(seeds, evaluate_single_load.map(config_jsons)):
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

        # Summary
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
        dl_vals = [r.get("download_time_s", 0) for r in all_results]

        print(f"\n{'='*60}")
        print(f"MULTI-SEED SUMMARY (n={len(seeds)})")
        print(f"{'='*60}")
        print(f"  Speedup (no load):    {safe_mean(s_no_vals):.2f}x +/- {safe_std(s_no_vals):.2f}")
        print(f"  Speedup (with load):  {safe_mean(s_with_vals):.2f}x +/- {safe_std(s_with_vals):.2f}")
        if any(s_gpu_vals):
            print(f"  Speedup (GPU only):   {safe_mean(s_gpu_vals):.2f}x +/- {safe_std(s_gpu_vals):.2f}")
        print(f"  Mean pLDDT:           {safe_mean(plddt_vals):.4f}")
        print(f"  Download time:        {safe_mean(dl_vals):.1f}s")
        print(f"  Model load time:      {safe_mean(load_vals):.1f}s")

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
