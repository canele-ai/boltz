"""Evaluator for dead-end kernel approaches on eval-v5.

Tests three configurations stacked on the winning config
(ODE-12 + TF32 + bf16 + bypass Lightning + recycling_steps=3):

  A. Control — bypass-only (the winning config baseline)
  B. + BMM triangle multiplication (replacing cuequivariance kernels)
  C. + Simulated INT8 quantization (quantize-dequantize, quality impact only)

All runs: 3 seeds in parallel, CA RMSD structural comparison against ground truth.

Usage:
    modal run orbits/deadend-kernels/eval_deadend.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent
BYPASS_WRAPPER = REPO_ROOT / "orbits" / "bypass-lightning" / "boltz_bypass_wrapper.py"

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
    )
    .pip_install("boltz==2.2.1")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .pip_install("biopython>=1.83")
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(BYPASS_WRAPPER),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
)

app = modal.App("boltz-eval-deadend-kernels", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)
ground_truth = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=False)

# ---------------------------------------------------------------------------
# Chain mappings for CA RMSD: predicted_chain -> reference_chain
# ---------------------------------------------------------------------------

CHAIN_MAPPINGS = {
    "small_complex":  {"A": "A", "B": "D"},      # 1BRS
    "medium_complex": {"A": "A", "B": "B", "C": "C"},  # 1DQJ
    "large_complex":  {"A": "A", "B": "B", "C": "C", "D": "D"},  # 2DN2
}

GROUND_TRUTH_FILES = {
    "small_complex":  "1BRS.cif",
    "medium_complex": "1DQJ.cif",
    "large_complex":  "2DN2.cif",
}

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 7]

BASE_CONFIG = {
    "sampling_steps": 12,
    "recycling_steps": 3,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "cuda_warmup": True,
    "diffusion_samples": 1,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path):
    """Inject cached MSA paths into input YAML."""
    import yaml

    target_name = input_yaml.stem
    cache_dir = msa_cache_root / target_name
    if not cache_dir.exists():
        return None
    msa_files = sorted(cache_dir.glob("*.csv"))
    if not msa_files:
        return None

    with input_yaml.open() as f:
        data = yaml.safe_load(f)

    local_msa_dir = work_dir / "cached_msas"
    local_msa_dir.mkdir(parents=True, exist_ok=True)

    entity_msa_map = {}
    for msa_file in msa_files:
        local_path = local_msa_dir / msa_file.name
        shutil.copy2(msa_file, local_path)
        parts = msa_file.stem.split("_")
        if len(parts) >= 2:
            entity_msa_map[parts[-1]] = str(local_path)

    if "sequences" not in data:
        return None

    entity_idx = 0
    injected = 0
    for seq_entry in data["sequences"]:
        if "protein" in seq_entry:
            entity_key = str(entity_idx)
            if entity_key in entity_msa_map:
                seq_entry["protein"]["msa"] = entity_msa_map[entity_key]
                injected += 1
            entity_idx += 1

    if injected == 0:
        return None

    cached_yaml = work_dir / f"{target_name}_cached.yaml"
    with cached_yaml.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return cached_yaml


def _load_config_yaml() -> dict:
    import yaml
    with Path("/eval/config.yaml").open() as f:
        return yaml.safe_load(f)


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    """Parse Boltz confidence JSON."""
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Prediction directory not found: {results_dir}"}

    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON files found"}

    with confidence_files[0].open() as f:
        conf = json.load(f)

    quality = {}
    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]
    return quality


def _compare_structures(
    pred_dir: Path,
    input_yaml: Path,
    tc_name: str,
    gt_root: Path = Path("/ground_truth"),
) -> dict[str, Any]:
    """Compute CA RMSD between predicted and ground-truth structures.

    Uses BioPython's Superimposer for optimal alignment.
    """
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.SVDSuperimposer import SVDSuperimposer
    import numpy as np

    result = {"ca_rmsd": None, "error": None, "n_atoms_aligned": 0}

    gt_file = GROUND_TRUTH_FILES.get(tc_name)
    chain_map = CHAIN_MAPPINGS.get(tc_name)
    if gt_file is None or chain_map is None:
        result["error"] = f"No ground truth config for {tc_name}"
        return result

    gt_path = gt_root / gt_file
    if not gt_path.exists():
        result["error"] = f"Ground truth file not found: {gt_path}"
        return result

    # Find predicted CIF -- search all boltz_results_* directories
    pred_base = None
    for d in pred_dir.iterdir():
        if d.is_dir() and d.name.startswith("boltz_results_"):
            pred_parent = d / "predictions"
            if pred_parent.exists():
                subdirs = [sd for sd in pred_parent.iterdir() if sd.is_dir()]
                if subdirs:
                    pred_base = subdirs[0]
                    break

    if pred_base is None:
        result["error"] = f"No prediction dir found in {pred_dir}"
        return result

    pred_cifs = sorted(pred_base.glob("*.cif"))
    if not pred_cifs:
        result["error"] = "No predicted CIF files found"
        return result

    pred_path = pred_cifs[0]

    try:
        # Parse structures
        cif_parser = MMCIFParser(QUIET=True)
        gt_structure = cif_parser.get_structure("gt", str(gt_path))
        pred_structure = cif_parser.get_structure("pred", str(pred_path))

        gt_model = gt_structure[0]
        pred_model = pred_structure[0]

        # Extract CA atoms per chain and align
        pred_cas = []
        gt_cas = []

        for pred_chain_id, gt_chain_id in chain_map.items():
            # Get predicted chain CA atoms
            if pred_chain_id not in pred_model:
                continue
            pred_chain = pred_model[pred_chain_id]

            if gt_chain_id not in gt_model:
                continue
            gt_chain = gt_model[gt_chain_id]

            pred_residues = [r for r in pred_chain.get_residues() if "CA" in r]
            gt_residues = [r for r in gt_chain.get_residues() if "CA" in r]

            # Align by sequence position (take min length)
            n = min(len(pred_residues), len(gt_residues))
            for i in range(n):
                pred_cas.append(pred_residues[i]["CA"].get_vector().get_array())
                gt_cas.append(gt_residues[i]["CA"].get_vector().get_array())

        if len(pred_cas) < 3:
            result["error"] = f"Too few CA atoms for alignment: {len(pred_cas)}"
            return result

        pred_coords = np.array(pred_cas)
        gt_coords = np.array(gt_cas)

        # SVD superimposition
        sup = SVDSuperimposer()
        sup.set(gt_coords, pred_coords)
        sup.run()
        rmsd = sup.get_rms()

        result["ca_rmsd"] = float(rmsd)
        result["n_atoms_aligned"] = len(pred_cas)

    except Exception as e:
        result["error"] = f"Structure comparison failed: {e}"

    return result


def _run_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
    approach: str = "control",
) -> dict[str, Any]:
    """Run a single prediction.

    approach: "control" | "bmm" | "sim_int8"
    """
    wrapper = str(Path("/eval/boltz_bypass_wrapper.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 12)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--gamma_0", str(config.get("gamma_0", 0.0)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
    ]

    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    if config.get("cuda_warmup", False):
        cmd.append("--cuda_warmup")

    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "high")
    cmd.extend(["--matmul_precision", precision])

    # For BMM approach: we need to inject monkey-patching code
    # We write a small wrapper that patches triangle mult then calls bypass
    if approach == "bmm":
        bmm_script = _write_bmm_wrapper(out_dir)
        cmd = [sys.executable, str(bmm_script)] + cmd[2:]  # replace wrapper

    elif approach == "sim_int8":
        int8_script = _write_sim_int8_wrapper(out_dir)
        cmd = [sys.executable, str(int8_script)] + cmd[2:]

    result: dict[str, Any] = {
        "wall_time_s": None,
        "predict_only_s": None,
        "quality": {},
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"Exit code {proc.returncode}.\n"
                f"STDOUT(last 1500): {proc.stdout[-1500:] if proc.stdout else '(empty)'}\n"
                f"STDERR(last 1500): {proc.stderr[-1500:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse phase timestamps
        for line in proc.stderr.split("\n"):
            if "[PHASE] predict_start=" in line:
                predict_start = float(line.split("=")[1])
            elif "[PHASE] predict_end=" in line:
                predict_end = float(line.split("=")[1])
                if predict_start is not None:
                    result["predict_only_s"] = predict_end - predict_start

        # Try parsing confidence from both the original and effective YAML names
        quality = _parse_confidence(out_dir, input_yaml)
        if quality.get("error") and "not found" in str(quality.get("error", "")):
            # Try with _cached suffix (happens when MSAs are injected)
            cached_name = input_yaml.stem + "_cached"
            cached_yaml = input_yaml.parent / f"{cached_name}.yaml"
            quality2 = _parse_confidence(out_dir, Path(f"/fake/{cached_name}.yaml"))
            # Actually, just search for any boltz_results directory
            for d in out_dir.iterdir():
                if d.is_dir() and d.name.startswith("boltz_results_"):
                    pred_dir = d / "predictions"
                    if pred_dir.exists():
                        subdirs = [sd for sd in pred_dir.iterdir() if sd.is_dir()]
                        if subdirs:
                            conf_files = sorted(subdirs[0].glob("confidence_*.json"))
                            if conf_files:
                                import json as _json
                                with conf_files[0].open() as _f:
                                    conf = _json.load(_f)
                                quality = {}
                                for key in ["confidence_score", "ptm", "iptm", "ligand_iptm",
                                           "protein_iptm", "complex_plddt", "complex_iplddt",
                                           "complex_pde", "complex_ipde"]:
                                    if key in conf:
                                        quality[key] = conf[key]
                                break
        result["quality"] = quality

    except subprocess.TimeoutExpired:
        result["error"] = "Timed out after 600s"
    except Exception as exc:
        result["error"] = f"Unexpected: {exc}"

    return result


def _write_bmm_wrapper(work_dir: Path) -> Path:
    """Write a wrapper script that patches triangle mult with BMM then calls bypass."""
    script = work_dir / "bmm_wrapper.py"
    script.write_text('''"""BMM triangle mult wrapper — full bypass wrapper with BMM triangle mult.

Reimplements the bypass wrapper logic with BMM triangle multiplication
replacing both cuequivariance and einsum paths.
"""
import gc
import sys
import time
import argparse
import torch

_CPU_ONLY_KEYS = frozenset([
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
    "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
    "record", "affinity_mw",
])

def _transfer_batch_to_device(batch, device):
    for key in batch:
        if key not in _CPU_ONLY_KEYS:
            batch[key] = batch[key].to(device)
    return batch

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="high",
                        choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.0)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--cuda_warmup", action="store_true")
    parser.add_argument("--msa_directory", type=str, default=None)

    our_args, boltz_args = parser.parse_known_args()
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    from dataclasses import dataclass
    import boltz.main as boltz_main
    from pytorch_lightning import Trainer as _OrigTrainer

    # ODE diffusion params
    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.0
        noise_scale: float = our_args.noise_scale
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

    @dataclass
    class PatchedBoltzDiffusionParams:
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.107
        noise_scale: float = our_args.noise_scale
        rho: float = 8
        step_scale: float = 1.638
        sigma_min: float = 0.0004
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True
        use_inference_model_cache: bool = True

    boltz_main.BoltzDiffusionParams = PatchedBoltzDiffusionParams

    # BMM triangle multiplication patch (replaces both cuequivariance and einsum)
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,
    )

    def triangle_out_fn(a, b):
        B, N, K, D = a.shape
        a_t = a.permute(0, 3, 1, 2).reshape(B * D, N, K)
        b_t = b.permute(0, 3, 1, 2).reshape(B * D, N, K)
        c = torch.bmm(a_t, b_t.transpose(1, 2))
        return c.reshape(B, D, N, N).permute(0, 2, 3, 1)

    def triangle_in_fn(a, b):
        B, K, N, D = a.shape
        a_t = a.permute(0, 3, 1, 2).reshape(B * D, K, N)
        b_t = b.permute(0, 3, 1, 2).reshape(B * D, K, N)
        c = torch.bmm(a_t.transpose(1, 2), b_t)
        return c.reshape(B, D, N, N).permute(0, 2, 3, 1)

    def forward_outgoing(self, x, mask, use_kernels=False):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = triangle_out_fn(a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    def forward_incoming(self, x, mask, use_kernels=False):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = triangle_in_fn(a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    TriangleMultiplicationOutgoing.forward = forward_outgoing
    TriangleMultiplicationIncoming.forward = forward_incoming
    print("[bmm-wrapper] Triangle mult patched with batched matmul", file=sys.stderr, flush=True)

    # Disable cuequivariance kernels (we replaced them)
    boltz_args.append("--no_kernels")

    _cuda_warmup = our_args.cuda_warmup

    def _bypass_predict(trainer_self, model, datamodule=None, return_predictions=False, **kwargs):
        writer = None
        for cb in trainer_self.callbacks:
            if hasattr(cb, "write_on_batch_end"):
                writer = cb
                break
        if writer is None:
            return None

        if datamodule is not None:
            dataloader = datamodule.predict_dataloader()
        else:
            return None

        device = torch.device("cuda")
        model.eval()
        model.to(device)

        if _cuda_warmup:
            print("[bmm-wrapper] CUDA warmup...", file=sys.stderr, flush=True)
            warmup_iter = iter(dataloader)
            try:
                warmup_batch = next(warmup_iter)
                warmup_batch = _transfer_batch_to_device(warmup_batch, device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model.predict_step(warmup_batch, 0)
                torch.cuda.synchronize()
                del warmup_batch
                gc.collect()
                torch.cuda.empty_cache()
            except StopIteration:
                pass
            dataloader = datamodule.predict_dataloader()

        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                batch = _transfer_batch_to_device(batch, device)
                output = model.predict_step(batch, batch_idx)
                writer.write_on_batch_end(
                    trainer=None, pl_module=model, prediction=output,
                    batch_indices=None, batch=batch, batch_idx=batch_idx, dataloader_idx=0,
                )
        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)
        if hasattr(writer, "failed"):
            print(f"Number of failed examples: {writer.failed}", flush=True)
        return None

    _OrigTrainer.predict = _bypass_predict

    sys.argv = [sys.argv[0]] + boltz_args
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)
    boltz_main.predict()
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
''')
    return script


def _write_sim_int8_wrapper(work_dir: Path) -> Path:
    """Write a wrapper that applies simulated INT8 quantization then calls bypass."""
    script = work_dir / "sim_int8_wrapper.py"
    script.write_text('''"""Simulated INT8 wrapper — quantize-dequantize weights then delegate to bypass."""
import sys
import os
import time
import torch

def apply_simulated_int8(model):
    """Apply simulated INT8 quantization (quantize-dequantize) to all large Linear layers."""
    n_quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                w = module.weight.data
                scale = w.abs().amax(dim=1, keepdim=True) / 127.0
                scale = scale.clamp(min=1e-8)
                w_int8 = (w / scale).round().clamp(-128, 127)
                module.weight.data = (w_int8 * scale)
                n_quantized += 1
    print(f"[sim-int8] Simulated INT8 quantization: {n_quantized} layers", file=sys.stderr)
    return model

# Strategy: patch Trainer.predict (from bypass wrapper) to also apply INT8
# before the actual prediction. The bypass wrapper patches Trainer.predict
# with _bypass_predict. We need to wrap THAT to add quantization.

import importlib.util

torch.set_float32_matmul_precision("high")

# Import boltz first
import boltz.main as boltz_main

# Import bypass wrapper to get _bypass_predict set up
spec = importlib.util.spec_from_file_location("bypass", "/eval/boltz_bypass_wrapper.py")
bypass_mod = importlib.util.module_from_spec(spec)

# We need to intercept the Trainer.predict to add INT8 quantization
# The bypass wrapper replaces Trainer.predict. We further wrap it.
from pytorch_lightning import Trainer as _OrigTrainer

# Store reference to bypass_predict (set by bypass wrapper's main())
_post_bypass_predict = None

class _Int8PredictHook:
    """Wraps the bypass predict to inject INT8 quantization."""
    def __init__(self):
        self._bypass_fn = None

    def wrap(self, fn):
        self._bypass_fn = fn
        return self._call

    def _call(self, trainer_self, model, **kwargs):
        if model is not None:
            print("[sim-int8] Applying simulated INT8 quantization...", file=sys.stderr)
            t0 = time.perf_counter()
            apply_simulated_int8(model)
            t1 = time.perf_counter()
            print(f"[sim-int8] Quantization applied in {t1-t0:.2f}s", file=sys.stderr)
        return self._bypass_fn(trainer_self, model, **kwargs)

hook = _Int8PredictHook()

# Patch: after bypass wrapper sets _OrigTrainer.predict, we wrap it further
_original_predict = _OrigTrainer.predict

class _LazyInt8Wrapper:
    """Lazily wraps the final predict function (set by bypass's main)."""
    def __get__(self, obj, objtype=None):
        return lambda *args, **kwargs: self._call(obj, *args, **kwargs)

    def _call(self, trainer_self, model=None, **kwargs):
        # Apply INT8 before calling whatever predict function is set
        if model is not None:
            print("[sim-int8] Applying simulated INT8 quantization...", file=sys.stderr)
            t0 = time.perf_counter()
            apply_simulated_int8(model)
            t1 = time.perf_counter()
            print(f"[sim-int8] Quantization applied in {t1-t0:.2f}s", file=sys.stderr)
        return _original_predict(trainer_self, model=model, **kwargs)

# We can't use descriptor protocol easily. Instead, monkey-patch after bypass runs.
# Alternative approach: bypass_wrapper.main() sets Trainer.predict = _bypass_predict.
# We store a reference to that and wrap it.

# Actually, simplest approach: just run the bypass main, then the predict hasn't
# happened yet (it happens inside boltz_main.predict). So we:
# 1. Let bypass set up patches (diffusion params, bf16, kernels, Trainer.predict)
# 2. Further wrap Trainer.predict with INT8
# 3. Then call boltz_main.predict()

# But bypass's main() calls boltz_main.predict() at the end. So we need to intercept.
# Simplest: re-implement the bypass logic with INT8 added.

# Actually, let's just patch in a different way. We can use the bypass wrapper's
# approach but insert INT8 quantization into the _bypass_predict function.

import gc
import argparse

_CPU_ONLY_KEYS = frozenset([
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
    "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
    "record", "affinity_mw",
])

def _transfer_batch_to_device(batch, device):
    for key in batch:
        if key not in _CPU_ONLY_KEYS:
            batch[key] = batch[key].to(device)
    return batch

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="high",
                        choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.0)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--cuda_warmup", action="store_true")
    parser.add_argument("--msa_directory", type=str, default=None)

    our_args, boltz_args = parser.parse_known_args()
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    from dataclasses import dataclass
    import boltz.main as boltz_main
    from pytorch_lightning import Trainer as _OrigTrainer

    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.0
        noise_scale: float = our_args.noise_scale
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

    @dataclass
    class PatchedBoltzDiffusionParams:
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.107
        noise_scale: float = our_args.noise_scale
        rho: float = 8
        step_scale: float = 1.638
        sigma_min: float = 0.0004
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True
        use_inference_model_cache: bool = True

    boltz_main.BoltzDiffusionParams = PatchedBoltzDiffusionParams

    if our_args.bf16_trunk:
        from boltz.model.layers.triangular_mult import (
            TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,
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

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
    else:
        try:
            import cuequivariance_torch
        except ImportError:
            boltz_args.append("--no_kernels")

    _cuda_warmup = our_args.cuda_warmup

    def _bypass_predict_int8(trainer_self, model, datamodule=None, return_predictions=False, **kwargs):
        writer = None
        for cb in trainer_self.callbacks:
            if hasattr(cb, "write_on_batch_end"):
                writer = cb
                break
        if writer is None:
            return None

        if datamodule is not None:
            dataloader = datamodule.predict_dataloader()
        else:
            return None

        device = torch.device("cuda")
        model.eval()
        model.to(device)

        # Apply simulated INT8 BEFORE warmup
        print("[sim-int8] Applying simulated INT8 quantization...", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        apply_simulated_int8(model)
        t1 = time.perf_counter()
        print(f"[sim-int8] Quantization applied in {t1-t0:.2f}s", file=sys.stderr, flush=True)

        if _cuda_warmup:
            warmup_iter = iter(dataloader)
            try:
                warmup_batch = next(warmup_iter)
                warmup_batch = _transfer_batch_to_device(warmup_batch, device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model.predict_step(warmup_batch, 0)
                torch.cuda.synchronize()
                del warmup_batch
                gc.collect()
                torch.cuda.empty_cache()
            except StopIteration:
                pass
            dataloader = datamodule.predict_dataloader()

        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                batch = _transfer_batch_to_device(batch, device)
                output = model.predict_step(batch, batch_idx)
                writer.write_on_batch_end(
                    trainer=None, pl_module=model, prediction=output,
                    batch_indices=None, batch=batch, batch_idx=batch_idx, dataloader_idx=0,
                )
        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)
        if hasattr(writer, "failed"):
            print(f"Number of failed examples: {writer.failed}", flush=True)
        return None

    _OrigTrainer.predict = _bypass_predict_int8

    sys.argv = [sys.argv[0]] + boltz_args
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)
    boltz_main.predict()
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)

main()
''')
    return script


# ---------------------------------------------------------------------------
# Modal function: run a single (approach, seed, test_case) evaluation
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=600,
    volumes={"/msa_cache": msa_cache, "/ground_truth": ground_truth},
)
def run_single(
    approach: str,
    seed: int,
    tc_name: str,
    tc_yaml_rel: str,
) -> str:
    """Run a single prediction and return JSON results."""
    config = dict(BASE_CONFIG)
    config["seed"] = seed

    tc_yaml = Path("/eval") / tc_yaml_rel
    if not tc_yaml.exists():
        return json.dumps({"error": f"YAML not found: {tc_yaml}", "approach": approach,
                           "seed": seed, "tc_name": tc_name})

    work_dir = Path(f"/tmp/boltz_eval/{approach}_{tc_name}_{seed}_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Inject cached MSAs
    msa_cache_root = Path("/msa_cache")
    effective_yaml = tc_yaml
    if msa_cache_root.exists() and any(msa_cache_root.iterdir()):
        cached = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
        if cached is not None:
            effective_yaml = cached
            config["_msa_cached"] = True

    # For BMM, disable cuequivariance kernels
    if approach == "bmm":
        config["enable_kernels"] = False

    print(f"[run_single] approach={approach}, seed={seed}, tc={tc_name}, "
          f"steps={config['sampling_steps']}, recycle={config['recycling_steps']}")

    pred_result = _run_prediction(effective_yaml, work_dir, config, approach=approach)

    # CA RMSD comparison
    ca_rmsd_result = {"ca_rmsd": None, "error": "skipped"}
    if pred_result["error"] is None:
        ca_rmsd_result = _compare_structures(work_dir, tc_yaml, tc_name)

    return json.dumps({
        "approach": approach,
        "seed": seed,
        "tc_name": tc_name,
        "wall_time_s": pred_result["wall_time_s"],
        "predict_only_s": pred_result.get("predict_only_s"),
        "quality": pred_result["quality"],
        "ca_rmsd": ca_rmsd_result.get("ca_rmsd"),
        "ca_rmsd_n_atoms": ca_rmsd_result.get("n_atoms_aligned", 0),
        "ca_rmsd_error": ca_rmsd_result.get("error"),
        "error": pred_result["error"],
    }, indent=2)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run all three approaches across 3 seeds and 3 test cases."""
    # Hard-code test cases to avoid pyyaml dependency in local entrypoint
    test_cases = [
        {"name": "small_complex", "yaml": "test_cases/small_complex.yaml"},
        {"name": "medium_complex", "yaml": "test_cases/medium_complex.yaml"},
        {"name": "large_complex", "yaml": "test_cases/large_complex.yaml"},
    ]
    approaches = ["control", "bmm", "sim_int8"]

    # Build all jobs: 3 approaches x 3 seeds x 3 test_cases = 27 jobs
    jobs = []
    for approach in approaches:
        for seed in SEEDS:
            for tc in test_cases:
                jobs.append((approach, seed, tc["name"], tc["yaml"]))

    print(f"[deadend-kernels] Launching {len(jobs)} jobs "
          f"({len(approaches)} approaches x {len(SEEDS)} seeds x {len(test_cases)} cases)")

    # Launch all in parallel via Modal .map()
    results_iter = run_single.map(
        [j[0] for j in jobs],
        [j[1] for j in jobs],
        [j[2] for j in jobs],
        [j[3] for j in jobs],
    )

    # Collect results
    all_results = []
    for result_json in results_iter:
        r = json.loads(result_json)
        all_results.append(r)
        status = "OK" if r.get("error") is None else "ERR"
        rmsd_str = f", CA_RMSD={r['ca_rmsd']:.3f}" if r.get("ca_rmsd") is not None else ""
        plddt = r.get("quality", {}).get("complex_plddt")
        plddt_str = f", pLDDT={plddt:.4f}" if plddt is not None else ""
        print(f"  [{status}] {r['approach']}/{r['tc_name']}/seed={r['seed']}: "
              f"{r.get('wall_time_s', 'N/A')}s{plddt_str}{rmsd_str}")

    # Aggregate by approach
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for approach in approaches:
        approach_results = [r for r in all_results if r["approach"] == approach]
        errors = [r for r in approach_results if r.get("error") is not None]
        ok = [r for r in approach_results if r.get("error") is None]

        print(f"\n--- {approach.upper()} ({len(ok)} ok, {len(errors)} errors) ---")

        if errors:
            for e in errors:
                print(f"  ERROR {e['tc_name']}/seed={e['seed']}: {e['error'][:200]}")

        # Per test case aggregation
        for tc in test_cases:
            tc_results = [r for r in ok if r["tc_name"] == tc["name"]]
            if not tc_results:
                print(f"  {tc['name']}: NO RESULTS")
                continue

            times = [r["wall_time_s"] for r in tc_results if r["wall_time_s"]]
            plddts = [r["quality"]["complex_plddt"] for r in tc_results
                      if r.get("quality", {}).get("complex_plddt") is not None]
            rmsds = [r["ca_rmsd"] for r in tc_results if r.get("ca_rmsd") is not None]

            t_mean = sum(times) / len(times) if times else 0
            p_mean = sum(plddts) / len(plddts) if plddts else 0
            r_mean = sum(rmsds) / len(rmsds) if rmsds else None

            import statistics
            t_std = statistics.stdev(times) if len(times) > 1 else 0
            p_std = statistics.stdev(plddts) if len(plddts) > 1 else 0
            r_std = statistics.stdev(rmsds) if rmsds and len(rmsds) > 1 else 0

            rmsd_str = f", CA_RMSD={r_mean:.3f}+/-{r_std:.3f}" if r_mean is not None else ""
            print(f"  {tc['name']}: time={t_mean:.1f}+/-{t_std:.1f}s, "
                  f"pLDDT={p_mean:.4f}+/-{p_std:.4f}{rmsd_str}")

        # Overall aggregation
        all_times = [r["wall_time_s"] for r in ok if r["wall_time_s"]]
        all_plddts = [r["quality"]["complex_plddt"] for r in ok
                      if r.get("quality", {}).get("complex_plddt") is not None]

        if all_times and all_plddts:
            # Mean per-complex time (average across seeds, then across complexes)
            tc_mean_times = []
            tc_mean_plddts = []
            for tc in test_cases:
                tc_r = [r for r in ok if r["tc_name"] == tc["name"]]
                tc_t = [r["wall_time_s"] for r in tc_r if r["wall_time_s"]]
                tc_p = [r["quality"]["complex_plddt"] for r in tc_r
                        if r.get("quality", {}).get("complex_plddt") is not None]
                if tc_t:
                    tc_mean_times.append(sum(tc_t) / len(tc_t))
                if tc_p:
                    tc_mean_plddts.append(sum(tc_p) / len(tc_p))

            overall_time = sum(tc_mean_times) / len(tc_mean_times) if tc_mean_times else 0
            overall_plddt = sum(tc_mean_plddts) / len(tc_mean_plddts) if tc_mean_plddts else 0
            print(f"  OVERALL: mean_time={overall_time:.1f}s, mean_pLDDT={overall_plddt:.4f}")

    # Full JSON dump
    print("\n\n--- FULL JSON ---")
    print(json.dumps(all_results, indent=2))
