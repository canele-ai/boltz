"""Evaluator for fast model loading approaches.

Tests multiple loading strategies:
1. Baseline: load_from_checkpoint (standard subprocess)
2. Safetensors: load_file + load_state_dict (subprocess)
3. Direct GPU: map_location="cuda:0" (subprocess)
4. Persistent model: load once, predict all test cases in-process

The persistent-model approach eliminates the model loading bottleneck entirely
by keeping the model in GPU memory across all test cases. Combined with
ODE-12 + TF32 + bf16, this should give the maximum possible speedup.

Usage:
    # Save weights first (one-time):
    modal run orbits/fast-load/save_safetensors.py

    # Then run eval:
    modal run orbits/fast-load/eval_fast_load.py --mode sanity
    modal run orbits/fast-load/eval_fast_load.py --mode bench-load
    modal run orbits/fast-load/eval_fast_load.py --mode persistent
    modal run orbits/fast-load/eval_fast_load.py --mode persistent --validate
    modal run orbits/fast-load/eval_fast_load.py --mode subprocess-sf
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

ORBIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORBIT_DIR.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
    )
    .pip_install("boltz==2.2.1")
    .pip_install("safetensors")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_fast.py"),
        remote_path="/eval/boltz_wrapper_fast.py",
    )
    .add_local_file(
        str(ORBIT_DIR / "persistent_predict.py"),
        remote_path="/eval/persistent_predict.py",
    )
)

app = modal.App("boltz-eval-fast-load", image=boltz_image)

weights_volume = modal.Volume.from_name("boltz-fast-load-weights", create_if_missing=True)
msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    with Path("/eval/config.yaml").open() as f:
        return yaml.safe_load(f)


def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path) -> Optional[Path]:
    """Inject cached MSA files into input YAML. Same logic as evaluator.py."""
    import yaml
    import shutil

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

    entity_msa_map: dict[str, str] = {}
    for msa_file in msa_files:
        local_path = local_msa_dir / msa_file.name
        shutil.copy2(msa_file, local_path)
        parts = msa_file.stem.split("_")
        if len(parts) >= 2:
            entity_id = parts[-1]
            entity_msa_map[entity_id] = str(local_path)

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

    print(f"[eval] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    """Parse Boltz confidence summary JSON."""
    raw_stem = input_yaml.stem
    # Try both with and without _cached suffix
    candidates = [raw_stem]
    if raw_stem.endswith("_cached"):
        candidates.append(raw_stem[:-7])  # without _cached
    else:
        candidates.append(raw_stem + "_cached")  # with _cached

    results_dir = None
    for target_name in candidates:
        # Try exact match first
        candidate_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name
        if candidate_dir.exists():
            results_dir = candidate_dir
            break
        # Try with any prediction subdirectory
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]
                break

    # Last resort: glob for any boltz_results directory
    if results_dir is None or not results_dir.exists():
        for pattern in out_dir.glob("boltz_results_*/predictions/*/"):
            results_dir = pattern
            break

    if results_dir is None or not results_dir.exists():
        return {"error": f"Prediction directory not found under {out_dir}"}

    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON files found"}

    with confidence_files[0].open() as f:
        conf = json.load(f)

    quality: dict[str, Any] = {}
    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]

    return quality


def _compute_aggregates(results: dict, eval_config: dict) -> dict:
    """Compute aggregate metrics and quality gate."""
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
    plddts_raw = [r["quality"]["complex_plddt"] for r in successful if "complex_plddt" in r["quality"]]
    plddts = [p for p in plddts_raw
              if p is not None and isinstance(p, (int, float))
              and not math.isnan(p) and not math.isinf(p) and 0.0 <= p <= 1.0]
    iptms = [r["quality"]["iptm"] for r in successful if "iptm" in r["quality"]]

    agg = {
        "num_successful": len(successful),
        "num_total": len(test_cases),
        "total_wall_time_s": total_time,
        "mean_wall_time_s": mean_time,
        "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
        "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
    }

    baseline = eval_config.get("baseline")
    if baseline:
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
# Mode 1: Benchmark loading approaches (no full eval, just load timing)
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S", timeout=3600,
    volumes={"/weights": weights_volume, "/msa_cache": msa_volume},
)
def bench_load(num_trials: int = 3) -> str:
    """Benchmark different model loading approaches.

    Measures load time only (no inference). Returns timing for each approach.
    """
    import torch
    from pathlib import Path

    results = {"trials": num_trials, "approaches": {}}

    cache = Path.home() / ".boltz"
    ckpt_path = cache / "boltz2_conf.ckpt"

    # Ensure checkpoint is downloaded
    subprocess.run(
        [sys.executable, "-c", "from boltz.main import download; download()"],
        capture_output=True, text=True, timeout=600,
    )

    if not ckpt_path.exists():
        return json.dumps({"error": f"Checkpoint not found at {ckpt_path}"})

    # --- Approach 0: Standard load_from_checkpoint (baseline) ---
    times = []
    for i in range(num_trials):
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        from boltz.model.models.boltz2 import Boltz2
        model = Boltz2.load_from_checkpoint(
            ckpt_path, strict=True, map_location="cpu",
            predict_args={"recycling_steps": 3, "sampling_steps": 200,
                          "diffusion_samples": 1, "write_confidence_summary": True},
            ema=False,
        )
        model = model.cuda()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del model
        torch.cuda.empty_cache()
        print(f"[bench] Standard load {i+1}: {times[-1]:.2f}s")

    results["approaches"]["standard_load_from_checkpoint"] = {
        "times": times,
        "mean": sum(times) / len(times),
        "description": "load_from_checkpoint(map_location='cpu') + .cuda()",
    }

    # --- Approach 1: Standard with direct GPU ---
    times = []
    for i in range(num_trials):
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        model = Boltz2.load_from_checkpoint(
            ckpt_path, strict=True, map_location="cuda:0",
            predict_args={"recycling_steps": 3, "sampling_steps": 200,
                          "diffusion_samples": 1, "write_confidence_summary": True},
            ema=False,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del model
        torch.cuda.empty_cache()
        print(f"[bench] Direct GPU load {i+1}: {times[-1]:.2f}s")

    results["approaches"]["checkpoint_direct_gpu"] = {
        "times": times,
        "mean": sum(times) / len(times),
        "description": "load_from_checkpoint(map_location='cuda:0')",
    }

    # --- Approach 2: Safetensors from volume (CPU then GPU) ---
    sf_path = Path("/weights/boltz2.safetensors")
    hparams_path = Path("/weights/boltz2_hparams.json")

    if sf_path.exists() and hparams_path.exists():
        from safetensors.torch import load_file

        with open(hparams_path) as f:
            hparams = json.load(f)
        hparams = _filter_hparams_for_boltz2(hparams)

        times = []
        for i in range(num_trials):
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            sd = load_file(str(sf_path), device="cpu")
            model = Boltz2(**hparams)
            model.load_state_dict(sd, strict=False)
            model = model.cuda()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del model, sd
            torch.cuda.empty_cache()
            print(f"[bench] Safetensors CPU->GPU {i+1}: {times[-1]:.2f}s")

        results["approaches"]["safetensors_cpu_then_gpu"] = {
            "times": times,
            "mean": sum(times) / len(times),
            "description": "load_file(device='cpu') + Boltz2(**hparams) + load_state_dict + .cuda()",
        }

        # --- Approach 3: Safetensors direct to GPU ---
        times = []
        for i in range(num_trials):
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            sd = load_file(str(sf_path), device="cuda:0")
            model = Boltz2(**hparams)
            model.load_state_dict(sd, strict=False)
            # Model is already on GPU from state_dict, but need to move structure
            model = model.cuda()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del model, sd
            torch.cuda.empty_cache()
            print(f"[bench] Safetensors direct GPU {i+1}: {times[-1]:.2f}s")

        results["approaches"]["safetensors_direct_gpu"] = {
            "times": times,
            "mean": sum(times) / len(times),
            "description": "load_file(device='cuda:0') + Boltz2(**hparams) + load_state_dict + .cuda()",
        }

        # --- Approach 4: torch state_dict from volume (direct GPU) ---
        pt_path = Path("/weights/boltz2_state_dict.pt")
        if pt_path.exists():
            times = []
            for i in range(num_trials):
                torch.cuda.empty_cache()
                t0 = time.perf_counter()
                sd = torch.load(str(pt_path), map_location="cuda:0", weights_only=True)
                model = Boltz2(**hparams)
                model.load_state_dict(sd, strict=False)
                model = model.cuda()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                del model, sd
                torch.cuda.empty_cache()
                print(f"[bench] Torch state_dict GPU {i+1}: {times[-1]:.2f}s")

            results["approaches"]["torch_state_dict_gpu"] = {
                "times": times,
                "mean": sum(times) / len(times),
                "description": "torch.load(map_location='cuda:0') + Boltz2(**hparams) + load_state_dict",
            }
    else:
        results["note"] = "Safetensors not found on volume. Run save_safetensors.py first."

    # Summary
    print("\n--- LOAD TIME SUMMARY ---")
    for name, data in results["approaches"].items():
        print(f"  {name}: {data['mean']:.2f}s (trials: {[f'{t:.2f}' for t in data['times']]})")

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Mode 2: Persistent model evaluation (load once, predict all)
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S", timeout=7200,
    volumes={"/weights": weights_volume, "/msa_cache": msa_volume},
)
def evaluate_persistent(config_json: str, num_runs: int = 1) -> str:
    """Load model once, run all test cases in-process.

    This is the key innovation: by loading once and predicting in a loop,
    we amortize the ~20s model loading cost across all test cases.
    Per-complex time = (data processing + GPU compute) only.

    Uses the persistent_predict.py helper which calls boltz internals
    directly without going through the CLI subprocess.
    """
    import statistics
    import torch

    config = json.loads(config_json)
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    # Apply optimizations before importing boltz
    matmul_precision = config.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(matmul_precision)
    torch.set_grad_enabled(False)

    # bf16 trunk patch
    if config.get("bf16_trunk", True):
        _patch_bf16_trunk()

    # ODE mode: monkey-patch diffusion params
    gamma_0 = config.get("gamma_0", 0.0)
    noise_scale = config.get("noise_scale", 1.003)
    _patch_diffusion_params(gamma_0, noise_scale)

    # Ensure CCD data and mols are downloaded (needed for data processing)
    from boltz.main import download_boltz2
    cache = Path.home() / ".boltz"
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    # Load model ONCE
    print("[persistent] Loading model...")
    t_load_start = time.perf_counter()
    model, load_method = _load_model_fast(config)
    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start
    print(f"[persistent] Model loaded in {load_time:.2f}s via {load_method}")

    # MSA cache
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())

    results: dict[str, Any] = {
        "config": config,
        "approach": "persistent_model",
        "load_method": load_method,
        "model_load_time_s": load_time,
        "num_runs": num_runs,
        "msa_cached": use_msa_cache,
        "per_complex": [],
        "aggregate": {},
    }

    results["env"] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    sampling_steps = config.get("sampling_steps", 12)
    recycling_steps = config.get("recycling_steps", 0)
    diffusion_samples = config.get("diffusion_samples", 1)
    seed = config.get("seed", 42)

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name, "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None, "quality": {},
            })
            continue

        run_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Inject cached MSAs
            effective_yaml = tc_yaml
            if use_msa_cache:
                cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached_yaml is not None:
                    effective_yaml = cached_yaml

            print(f"[persistent] {tc_name} run {run_idx+1}/{num_runs}")

            try:
                t0 = time.perf_counter()
                _run_persistent_prediction(
                    model, effective_yaml, work_dir,
                    sampling_steps=sampling_steps,
                    recycling_steps=recycling_steps,
                    diffusion_samples=diffusion_samples,
                    seed=seed,
                    use_kernels=config.get("enable_kernels", True),
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                wall_time = t1 - t0
                quality = _parse_confidence(work_dir, effective_yaml)

                run_times.append(wall_time)
                run_qualities.append(quality)
                print(f"[persistent] {tc_name} run {run_idx+1}: {wall_time:.1f}s, "
                      f"pLDDT={quality.get('complex_plddt', 'N/A')}")

            except Exception as exc:
                last_error = f"Error: {exc}"
                import traceback
                traceback.print_exc()
                break

        if last_error:
            entry = {"name": tc_name, "wall_time_s": None, "quality": {}, "error": last_error}
        else:
            median_time = statistics.median(run_times) if run_times else None
            all_plddts = [q["complex_plddt"] for q in run_qualities if "complex_plddt" in q]
            mean_plddt = (sum(all_plddts) / len(all_plddts)) if all_plddts else None

            merged_quality = dict(run_qualities[-1]) if run_qualities else {}
            if mean_plddt is not None:
                merged_quality["complex_plddt"] = mean_plddt

            entry = {
                "name": tc_name, "wall_time_s": median_time,
                "quality": merged_quality, "error": None, "run_times": run_times,
            }

        results["per_complex"].append(entry)

    results["aggregate"] = _compute_aggregates(results, eval_config)

    # Add model load time to results for full accounting
    agg = results["aggregate"]
    if agg.get("mean_wall_time_s") is not None:
        n_complexes = agg.get("num_successful", len(test_cases))
        amortized_load = load_time / n_complexes if n_complexes > 0 else load_time
        agg["amortized_load_time_per_complex_s"] = amortized_load
        agg["mean_wall_time_with_amortized_load_s"] = agg["mean_wall_time_s"] + amortized_load

        baseline = eval_config.get("baseline", {})
        baseline_time = baseline.get("mean_wall_time_s")
        if baseline_time and agg["mean_wall_time_with_amortized_load_s"] > 0:
            agg["speedup_with_amortized_load"] = baseline_time / agg["mean_wall_time_with_amortized_load_s"]

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Mode 3: Subprocess with safetensors loading
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S", timeout=7200,
    volumes={"/weights": weights_volume, "/msa_cache": msa_volume},
)
def evaluate_subprocess_sf(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation using subprocess with safetensors loading.

    Same subprocess model as standard evaluator, but wrapper uses
    safetensors for faster loading.
    """
    import statistics

    config = json.loads(config_json)
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())

    results: dict[str, Any] = {
        "config": config,
        "approach": "subprocess_safetensors",
        "num_runs": num_runs,
        "msa_cached": use_msa_cache,
        "per_complex": [],
        "aggregate": {},
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name, "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None, "quality": {},
            })
            continue

        run_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            effective_yaml = tc_yaml
            run_config = dict(config)
            if use_msa_cache:
                cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached_yaml is not None:
                    effective_yaml = cached_yaml
                    run_config["_msa_cached"] = True

            print(f"[subprocess-sf] {tc_name} run {run_idx+1}/{num_runs}")

            pred_result = _run_subprocess_sf(effective_yaml, work_dir, run_config)

            if pred_result["error"]:
                last_error = pred_result["error"]
                print(f"[subprocess-sf] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                print(f"[subprocess-sf] {tc_name} run {run_idx+1}: "
                      f"{pred_result['wall_time_s']:.1f}s, "
                      f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")
            run_qualities.append(pred_result["quality"])

        if last_error:
            entry = {"name": tc_name, "wall_time_s": None, "quality": {}, "error": last_error}
        else:
            median_time = statistics.median(run_times) if run_times else None
            all_plddts = [q["complex_plddt"] for q in run_qualities if "complex_plddt" in q]
            mean_plddt = (sum(all_plddts) / len(all_plddts)) if all_plddts else None
            merged_quality = dict(run_qualities[-1]) if run_qualities else {}
            if mean_plddt is not None:
                merged_quality["complex_plddt"] = mean_plddt
            entry = {
                "name": tc_name, "wall_time_s": median_time,
                "quality": merged_quality, "error": None, "run_times": run_times,
            }

        results["per_complex"].append(entry)

    results["aggregate"] = _compute_aggregates(results, eval_config)
    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _patch_bf16_trunk():
    """Remove .float() upcast in triangular_mult.py for bf16 trunk."""
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
    print("[persistent] bf16 trunk patch applied")


def _patch_diffusion_params(g0: float, ns: float):
    """Monkey-patch Boltz2DiffusionParams for ODE mode."""
    import dataclasses
    import boltz.main as boltz_main

    # Use make_dataclass to avoid class-body scoping issue with parameter names
    _g0 = g0
    _ns = ns

    PatchedBoltz2DiffusionParams = dataclasses.make_dataclass(
        "PatchedBoltz2DiffusionParams",
        [
            ("gamma_0", float, dataclasses.field(default=_g0)),
            ("gamma_min", float, dataclasses.field(default=1.0)),
            ("noise_scale", float, dataclasses.field(default=_ns)),
            ("rho", float, dataclasses.field(default=7)),
            ("step_scale", float, dataclasses.field(default=1.5)),
            ("sigma_min", float, dataclasses.field(default=0.0001)),
            ("sigma_max", float, dataclasses.field(default=160.0)),
            ("sigma_data", float, dataclasses.field(default=16.0)),
            ("P_mean", float, dataclasses.field(default=-1.2)),
            ("P_std", float, dataclasses.field(default=1.5)),
            ("coordinate_augmentation", bool, dataclasses.field(default=True)),
            ("alignment_reverse_diff", bool, dataclasses.field(default=True)),
            ("synchronize_sigmas", bool, dataclasses.field(default=True)),
        ],
    )
    boltz_main.Boltz2DiffusionParams = PatchedBoltz2DiffusionParams

    PatchedBoltzDiffusionParams = dataclasses.make_dataclass(
        "PatchedBoltzDiffusionParams",
        [
            ("gamma_0", float, dataclasses.field(default=_g0)),
            ("gamma_min", float, dataclasses.field(default=1.107)),
            ("noise_scale", float, dataclasses.field(default=_ns)),
            ("rho", float, dataclasses.field(default=8)),
            ("step_scale", float, dataclasses.field(default=1.638)),
            ("sigma_min", float, dataclasses.field(default=0.0004)),
            ("sigma_max", float, dataclasses.field(default=160.0)),
            ("sigma_data", float, dataclasses.field(default=16.0)),
            ("P_mean", float, dataclasses.field(default=-1.2)),
            ("P_std", float, dataclasses.field(default=1.5)),
            ("coordinate_augmentation", bool, dataclasses.field(default=True)),
            ("alignment_reverse_diff", bool, dataclasses.field(default=True)),
            ("synchronize_sigmas", bool, dataclasses.field(default=True)),
            ("use_inference_model_cache", bool, dataclasses.field(default=True)),
        ],
    )
    boltz_main.BoltzDiffusionParams = PatchedBoltzDiffusionParams
    print(f"[persistent] Diffusion params patched: gamma_0={_g0}, noise_scale={_ns}")


def _filter_hparams_for_boltz2(hparams: dict) -> dict:
    """Filter hparams to only include parameters accepted by Boltz2.__init__.

    The checkpoint may have been saved with a newer version of Boltz that includes
    extra hyperparameters (e.g., mse_rotational_alignment). Passing these to the
    installed version's __init__ causes TypeError.
    """
    import inspect
    from boltz.model.models.boltz2 import Boltz2

    sig = inspect.signature(Boltz2.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    filtered = {}
    removed = []
    for k, v in hparams.items():
        if k in valid_params:
            filtered[k] = v
        else:
            removed.append(k)

    if removed:
        print(f"[load] Filtered out {len(removed)} unknown hparams: {removed}")

    return filtered


def _load_model_fast(config: dict):
    """Load Boltz2 model using fastest available method.

    Priority:
    1. Safetensors from volume (zero-copy mmap)
    2. State dict from volume (torch.load)
    3. Standard checkpoint (fallback)
    """
    import torch
    from boltz.model.models.boltz2 import Boltz2
    from dataclasses import asdict
    import boltz.main as boltz_main

    sampling_steps = config.get("sampling_steps", 12)
    recycling_steps = config.get("recycling_steps", 0)
    diffusion_samples = config.get("diffusion_samples", 1)
    use_kernels = config.get("enable_kernels", True)

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    diffusion_params = boltz_main.Boltz2DiffusionParams()
    pairformer_args = boltz_main.PairformerArgsV2()
    msa_args = boltz_main.MSAModuleArgs(
        subsample_msa=True,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )
    steering_args = boltz_main.BoltzSteeringParams()

    sf_path = Path("/weights/boltz2.safetensors")
    hparams_path = Path("/weights/boltz2_hparams.json")
    pt_path = Path("/weights/boltz2_state_dict.pt")

    # Try safetensors first
    if sf_path.exists() and hparams_path.exists():
        from safetensors.torch import load_file

        print("[load] Using safetensors from volume")
        with open(hparams_path) as f:
            hparams = json.load(f)

        # Override runtime params
        hparams["predict_args"] = predict_args
        hparams["diffusion_process_args"] = asdict(diffusion_params)
        hparams["pairformer_args"] = asdict(pairformer_args)
        hparams["msa_args"] = asdict(msa_args)
        hparams["steering_args"] = asdict(steering_args)
        hparams["ema"] = False
        hparams["use_kernels"] = use_kernels

        # Filter out any hparams not accepted by Boltz2.__init__
        hparams = _filter_hparams_for_boltz2(hparams)

        sd = load_file(str(sf_path), device="cuda:0")
        model = Boltz2(**hparams)
        model.load_state_dict(sd, strict=False)  # strict=False for version compat
        model = model.cuda()
        model.eval()
        torch.cuda.synchronize()
        return model, "safetensors_direct_gpu"

    # Try torch state dict
    if pt_path.exists() and hparams_path.exists():
        print("[load] Using torch state_dict from volume")
        with open(hparams_path) as f:
            hparams = json.load(f)

        hparams["predict_args"] = predict_args
        hparams["diffusion_process_args"] = asdict(diffusion_params)
        hparams["pairformer_args"] = asdict(pairformer_args)
        hparams["msa_args"] = asdict(msa_args)
        hparams["steering_args"] = asdict(steering_args)
        hparams["ema"] = False
        hparams["use_kernels"] = use_kernels

        hparams = _filter_hparams_for_boltz2(hparams)

        sd = torch.load(str(pt_path), map_location="cuda:0", weights_only=True)
        model = Boltz2(**hparams)
        model.load_state_dict(sd, strict=False)
        model = model.cuda()
        model.eval()
        torch.cuda.synchronize()
        return model, "torch_state_dict_gpu"

    # Fallback: standard checkpoint
    print("[load] Falling back to standard load_from_checkpoint")
    subprocess.run(
        [sys.executable, "-c", "from boltz.main import download; download()"],
        capture_output=True, text=True, timeout=600,
    )
    cache = Path.home() / ".boltz"
    ckpt_path = cache / "boltz2_conf.ckpt"

    model = Boltz2.load_from_checkpoint(
        ckpt_path, strict=True,
        predict_args=predict_args,
        map_location="cuda:0",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=use_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    torch.cuda.synchronize()
    return model, "standard_checkpoint_gpu"


def _run_persistent_prediction(
    model,
    input_yaml: Path,
    out_dir: Path,
    sampling_steps: int = 12,
    recycling_steps: int = 0,
    diffusion_samples: int = 1,
    seed: int = 42,
    use_kernels: bool = True,
):
    """Run prediction using a pre-loaded model (no subprocess, no reload).

    This is the core of the persistent-model approach. It calls boltz
    internals directly, replicating the pipeline from boltz.main.predict()
    but skipping model loading entirely.
    """
    import warnings
    import torch
    from pytorch_lightning import Trainer, seed_everything

    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.write.writer import BoltzWriter
    from boltz.data.types import Manifest
    import boltz.main as boltz_main

    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")

    if seed is not None:
        seed_everything(seed)

    # Update model predict_args for this run
    model.predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    cache = Path.home() / ".boltz"
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    # Replicate the boltz predict() pipeline exactly:
    # 1. Create output directory structure (like predict() does)
    input_path = Path(input_yaml)
    boltz_out_dir = out_dir / f"boltz_results_{input_path.stem}"
    boltz_out_dir.mkdir(parents=True, exist_ok=True)

    # 2. check_inputs
    data = boltz_main.check_inputs(input_path)

    # 3. process_inputs (data processing, MSA, etc.)
    boltz_main.process_inputs(
        data=data,
        out_dir=boltz_out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=8192,
    )

    # 4. Load manifest
    manifest = Manifest.load(boltz_out_dir / "processed" / "manifest.json")

    # 5. Filter (with override=True to always re-predict)
    filtered_manifest = boltz_main.filter_inputs_structure(
        manifest=manifest,
        outdir=boltz_out_dir,
        override=True,
    )

    if not filtered_manifest.records:
        raise RuntimeError(f"No records to predict for {input_yaml}")

    # 6. Build processed data paths (same as predict())
    processed_dir = boltz_out_dir / "processed"
    processed = boltz_main.BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # 7. Create data module
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir if mol_dir.exists() else None,
        num_workers=2,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )

    # 8. Create writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=boltz_out_dir / "predictions",
        output_format="mmcif",
        boltz2=True,
        write_embeddings=False,
    )

    # 9. Create trainer (reuse model on GPU)
    trainer = Trainer(
        default_root_dir=boltz_out_dir,
        strategy="auto",
        callbacks=[pred_writer],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
    )

    # 10. Run prediction (model is already on GPU -- no reload!)
    trainer.predict(
        model,
        datamodule=data_module,
        return_predictions=False,
    )


def _run_subprocess_sf(input_yaml: Path, out_dir: Path, config: dict) -> dict:
    """Run prediction using subprocess with safetensors fast-loading wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_fast.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 12)),
        "--recycling_steps", str(config.get("recycling_steps", 0)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--gamma_0", str(config.get("gamma_0", 0.0)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
        "--matmul_precision", config.get("matmul_precision", "high"),
    ]

    if config.get("bf16_trunk", True):
        cmd.append("--bf16_trunk")
    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")
    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Point to safetensors weights
    cmd.extend(["--weights_dir", "/weights"])

    result = {"wall_time_s": None, "quality": {}, "error": None}

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t1 = time.perf_counter()
        result["wall_time_s"] = t1 - t0

        if proc.returncode != 0:
            result["error"] = (
                f"Exit code {proc.returncode}\n"
                f"STDOUT: {proc.stdout[-2000:]}\n"
                f"STDERR: {proc.stderr[-2000:]}"
            )
            return result

        result["quality"] = _parse_confidence(out_dir, input_yaml)
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout 1800s"
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S", timeout=600,
    volumes={"/weights": weights_volume, "/msa_cache": msa_volume},
)
def sanity_check() -> str:
    """Verify environment, weights, and MSA cache."""
    import torch
    results = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)

    try:
        import safetensors
        results["safetensors_version"] = safetensors.__version__
    except ImportError:
        results["safetensors_version"] = None

    try:
        import cuequivariance_torch
        results["cuequivariance_torch"] = cuequivariance_torch.__version__
    except ImportError:
        results["cuequivariance_torch"] = None

    try:
        import boltz
        results["boltz"] = getattr(boltz, "__version__", "unknown")
    except ImportError:
        results["boltz"] = None

    # Check weights volume
    weights_dir = Path("/weights")
    if weights_dir.exists():
        files = list(weights_dir.iterdir())
        results["weights_files"] = [f.name for f in files]
        results["weights_sizes_mb"] = {f.name: f.stat().st_size / 1e6 for f in files}
    else:
        results["weights_files"] = []

    # Check MSA cache
    msa_dir = Path("/msa_cache")
    if msa_dir.exists():
        dirs = [d.name for d in msa_dir.iterdir() if d.is_dir()]
        results["msa_cache_targets"] = dirs
    else:
        results["msa_cache_targets"] = []

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "sanity",
    config: str = "",
    num_runs: int = 1,
    validate: bool = False,
):
    """Fast model loading evaluation harness.

    Modes:
        sanity       - Verify environment and weights
        bench-load   - Benchmark loading approaches (timing only)
        persistent   - Load once, predict all (main approach)
        subprocess-sf - Subprocess with safetensors wrapper

    Usage:
        modal run orbits/fast-load/eval_fast_load.py --mode sanity
        modal run orbits/fast-load/eval_fast_load.py --mode bench-load
        modal run orbits/fast-load/eval_fast_load.py --mode persistent
        modal run orbits/fast-load/eval_fast_load.py --mode persistent --validate
        modal run orbits/fast-load/eval_fast_load.py --mode subprocess-sf
    """
    if validate:
        num_runs = 3

    default_config = {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "noise_scale": 1.003,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "diffusion_samples": 1,
        "seed": 42,
    }

    if config:
        cfg = json.loads(config)
        default_config.update(cfg)

    if mode == "sanity":
        print("[fast-load] Running sanity check...")
        result = sanity_check.remote()
        print(result)

    elif mode == "bench-load":
        print("[fast-load] Benchmarking loading approaches...")
        result_json = bench_load.remote(num_trials=3)
        print(result_json)

    elif mode == "persistent":
        print(f"[fast-load] Persistent model eval (num_runs={num_runs})...")
        result_json = evaluate_persistent.remote(
            json.dumps(default_config), num_runs=num_runs
        )
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "subprocess-sf":
        print(f"[fast-load] Subprocess safetensors eval (num_runs={num_runs})...")
        result_json = evaluate_subprocess_sf.remote(
            json.dumps(default_config), num_runs=num_runs
        )
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    print("\n--- SUMMARY ---")
    if result.get("model_load_time_s"):
        print(f"  Model load: {result['model_load_time_s']:.1f}s ({result.get('load_method', 'unknown')})")
    if agg.get("mean_wall_time_s"):
        print(f"  Mean time (compute only): {agg['mean_wall_time_s']:.1f}s")
    if agg.get("mean_wall_time_with_amortized_load_s"):
        print(f"  Mean time (+ amortized load): {agg['mean_wall_time_with_amortized_load_s']:.1f}s")
    if agg.get("mean_plddt"):
        print(f"  Mean pLDDT: {agg['mean_plddt']:.4f}")
    if agg.get("speedup"):
        print(f"  Speedup (compute only): {agg['speedup']:.2f}x")
    if agg.get("speedup_with_amortized_load"):
        print(f"  Speedup (+ amortized load): {agg['speedup_with_amortized_load']:.2f}x")
    if agg.get("plddt_delta_pp") is not None:
        print(f"  pLDDT delta: {agg['plddt_delta_pp']:+.2f} pp")
    if agg.get("passes_quality_gate") is not None:
        print(f"  Quality gate: {'PASS' if agg['passes_quality_gate'] else 'FAIL'}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:100]}")
        else:
            t = pc.get("wall_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_times", [])
            times_str = ", ".join(f"{x:.1f}s" for x in times) if times else ""
            t_str = f"{t:.1f}s" if t else "N/A"
            p_str = f"{p:.4f}" if p else "N/A"
            print(f"  {pc['name']}: {t_str}, pLDDT={p_str}" + (f" [{times_str}]" if times_str else ""))
