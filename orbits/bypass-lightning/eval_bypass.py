"""Evaluator for bypass-lightning: direct model inference without Lightning Trainer.

Builds on eval-v2-winner (ODE + TF32 + bf16) and adds:
- Lightning Trainer bypass: direct predict_step calls (~6.7s saved)
- Optional CUDA warmup: pre-JIT kernel compilation (~2.2s saved)

Uses pre-cached MSAs mounted at /msa_cache for reproducible GPU-only timing.

Usage:
    # Sanity check
    modal run orbits/bypass-lightning/eval_bypass.py --mode sanity

    # Single eval (1 run)
    modal run orbits/bypass-lightning/eval_bypass.py --mode eval

    # Validation (3 runs for stable timing)
    modal run orbits/bypass-lightning/eval_bypass.py --mode eval --validate

    # Compare with and without warmup
    modal run orbits/bypass-lightning/eval_bypass.py --mode sweep
"""

from __future__ import annotations

import json
import math
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
STACKED_WRAPPER = REPO_ROOT / "research" / "solutions" / "eval-v2-winner" / "boltz_wrapper_stacked.py"

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
    .add_local_file(
        str(ORBIT_DIR / "boltz_bypass_wrapper.py"),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
    .add_local_file(
        str(STACKED_WRAPPER),
        remote_path="/eval/boltz_wrapper_stacked.py",
    )
)

app = modal.App("boltz-eval-bypass-lightning", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
    "gamma_0": 0.8,
    "noise_scale": 1.003,
    "enable_kernels": True,
    "bf16_trunk": False,
    "cuda_warmup": False,
}

# Configurations to test
SWEEP_CONFIGS = {
    # Current best from eval-v2-winner (via Lightning Trainer)
    "ODE-12/0r+TF32+bf16 (Trainer)": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "cuda_warmup": False,
    },
    # Same config but bypassing Lightning
    "ODE-12/0r+TF32+bf16 (bypass)": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "cuda_warmup": False,
    },
    # Bypass + CUDA warmup
    "ODE-12/0r+TF32+bf16 (bypass+warmup)": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "cuda_warmup": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _run_boltz_bypass(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction using the bypass-lightning wrapper."""
    wrapper = str(Path("/eval/boltz_bypass_wrapper.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--gamma_0", str(config.get("gamma_0", 0.8)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
    ]

    # Kernel control
    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    # bf16 trunk
    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    # CUDA warmup
    if config.get("cuda_warmup", False):
        cmd.append("--cuda_warmup")

    # MSA handling: use cached MSAs
    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    result: dict[str, Any] = {
        "wall_time_s": None,
        "predict_only_s": None,
        "quality": {},
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"boltz predict exited with code {proc.returncode}.\n"
                f"STDOUT: {proc.stdout[-2000:] if proc.stdout else '(empty)'}\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse phase timestamps from stderr for predict-only timing
        predict_start = None
        predict_end = None
        for line in proc.stderr.split("\n"):
            if "[PHASE] predict_start=" in line:
                predict_start = float(line.split("=")[1])
            elif "[PHASE] predict_end=" in line:
                predict_end = float(line.split("=")[1])

        if predict_start is not None and predict_end is not None:
            result["predict_only_s"] = predict_end - predict_start

        quality = _parse_confidence(out_dir, input_yaml)
        result["quality"] = quality

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


def _run_boltz_trainer(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction using the standard Lightning Trainer wrapper (for comparison)."""
    wrapper = str(Path("/eval/boltz_wrapper_stacked.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--gamma_0", str(config.get("gamma_0", 0.8)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
    ]

    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    result: dict[str, Any] = {
        "wall_time_s": None,
        "quality": {},
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"boltz predict exited with code {proc.returncode}.\n"
                f"STDOUT: {proc.stdout[-2000:] if proc.stdout else '(empty)'}\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        quality = _parse_confidence(out_dir, input_yaml)
        result["quality"] = quality

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    quality: dict[str, Any] = {}

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

    conf_path = confidence_files[0]
    with conf_path.open() as f:
        conf = json.load(f)

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

    # Also compute predict-only timing if available
    predict_times = [r.get("predict_only_s") for r in successful if r.get("predict_only_s")]
    mean_predict_only = sum(predict_times) / len(predict_times) if predict_times else None

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
        "mean_predict_only_s": mean_predict_only,
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
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_cache},
)
def evaluate_config(
    config_json: str,
    num_runs: int = 1,
    use_trainer: bool = False,
) -> str:
    """Run evaluation for a single configuration."""
    import statistics

    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    # Use cached MSAs if available
    msa_cache_path = Path("/msa_cache")
    if msa_cache_path.exists() and any(msa_cache_path.iterdir()):
        merged["msa_directory"] = str(msa_cache_path)
        print(f"[eval-bypass] Using cached MSAs from {msa_cache_path}")
    else:
        print("[eval-bypass] WARNING: No cached MSAs found, using MSA server")

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": merged,
        "num_runs": num_runs,
        "use_trainer": use_trainer,
        "per_complex": [],
        "aggregate": {},
    }

    import torch
    results["env"] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import cuequivariance_torch
        results["env"]["cuequivariance_torch"] = cuequivariance_torch.__version__
    except ImportError:
        results["env"]["cuequivariance_torch"] = None

    run_fn = _run_boltz_trainer if use_trainer else _run_boltz_bypass

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

        run_times = []
        run_predict_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            mode_str = "Trainer" if use_trainer else "bypass"
            print(
                f"[eval-bypass] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"mode={mode_str}, steps={merged['sampling_steps']}, "
                f"recycle={merged['recycling_steps']}, "
                f"gamma_0={merged.get('gamma_0', 0.8)}, "
                f"warmup={merged.get('cuda_warmup', False)}"
            )

            pred_result = run_fn(tc_yaml, work_dir, merged)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-bypass] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                if pred_result.get("predict_only_s") is not None:
                    run_predict_times.append(pred_result["predict_only_s"])
                print(f"[eval-bypass] {tc_name} run {run_idx + 1}: "
                      f"{pred_result['wall_time_s']:.1f}s"
                      f" (predict_only: {pred_result.get('predict_only_s', 'N/A')}s), "
                      f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")
            run_qualities.append(pred_result["quality"])

        if last_error is not None:
            entry: dict[str, Any] = {
                "name": tc_name,
                "wall_time_s": None,
                "predict_only_s": None,
                "quality": {},
                "error": last_error,
            }
        else:
            median_time = statistics.median(run_times) if run_times else None
            median_predict = statistics.median(run_predict_times) if run_predict_times else None
            all_plddts = [
                q["complex_plddt"] for q in run_qualities if "complex_plddt" in q
            ]
            mean_plddt_runs = (sum(all_plddts) / len(all_plddts)) if all_plddts else None

            merged_quality: dict[str, Any] = {}
            if run_qualities:
                merged_quality = dict(run_qualities[-1])
                if mean_plddt_runs is not None:
                    merged_quality["complex_plddt"] = mean_plddt_runs

            entry = {
                "name": tc_name,
                "wall_time_s": median_time,
                "predict_only_s": median_predict,
                "quality": merged_quality,
                "error": None,
                "run_times": run_times,
                "predict_times": run_predict_times,
            }

        results["per_complex"].append(entry)

    results["aggregate"] = _compute_aggregates(results, eval_config)
    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=600, volumes={"/msa_cache": msa_cache})
def sanity_check() -> str:
    """Verify environment and wrapper."""
    import torch
    results = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)

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

    # Check MSA cache
    msa_cache_path = Path("/msa_cache")
    if msa_cache_path.exists():
        msa_files = list(msa_cache_path.glob("*"))
        results["msa_cache_files"] = len(msa_files)
    else:
        results["msa_cache_files"] = 0

    # Test wrapper import
    import importlib.util
    wrapper_path = "/eval/boltz_bypass_wrapper.py"
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    if spec and spec.loader:
        results["bypass_wrapper_found"] = True
    else:
        results["bypass_wrapper_found"] = False

    # Also check stacked wrapper for comparison runs
    stacked_path = "/eval/boltz_wrapper_stacked.py"
    spec2 = importlib.util.spec_from_file_location("stacked", stacked_path)
    results["stacked_wrapper_found"] = bool(spec2 and spec2.loader)

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    config: str = "",
    num_runs: int = 1,
    validate: bool = False,
    use_trainer: bool = False,
):
    """Bypass-lightning evaluation harness.

    Modes:
        sanity  - Verify environment
        eval    - Run bypass config (default: ODE-12/0r+TF32+bf16+bypass+warmup)
        sweep   - Compare Trainer vs bypass vs bypass+warmup

    Usage:
        modal run orbits/bypass-lightning/eval_bypass.py --mode sanity
        modal run orbits/bypass-lightning/eval_bypass.py --mode eval
        modal run orbits/bypass-lightning/eval_bypass.py --mode eval --validate
        modal run orbits/bypass-lightning/eval_bypass.py --mode eval --use-trainer
        modal run orbits/bypass-lightning/eval_bypass.py --mode sweep --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-bypass] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "eval":
        if config:
            cfg = json.loads(config)
        else:
            # Default: full bypass + warmup
            cfg = SWEEP_CONFIGS["ODE-12/0r+TF32+bf16 (bypass+warmup)"]

        print(f"[eval-bypass] Evaluating: {json.dumps(cfg)} "
              f"(num_runs={num_runs}, use_trainer={use_trainer})")
        result_json = evaluate_config.remote(
            json.dumps(cfg), num_runs=num_runs, use_trainer=use_trainer
        )
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        print(f"[eval-bypass] Running sweep (num_runs={num_runs})")

        # Run Trainer baseline, bypass, and bypass+warmup
        configs_to_run = [
            ("ODE-12/0r+TF32+bf16 (Trainer)", True),
            ("ODE-12/0r+TF32+bf16 (bypass)", False),
            ("ODE-12/0r+TF32+bf16 (bypass+warmup)", False),
        ]

        all_results = {}
        for name, trainer_mode in configs_to_run:
            cfg = SWEEP_CONFIGS[name]
            print(f"\n{'='*60}")
            print(f"CONFIG: {name} (trainer={trainer_mode})")
            print(f"{'='*60}")

            result_json = evaluate_config.remote(
                json.dumps(cfg), num_runs=num_runs, use_trainer=trainer_mode
            )
            result = json.loads(result_json)
            all_results[name] = result
            _print_summary(result)

        # Comparison table
        print(f"\n{'='*70}")
        print("SWEEP COMPARISON")
        print(f"{'='*70}")
        print(f"{'Config':<40} {'Time(s)':>8} {'Pred(s)':>8} {'pLDDT':>8} {'Speedup':>8} {'Gate':>6}")
        print("-" * 80)
        for name, _ in configs_to_run:
            result = all_results[name]
            agg = result.get("aggregate", {})
            t = agg.get("mean_wall_time_s")
            po = agg.get("mean_predict_only_s")
            p = agg.get("mean_plddt")
            s = agg.get("speedup")
            g = agg.get("passes_quality_gate")
            t_str = f"{t:.1f}" if t else "ERR"
            po_str = f"{po:.1f}" if po else "N/A"
            p_str = f"{p:.4f}" if p else "ERR"
            s_str = f"{s:.2f}x" if s else "ERR"
            g_str = "PASS" if g else "FAIL"
            print(f"{name:<40} {t_str:>8} {po_str:>8} {p_str:>8} {s_str:>8} {g_str:>6}")

        print("\n--- FULL SWEEP RESULTS ---")
        print(json.dumps(all_results, indent=2))


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    mean_time = agg.get("mean_wall_time_s")
    predict_only = agg.get("mean_predict_only_s")
    mean_plddt = agg.get("mean_plddt")
    speedup = agg.get("speedup")
    plddt_delta = agg.get("plddt_delta_pp")
    passes = agg.get("passes_quality_gate")

    if mean_time is not None:
        print(f"  Mean wall time: {mean_time:.1f}s")
    if predict_only is not None:
        print(f"  Mean predict-only: {predict_only:.1f}s")
    if mean_plddt is not None:
        print(f"  Mean pLDDT: {mean_plddt:.4f}")
    if speedup is not None:
        print(f"  Speedup vs baseline: {speedup:.2f}x")
    if plddt_delta is not None:
        print(f"  pLDDT delta: {plddt_delta:+.2f} pp")
    if passes is not None:
        print(f"  Quality gate: {'PASS' if passes else 'FAIL'}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:100]}")
        else:
            t = pc.get("wall_time_s")
            po = pc.get("predict_only_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_times", [])
            times_str = ", ".join(f"{x:.1f}s" for x in times) if times else ""
            po_str = f", predict={po:.1f}s" if po else ""
            print(f"  {pc['name']}: {t:.1f}s{po_str}, pLDDT={p:.4f}"
                  + (f" [{times_str}]" if times_str else ""))
