"""Evaluator for torch-upgrade-kernels experiment.

Builds a Modal image with torch 2.6+ and cuequivariance CUDA kernels,
then benchmarks Boltz-2 inference with kernels enabled vs disabled.

The hypothesis: upgrading torch from 2.5.1 to 2.6+ resolves the cublas
version conflict that blocks cuequivariance custom CUDA kernels. These
kernels provide fused implementations of triangular_multiplicative_update
and triangle_attention — the dominant Pairformer operations.

Usage:
    # Sanity check: verify image builds and kernels import
    modal run orbits/torch-upgrade-kernels/eval_kernels.py --mode sanity

    # Run single config with kernels enabled
    modal run orbits/torch-upgrade-kernels/eval_kernels.py --mode eval --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "enable_kernels": true}'

    # Run comparison: kernels ON vs OFF
    modal run orbits/torch-upgrade-kernels/eval_kernels.py --mode compare --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}'

    # Validate with 3 runs
    modal run orbits/torch-upgrade-kernels/eval_kernels.py --mode eval --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "enable_kernels": true}' --validate
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
# Modal setup — torch 2.6+ image with cuequivariance
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent

# Image WITH cuequivariance kernels (torch 2.6+)
boltz_image_kernels = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        # torch 2.6.0 with CUDA 12.6 — ships with cublas 12.6+
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
    )
    .pip_install(
        # Install boltz after torch to avoid torch version conflicts
        "boltz==2.2.1",
    )
    .pip_install(
        # Install cuequivariance CUDA kernels — requires cublas >= 12.5
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_kernels.py"),
        remote_path="/eval/boltz_wrapper_kernels.py",
    )
)

# Image WITHOUT cuequivariance kernels (torch 2.6+, for fair comparison)
boltz_image_no_kernels = (
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
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_kernels.py"),
        remote_path="/eval/boltz_wrapper_kernels.py",
    )
)

app = modal.App("boltz-eval-torch-upgrade-kernels")

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "compile_pairformer": False,
    "compile_structure": False,
    "compile_confidence": False,
    "compile_msa": False,
    "diffusion_samples": 1,
    "seed": 42,
    "gamma_0": 0.8,
    "noise_scale": 1.003,
    "enable_kernels": False,
}


# ---------------------------------------------------------------------------
# Helpers (run inside Modal container)
# ---------------------------------------------------------------------------


def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction."""
    wrapper = str(Path("/eval/boltz_wrapper_kernels.py"))
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
    if config.get("enable_kernels"):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    # MSA handling
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


# ---------------------------------------------------------------------------
# Modal function: sanity check (kernels image)
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=600, image=boltz_image_kernels)
def sanity_check_kernels() -> str:
    """Verify that the torch 2.6+ image builds and cuequivariance imports."""
    results = {}

    # Check torch version
    import torch
    results["torch_version"] = torch.__version__
    results["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        results["cuda_version"] = torch.version.cuda
        results["gpu_name"] = torch.cuda.get_device_name(0)

    # Check cublas
    try:
        import nvidia.cublas.cu12 as cublas
        results["cublas_version"] = getattr(cublas, "__version__", "unknown")
    except ImportError:
        results["cublas_version"] = "not found as separate package"

    # Check cuequivariance
    try:
        import cuequivariance
        results["cuequivariance_version"] = cuequivariance.__version__
        results["cuequivariance_available"] = True
    except ImportError as e:
        results["cuequivariance_available"] = False
        results["cuequivariance_error"] = str(e)

    try:
        import cuequivariance_torch
        results["cuequivariance_torch_version"] = cuequivariance_torch.__version__
        results["cuequivariance_torch_available"] = True
    except ImportError as e:
        results["cuequivariance_torch_available"] = False
        results["cuequivariance_torch_error"] = str(e)

    # Check if the kernel functions are importable
    try:
        from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
        results["triangle_multiplicative_update_available"] = True
    except ImportError as e:
        results["triangle_multiplicative_update_available"] = False
        results["triangle_multiplicative_update_error"] = str(e)

    try:
        from cuequivariance_torch.primitives.triangle import triangle_attention
        results["triangle_attention_available"] = True
    except ImportError as e:
        results["triangle_attention_available"] = False
        results["triangle_attention_error"] = str(e)

    # Check boltz
    try:
        import boltz
        results["boltz_version"] = getattr(boltz, "__version__", "unknown")
        results["boltz_available"] = True
    except ImportError as e:
        results["boltz_available"] = False
        results["boltz_error"] = str(e)

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Modal function: evaluate a configuration
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=7200, image=boltz_image_kernels)
def evaluate_with_kernels(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation with cuequivariance kernels available."""
    return _evaluate_impl(config_json, num_runs)


@app.function(gpu="L40S", timeout=7200, image=boltz_image_no_kernels)
def evaluate_no_kernels(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation without cuequivariance kernels."""
    return _evaluate_impl(config_json, num_runs)


def _evaluate_impl(config_json: str, num_runs: int = 1) -> str:
    """Shared evaluation implementation."""
    import statistics

    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": merged,
        "num_runs": num_runs,
        "per_complex": [],
        "aggregate": {},
    }

    # Log environment info
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
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[eval-kernels] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={merged['sampling_steps']}, recycle={merged['recycling_steps']}, "
                f"gamma_0={merged.get('gamma_0', 0.8)}, "
                f"kernels={'ON' if merged.get('enable_kernels') else 'OFF'}"
            )

            pred_result = _run_boltz_prediction(tc_yaml, work_dir, merged)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-kernels] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                print(f"[eval-kernels] {tc_name} run {run_idx + 1}: "
                      f"{pred_result['wall_time_s']:.1f}s, "
                      f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")
            run_qualities.append(pred_result["quality"])

        if last_error is not None:
            entry: dict[str, Any] = {
                "name": tc_name,
                "wall_time_s": None,
                "quality": {},
                "error": last_error,
            }
        else:
            median_time = statistics.median(run_times) if run_times else None
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
                "quality": merged_quality,
                "error": None,
                "run_times": run_times,
            }

        results["per_complex"].append(entry)

    # Compute aggregates
    successful = [
        r for r in results["per_complex"]
        if r["error"] is None and r["wall_time_s"] is not None
    ]

    if len(successful) < len(test_cases):
        failed_names = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        results["aggregate"] = {
            "error": f"Not all test cases succeeded. Failed: {failed_names}",
            "num_successful": len(successful),
            "num_total": len(test_cases),
            "speedup": 0,
            "passes_quality_gate": False,
        }
        return json.dumps(results, indent=2)

    if successful:
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

        results["aggregate"] = {
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
                results["aggregate"]["speedup"] = baseline_time / mean_time
            if baseline_plddt is not None and plddts:
                mean_plddt = sum(plddts) / len(plddts)
                results["aggregate"]["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
                regression = (baseline_plddt - mean_plddt) * 100.0
                results["aggregate"]["passes_quality_gate"] = regression <= 2.0

                if baseline.get("per_complex"):
                    baseline_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                    per_complex_violations = {}
                    for r in successful:
                        bl_case = baseline_by_name.get(r["name"])
                        if bl_case and bl_case.get("complex_plddt") is not None:
                            case_plddt = r["quality"].get("complex_plddt")
                            if case_plddt is None:
                                results["aggregate"]["passes_quality_gate"] = False
                                per_complex_violations[r["name"]] = "missing pLDDT"
                            else:
                                case_regression = (bl_case["complex_plddt"] - case_plddt) * 100.0
                                if case_regression > 5.0:
                                    results["aggregate"]["passes_quality_gate"] = False
                                    per_complex_violations[r["name"]] = f"-{case_regression:.1f}pp"
                    if per_complex_violations:
                        results["aggregate"]["per_complex_regression"] = per_complex_violations

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
    """Torch upgrade + kernels evaluation harness.

    Modes:
        sanity   - Verify image builds and cuequivariance imports
        eval     - Run evaluation with given config
        compare  - Run same config with kernels ON and OFF

    Usage:
        modal run orbits/torch-upgrade-kernels/eval_kernels.py --mode sanity
        modal run orbits/torch-upgrade-kernels/eval_kernels.py --mode eval --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "enable_kernels": true}'
        modal run orbits/torch-upgrade-kernels/eval_kernels.py --mode compare --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}' --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-kernels] Running sanity check...")
        result_json = sanity_check_kernels.remote()
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

        # Summarize
        all_ok = True
        for key in ["cuequivariance_available", "cuequivariance_torch_available",
                     "triangle_multiplicative_update_available", "triangle_attention_available",
                     "boltz_available"]:
            status = "OK" if result.get(key) else "FAIL"
            if not result.get(key):
                all_ok = False
            print(f"  {key}: {status}")

        if all_ok:
            print("\n[eval-kernels] Sanity check PASSED - all kernels available!")
        else:
            print("\n[eval-kernels] Sanity check FAILED - some components missing")
            sys.exit(1)

    elif mode == "eval":
        if not config:
            config = '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "enable_kernels": true}'

        cfg = json.loads(config)
        print(f"[eval-kernels] Evaluating config: {json.dumps(cfg)} (num_runs={num_runs})")

        if cfg.get("enable_kernels", False):
            result_json = evaluate_with_kernels.remote(json.dumps(cfg), num_runs=num_runs)
        else:
            result_json = evaluate_no_kernels.remote(json.dumps(cfg), num_runs=num_runs)

        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

        _print_summary(result)

    elif mode == "compare":
        if not config:
            config = '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}'

        cfg = json.loads(config)

        # Run with kernels ON
        cfg_on = {**cfg, "enable_kernels": True}
        print(f"[eval-kernels] Running with kernels ON: {json.dumps(cfg_on)} (num_runs={num_runs})")
        result_on_json = evaluate_with_kernels.remote(json.dumps(cfg_on), num_runs=num_runs)

        # Run with kernels OFF (on same torch 2.6 image, but without cuequivariance)
        cfg_off = {**cfg, "enable_kernels": False}
        print(f"[eval-kernels] Running with kernels OFF: {json.dumps(cfg_off)} (num_runs={num_runs})")
        result_off_json = evaluate_no_kernels.remote(json.dumps(cfg_off), num_runs=num_runs)

        result_on = json.loads(result_on_json)
        result_off = json.loads(result_off_json)

        print("\n" + "=" * 60)
        print("COMPARISON: Kernels ON vs OFF")
        print("=" * 60)

        print("\n--- Kernels ON ---")
        print(json.dumps(result_on, indent=2))
        _print_summary(result_on)

        print("\n--- Kernels OFF ---")
        print(json.dumps(result_off, indent=2))
        _print_summary(result_off)

        # Save comparison
        comparison = {
            "kernels_on": result_on,
            "kernels_off": result_off,
        }
        # Write to stdout as JSON for capture
        print("\n--- COMPARISON JSON ---")
        print(json.dumps(comparison, indent=2))


def _print_summary(result: dict) -> str:
    agg = result.get("aggregate", {})
    speedup = agg.get("speedup")
    plddt_delta = agg.get("plddt_delta_pp")
    passes = agg.get("passes_quality_gate")
    mean_time = agg.get("mean_wall_time_s")
    mean_plddt = agg.get("mean_plddt")

    if mean_time is not None:
        print(f"\n[eval-kernels] Mean time: {mean_time:.1f}s")
    if mean_plddt is not None:
        print(f"[eval-kernels] Mean pLDDT: {mean_plddt:.4f}")
    if speedup is not None:
        print(f"[eval-kernels] Speedup: {speedup:.2f}x")
    if plddt_delta is not None:
        print(f"[eval-kernels] pLDDT delta: {plddt_delta:+.2f} pp")
    if passes is not None:
        status = "PASS" if passes else "FAIL"
        print(f"[eval-kernels] Quality gate: {status}")
