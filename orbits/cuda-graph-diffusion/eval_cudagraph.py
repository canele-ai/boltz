"""Evaluator for CUDA graph capture on diffusion score model.

Tests whether torch.compile(mode="reduce-overhead") or manual CUDA graph
capture can speed up the 20-step ODE diffusion loop beyond the eval-v2-winner
baseline (1.34x).

All configurations inherit from eval-v2-winner:
- ODE sampling (gamma_0=0), 20 steps, recycling=0
- TF32 matmul precision
- bf16 trunk

Usage:
    # Sanity check
    modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode sanity

    # Single config
    modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode eval \
        --config-name reduce-overhead

    # Full sweep with 3 seeds
    modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode sweep --validate

    # Profile CPU vs GPU overhead
    modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode profile
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
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_cudagraph.py"),
        remote_path="/eval/boltz_wrapper_cudagraph.py",
    )
)

app = modal.App("boltz-eval-cuda-graph", image=boltz_image)

# ---------------------------------------------------------------------------
# Eval-v3 baseline (from config.yaml)
# ---------------------------------------------------------------------------

EVAL_V3_BASELINE = {
    "mean_wall_time_s": 47.549,
    "mean_plddt": 0.7170,
    "per_complex": [
        {"name": "small_complex", "wall_time_s": 36.5, "complex_plddt": 0.8345},
        {"name": "medium_complex", "wall_time_s": 42.2, "complex_plddt": 0.5095},
        {"name": "large_complex", "wall_time_s": 64.0, "complex_plddt": 0.8070},
    ],
}

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
    # CUDA graph options
    "compile_score_reduce_overhead": False,
    "compile_score_reduce_overhead_lazy": False,
    "compile_score_default": False,
    "compile_score_max_autotune": False,
    "manual_cuda_graph": False,
}

# Configurations to test
SWEEP_CONFIGS = {
    # Baseline: eval-v2-winner (no CUDA graph) for comparison
    "baseline-v2w": {
        "sampling_steps": 20,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "compile_score_reduce_overhead": False,
    },
    # torch.compile with reduce-overhead mode (CUDA graph via init patch)
    "reduce-overhead": {
        "sampling_steps": 20,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "compile_score_reduce_overhead": True,
    },
    # torch.compile with reduce-overhead mode (lazy, on first sample call)
    "reduce-overhead-lazy": {
        "sampling_steps": 20,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "compile_score_reduce_overhead_lazy": True,
    },
    # torch.compile with default mode (kernel fusion, no CUDA graphs)
    "compile-default": {
        "sampling_steps": 20,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "compile_score_default": True,
    },
    # torch.compile with max-autotune mode
    "compile-max-autotune": {
        "sampling_steps": 20,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "compile_score_max_autotune": True,
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


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction with the CUDA graph wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_cudagraph.py"))
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

    # CUDA graph / compile options
    if config.get("compile_score_reduce_overhead", False):
        cmd.append("--compile_score_reduce_overhead")
    elif config.get("compile_score_reduce_overhead_lazy", False):
        cmd.append("--compile_score_reduce_overhead_lazy")
    elif config.get("compile_score_default", False):
        cmd.append("--compile_score_default")
    elif config.get("compile_score_max_autotune", False):
        cmd.append("--compile_score_max_autotune")
    elif config.get("manual_cuda_graph", False):
        cmd.append("--manual_cuda_graph")

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
            timeout=3600,  # 60 min (compile can be slow)
        )
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"boltz predict exited with code {proc.returncode}.\n"
                f"STDOUT: {proc.stdout[-3000:] if proc.stdout else '(empty)'}\n"
                f"STDERR: {proc.stderr[-3000:] if proc.stderr else '(empty)'}"
            )
            return result

        quality = _parse_confidence(out_dir, input_yaml)
        result["quality"] = quality

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 3600s"
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
# Modal functions
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=10800)
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation for a single configuration."""
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

            cuda_graph_mode = "none"
            if merged.get("compile_score_reduce_overhead"):
                cuda_graph_mode = "reduce-overhead"
            elif merged.get("compile_score_reduce_overhead_lazy"):
                cuda_graph_mode = "reduce-overhead-lazy"
            elif merged.get("compile_score_default"):
                cuda_graph_mode = "compile-default"
            elif merged.get("compile_score_max_autotune"):
                cuda_graph_mode = "compile-max-autotune"
            elif merged.get("manual_cuda_graph"):
                cuda_graph_mode = "manual"

            print(
                f"[eval-cudagraph] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={merged['sampling_steps']}, recycle={merged['recycling_steps']}, "
                f"cuda_graph={cuda_graph_mode}, "
                f"TF32={merged.get('matmul_precision') == 'high'}, "
                f"bf16_trunk={merged.get('bf16_trunk', False)}"
            )

            pred_result = _run_boltz_prediction(tc_yaml, work_dir, merged)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-cudagraph] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                print(f"[eval-cudagraph] {tc_name} run {run_idx + 1}: "
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

    results["aggregate"] = _compute_aggregates(results, eval_config)
    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=600)
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

    # Test wrapper import
    import importlib.util
    wrapper_path = "/eval/boltz_wrapper_cudagraph.py"
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    if spec and spec.loader:
        results["wrapper_found"] = True
    else:
        results["wrapper_found"] = False

    # Check CUDA graph support
    if torch.cuda.is_available():
        try:
            g = torch.cuda.CUDAGraph()
            results["cuda_graph_supported"] = True
        except Exception as e:
            results["cuda_graph_supported"] = False
            results["cuda_graph_error"] = str(e)

    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=7200)
def profile_diffusion() -> str:
    """Profile the diffusion loop to measure CPU vs GPU overhead.

    This runs a single prediction with torch.profiler to capture:
    - Total wall time
    - CPU time (Python overhead, kernel dispatch)
    - GPU time (actual computation)
    - Kernel launch overhead per diffusion step
    """
    import torch
    import boltz.main as boltz_main

    results = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "profiling_notes": [],
    }

    # We cannot easily run the profiler inside the subprocess model.
    # Instead, provide guidance on how to profile manually.
    results["profiling_notes"] = [
        "To profile, run inside a Modal container with nsys or torch.profiler:",
        "  nsys profile -o diffusion_profile python boltz_wrapper_cudagraph.py ...",
        "  or add torch.profiler.profile() around the sample() call",
        "",
        "Key metrics to measure:",
        "  1. CPU overhead between kernel launches (should be ~100us/layer)",
        "  2. GPU compute per diffusion step (24 transformer layers)",
        "  3. Total CPU vs GPU time ratio",
        "",
        "If GPU utilization > 95%, CUDA graphs won't help much.",
        "If CPU overhead is significant (>5%), CUDA graphs could help.",
    ]

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    config: str = "",
    config_name: str = "",
    num_runs: int = 1,
    validate: bool = False,
):
    """CUDA graph evaluation harness.

    Modes:
        sanity  - Verify environment
        eval    - Run single config
        sweep   - Run all predefined configurations
        profile - Profile diffusion loop overhead

    Usage:
        modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode sanity
        modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode eval --config-name reduce-overhead
        modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode sweep --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-cudagraph] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "profile":
        print("[eval-cudagraph] Running profiler...")
        result_json = profile_diffusion.remote()
        print(result_json)

    elif mode == "eval":
        if config_name and config_name in SWEEP_CONFIGS:
            cfg = SWEEP_CONFIGS[config_name]
        elif config:
            cfg = json.loads(config)
        else:
            cfg = SWEEP_CONFIGS["reduce-overhead"]

        print(f"[eval-cudagraph] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        print(f"[eval-cudagraph] Running sweep of {len(SWEEP_CONFIGS)} configs (num_runs={num_runs})")

        config_names = list(SWEEP_CONFIGS.keys())
        config_jsons = [json.dumps(SWEEP_CONFIGS[name]) for name in config_names]
        num_runs_list = [num_runs] * len(config_names)

        all_results = {}
        for name, result_json in zip(
            config_names,
            evaluate_config.map(config_jsons, num_runs_list)
        ):
            result = json.loads(result_json)
            all_results[name] = result
            print(f"\n{'='*60}")
            print(f"CONFIG: {name}")
            print(f"{'='*60}")
            _print_summary(result)

        # Print comparison table
        print(f"\n{'='*60}")
        print("SWEEP COMPARISON")
        print(f"{'='*60}")
        print(f"{'Config':<30} {'Time(s)':>8} {'pLDDT':>8} {'Delta(pp)':>10} {'Speedup':>8} {'Gate':>6}")
        print("-" * 75)
        for name in config_names:
            result = all_results[name]
            agg = result.get("aggregate", {})
            t = agg.get("mean_wall_time_s")
            p = agg.get("mean_plddt")
            d = agg.get("plddt_delta_pp")
            s = agg.get("speedup")
            g = agg.get("passes_quality_gate")
            t_str = f"{t:.1f}" if t else "ERR"
            p_str = f"{p:.4f}" if p else "ERR"
            d_str = f"{d:+.2f}" if d is not None else "N/A"
            s_str = f"{s:.2f}x" if s else "ERR"
            g_str = "PASS" if g else "FAIL"
            print(f"{name:<30} {t_str:>8} {p_str:>8} {d_str:>10} {s_str:>8} {g_str:>6}")

        print("\n--- FULL SWEEP RESULTS ---")
        print(json.dumps(all_results, indent=2))


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    mean_time = agg.get("mean_wall_time_s")
    mean_plddt = agg.get("mean_plddt")
    speedup = agg.get("speedup")
    plddt_delta = agg.get("plddt_delta_pp")
    passes = agg.get("passes_quality_gate")

    if mean_time is not None:
        print(f"  Mean time: {mean_time:.1f}s")
    if mean_plddt is not None:
        print(f"  Mean pLDDT: {mean_plddt:.4f}")
    if speedup is not None:
        print(f"  Speedup: {speedup:.2f}x")
    if plddt_delta is not None:
        print(f"  pLDDT delta: {plddt_delta:+.2f} pp")
    if passes is not None:
        print(f"  Quality gate: {'PASS' if passes else 'FAIL'}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:200]}")
        else:
            t = pc.get("wall_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_times", [])
            times_str = ", ".join(f"{x:.1f}s" for x in times) if times else ""
            t_str = f"{t:.1f}s" if t else "N/A"
            p_str = f"{p:.4f}" if p else "N/A"
            print(f"  {pc['name']}: {t_str}, pLDDT={p_str}" + (f" [{times_str}]" if times_str else ""))
