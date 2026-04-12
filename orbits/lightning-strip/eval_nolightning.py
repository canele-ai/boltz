"""Evaluator for lightning-stripped inference.

Compares the no-Lightning wrapper against the stacked wrapper (parent orbit)
to measure the overhead of PyTorch Lightning's Trainer.predict() path.

Usage:
    # Single config (same as parent orbit best: ODE-20/0r+TF32+bf16)
    modal run orbits/lightning-strip/eval_nolightning.py --mode eval \
        --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "matmul_precision": "high", "bf16_trunk": true}'

    # Compare Lightning vs no-Lightning
    modal run orbits/lightning-strip/eval_nolightning.py --mode compare

    # Sanity check
    modal run orbits/lightning-strip/eval_nolightning.py --mode sanity
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
# Also need the parent orbit's stacked wrapper for comparison
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
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_nolightning.py"),
        remote_path="/eval/boltz_wrapper_nolightning.py",
    )
    .add_local_file(
        str(PARENT_ORBIT_DIR / "boltz_wrapper_stacked.py"),
        remote_path="/eval/boltz_wrapper_stacked.py",
    )
)

app = modal.App("boltz-eval-nolightning", image=boltz_image)

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
}

# Best config from parent orbit
BEST_CONFIG = {
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
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _build_nolightning_cmd(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    """Build command for the no-lightning wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_nolightning.py"))
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

    return cmd


def _build_stacked_cmd(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    """Build command for the stacked wrapper (with Lightning)."""
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

    return cmd


def _run_prediction(cmd: list[str]) -> dict[str, Any]:
    """Run a prediction and return timing + quality."""
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
                f"exited with code {proc.returncode}.\n"
                f"STDOUT: {proc.stdout[-2000:] if proc.stdout else '(empty)'}\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

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

@app.function(gpu="L40S", timeout=7200)
def evaluate_config(config_json: str, num_runs: int = 1, wrapper: str = "nolightning") -> str:
    """Run evaluation with specified wrapper."""
    import statistics

    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": merged,
        "wrapper": wrapper,
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

            if wrapper == "nolightning":
                cmd = _build_nolightning_cmd(tc_yaml, work_dir, merged)
            else:
                cmd = _build_stacked_cmd(tc_yaml, work_dir, merged)

            print(
                f"[eval-{wrapper}] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={merged['sampling_steps']}, recycle={merged['recycling_steps']}"
            )

            pred_result = _run_prediction(cmd)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-{wrapper}] ERROR: {last_error[:500]}")
                break

            # Parse quality
            quality = _parse_confidence(work_dir, tc_yaml)
            pred_result["quality"] = quality

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                print(f"[eval-{wrapper}] {tc_name} run {run_idx + 1}: "
                      f"{pred_result['wall_time_s']:.1f}s, "
                      f"pLDDT={quality.get('complex_plddt', 'N/A')}")
            run_qualities.append(quality)

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


@app.function(gpu="L40S", timeout=7200)
def compare_wrappers(config_json: str, num_runs: int = 1) -> str:
    """Run the same config with both wrappers on the same GPU for fair comparison.

    This runs sequentially on the same container to eliminate cross-container variance.
    """
    # Run stacked (with Lightning) first
    print("=" * 60)
    print("PHASE 1: Running WITH Lightning (stacked wrapper)")
    print("=" * 60)
    stacked_json = evaluate_config.local(config_json, num_runs=num_runs, wrapper="stacked")
    stacked = json.loads(stacked_json)

    # Run no-lightning second
    print("=" * 60)
    print("PHASE 2: Running WITHOUT Lightning (nolightning wrapper)")
    print("=" * 60)
    nolightning_json = evaluate_config.local(config_json, num_runs=num_runs, wrapper="nolightning")
    nolightning = json.loads(nolightning_json)

    # Compare
    comparison = {
        "config": json.loads(config_json),
        "stacked": stacked,
        "nolightning": nolightning,
        "comparison": {},
    }

    s_agg = stacked.get("aggregate", {})
    n_agg = nolightning.get("aggregate", {})

    s_time = s_agg.get("mean_wall_time_s")
    n_time = n_agg.get("mean_wall_time_s")

    if s_time and n_time:
        comparison["comparison"]["stacked_mean_time"] = s_time
        comparison["comparison"]["nolightning_mean_time"] = n_time
        comparison["comparison"]["time_saved_s"] = s_time - n_time
        comparison["comparison"]["time_saved_pct"] = ((s_time - n_time) / s_time) * 100
        comparison["comparison"]["nolightning_vs_stacked_speedup"] = s_time / n_time

    s_plddt = s_agg.get("mean_plddt")
    n_plddt = n_agg.get("mean_plddt")
    if s_plddt and n_plddt:
        comparison["comparison"]["stacked_plddt"] = s_plddt
        comparison["comparison"]["nolightning_plddt"] = n_plddt
        comparison["comparison"]["plddt_diff_pp"] = (n_plddt - s_plddt) * 100

    return json.dumps(comparison, indent=2)


@app.function(gpu="L40S", timeout=600)
def sanity_check() -> str:
    """Verify environment and both wrappers."""
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

    # Check both wrappers exist
    results["nolightning_wrapper"] = Path("/eval/boltz_wrapper_nolightning.py").exists()
    results["stacked_wrapper"] = Path("/eval/boltz_wrapper_stacked.py").exists()

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
    seed: int = -1,
):
    """Lightning-stripped evaluation harness.

    Modes:
        sanity   - Verify environment
        eval     - Run nolightning wrapper
        compare  - Run both wrappers on same GPU for fair comparison
        multi    - Run nolightning with multiple seeds (parallel containers)

    Usage:
        modal run orbits/lightning-strip/eval_nolightning.py --mode sanity
        modal run orbits/lightning-strip/eval_nolightning.py --mode eval
        modal run orbits/lightning-strip/eval_nolightning.py --mode compare --validate
        modal run orbits/lightning-strip/eval_nolightning.py --mode multi --seed 42
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-nolightning] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "eval":
        cfg = json.loads(config) if config else dict(BEST_CONFIG)
        if seed >= 0:
            cfg["seed"] = seed
        print(f"[eval-nolightning] Evaluating nolightning: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs, wrapper="nolightning")
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "compare":
        cfg = json.loads(config) if config else dict(BEST_CONFIG)
        print(f"[eval-nolightning] Comparing wrappers: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = compare_wrappers.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

        comp = result.get("comparison", {})
        if comp:
            print(f"\n{'='*60}")
            print("COMPARISON: Lightning vs No-Lightning")
            print(f"{'='*60}")
            print(f"  Stacked (Lightning) mean time: {comp.get('stacked_mean_time', 'N/A'):.1f}s")
            print(f"  No-Lightning mean time:        {comp.get('nolightning_mean_time', 'N/A'):.1f}s")
            print(f"  Time saved:                    {comp.get('time_saved_s', 'N/A'):.1f}s ({comp.get('time_saved_pct', 'N/A'):.1f}%)")
            print(f"  No-Lightning vs Stacked:       {comp.get('nolightning_vs_stacked_speedup', 'N/A'):.3f}x")
            if comp.get("plddt_diff_pp") is not None:
                print(f"  pLDDT difference:              {comp.get('plddt_diff_pp', 'N/A'):+.2f} pp")

    elif mode == "multi":
        # Run with 3 seeds in parallel
        seeds = [42, 123, 7]
        cfg_base = json.loads(config) if config else dict(BEST_CONFIG)

        config_jsons = []
        for s in seeds:
            c = dict(cfg_base)
            c["seed"] = s
            config_jsons.append(json.dumps(c))

        print(f"[eval-nolightning] Running {len(seeds)} seeds in parallel: {seeds}")
        all_results = []
        for s, result_json in zip(
            seeds,
            evaluate_config.map(config_jsons, [num_runs]*len(seeds), ["nolightning"]*len(seeds))
        ):
            result = json.loads(result_json)
            all_results.append(result)
            agg = result.get("aggregate", {})
            print(f"  Seed {s}: time={agg.get('mean_wall_time_s', 'N/A'):.1f}s, "
                  f"pLDDT={agg.get('mean_plddt', 'N/A'):.4f}, "
                  f"speedup={agg.get('speedup', 'N/A'):.2f}x")

        # Compute mean across seeds
        speedups = [r["aggregate"]["speedup"] for r in all_results if r["aggregate"].get("speedup")]
        plddts = [r["aggregate"]["mean_plddt"] for r in all_results if r["aggregate"].get("mean_plddt")]
        times = [r["aggregate"]["mean_wall_time_s"] for r in all_results if r["aggregate"].get("mean_wall_time_s")]

        if speedups:
            mean_speedup = sum(speedups) / len(speedups)
            std_speedup = (sum((s - mean_speedup)**2 for s in speedups) / len(speedups))**0.5
            mean_plddt = sum(plddts) / len(plddts)
            mean_time = sum(times) / len(times)
            print(f"\n{'='*60}")
            print(f"MULTI-SEED SUMMARY (n={len(seeds)})")
            print(f"{'='*60}")
            print(f"  Mean speedup: {mean_speedup:.2f}x +/- {std_speedup:.2f}")
            print(f"  Mean pLDDT:   {mean_plddt:.4f}")
            print(f"  Mean time:    {mean_time:.1f}s")

        # Dump all results
        print("\n--- ALL RESULTS ---")
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
            print(f"  {pc['name']}: ERROR - {pc['error'][:100]}")
        else:
            t = pc.get("wall_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_times", [])
            times_str = ", ".join(f"{x:.1f}s" for x in times) if times else ""
            pstr = f"{p:.4f}" if p is not None else "N/A"
            tstr = f"{t:.1f}s" if t is not None else "N/A"
            print(f"  {pc['name']}: {tstr}, pLDDT={pstr}" + (f" [{times_str}]" if times_str else ""))
