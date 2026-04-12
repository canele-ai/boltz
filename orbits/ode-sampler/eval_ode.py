"""Evaluator for ODE sampler experiment.

Runs the standard Boltz-2 evaluation but patches gamma_0 to 0
in the diffusion process, converting the stochastic SDE sampler
to a deterministic ODE sampler (first-order Euler).

Usage:
    modal run orbits/ode-sampler/eval_ode.py --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}'
    modal run orbits/ode-sampler/eval_ode.py --config '{"sampling_steps": 20, "recycling_steps": 0}' --validate
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
        "torch==2.5.1",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "boltz==2.2.1",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(str(ORBIT_DIR / "boltz_wrapper_ode.py"), remote_path="/eval/boltz_wrapper_ode.py")
)

app = modal.App("boltz-eval-ode-sampler", image=boltz_image)

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
    "gamma_0": 0.8,  # default stochastic; 0.0 = deterministic ODE
    "noise_scale": 1.003,  # default; 1.0 for pure ODE
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
    """Run a single Boltz-2 prediction with ODE sampler support."""
    wrapper = str(Path("/eval/boltz_wrapper_ode.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--no_kernels",
        "--gamma_0", str(config.get("gamma_0", 0.8)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
    ]

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
# Modal function: evaluate
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=7200)
def evaluate(config_json: str, num_runs: int = 1) -> str:
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
                f"[eval-ode] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={merged['sampling_steps']}, recycle={merged['recycling_steps']}, "
                f"gamma_0={merged.get('gamma_0', 0.8)}"
            )

            pred_result = _run_boltz_prediction(tc_yaml, work_dir, merged)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
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
    config: str = "",
    num_runs: int = 1,
    validate: bool = False,
):
    """ODE sampler evaluation harness.

    Usage:
        modal run orbits/ode-sampler/eval_ode.py --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}'
        modal run orbits/ode-sampler/eval_ode.py --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}' --validate
    """
    if validate:
        num_runs = 3

    if not config:
        config = '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}'

    cfg = json.loads(config)
    print(f"[eval-ode] Evaluating config: {json.dumps(cfg)} (num_runs={num_runs})")
    result_json = evaluate.remote(json.dumps(cfg), num_runs=num_runs)
    result = json.loads(result_json)
    print(json.dumps(result, indent=2))

    agg = result.get("aggregate", {})
    speedup = agg.get("speedup")
    plddt_delta = agg.get("plddt_delta_pp")
    passes = agg.get("passes_quality_gate")

    if speedup is not None:
        print(f"\n[eval-ode] Speedup: {speedup:.2f}x")
    if plddt_delta is not None:
        print(f"[eval-ode] pLDDT delta: {plddt_delta:+.2f} pp")
    if passes is not None:
        status = "PASS" if passes else "FAIL"
        print(f"[eval-ode] Quality gate: {status}")
