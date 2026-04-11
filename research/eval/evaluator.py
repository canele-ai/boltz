"""Evaluation harness for Boltz-2 inference speedup research.

Known limitations (eval-v1):
- matmul_precision and compile_* flags in config are recorded for documentation
  but NOT applied to the subprocess. To test these, patch src/boltz/main.py directly.
- pLDDT (model confidence) is used as proxy for true lDDT.
- MSA server latency adds noise; pre-compute MSAs for precise GPU-only timing.

Runs Boltz-2 inference on a fixed test set, measures wall-clock time and
quality metrics (pLDDT, iPTM), and compares against a stored baseline.

Usage:
    modal run research/eval/evaluator.py --sanity-check
    modal run research/eval/evaluator.py --baseline
    modal run research/eval/evaluator.py --config '{"sampling_steps": 20}'

Architecture notes:

Wrapper approach for matmul_precision and compile flags
    The Boltz CLI does not expose matmul_precision or compile_* flags. Rather
    than patching Boltz internals, this harness delegates to boltz_wrapper.py,
    a thin shim that calls torch.set_float32_matmul_precision() *before*
    importing boltz and then forwards the remaining argv to boltz.main.predict.
    The evaluator passes --matmul_precision and --compile_* flags to the wrapper
    subprocess; the wrapper strips those flags before calling boltz.

MSA caching for reproducible timing
    Network round-trips to the MSA server add non-deterministic latency that
    pollutes timing comparisons. To remove this variance, configs may supply an
    "msa_directory" key pointing to a directory of pre-fetched MSA files. When
    present, the evaluator passes --msa_directory to Boltz instead of
    --use_msa_server. Recommended workflow: run the baseline once with
    use_msa_server to populate a Modal persistent volume, then set
    msa_directory in all subsequent eval configs.

Per-complex quality floor
    In addition to the aggregate mean pLDDT gate (<=2 pp regression), the
    evaluator enforces a per-complex floor: no single test case may regress by
    more than 5 pp relative to its baseline value. This prevents a strong
    improvement on easy cases from masking a catastrophic failure on a hard one.
"""

from __future__ import annotations

import argparse
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

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "boltz==2.2.1",
    )
)

app = modal.App("boltz-eval-harness", image=boltz_image)

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
}


# ---------------------------------------------------------------------------
# Helpers (run inside Modal container)
# ---------------------------------------------------------------------------


def _load_config_yaml() -> dict:
    """Load research/eval/config.yaml from the mounted volume."""
    import yaml

    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _save_config_yaml(data: dict) -> str:
    """Serialize config.yaml contents and return as string.

    We cannot write back to the mount, so we return the YAML string
    for the caller to persist.
    """
    import yaml

    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction and collect timing + quality metrics.

    Parameters
    ----------
    input_yaml : Path
        Path to the input YAML file for Boltz.
    out_dir : Path
        Directory where Boltz writes its output.
    config : dict
        Inference configuration knobs.

    Returns
    -------
    dict with keys:
        wall_time_s        - end-to-end wall time including MSA
        quality            - dict of quality metrics (plddt, iptm, ...)
        error              - None on success, error string on failure
    """
    # Delegate to boltz_wrapper.py so that matmul_precision and compile flags
    # are applied before boltz is imported (the Boltz CLI does not expose them).
    wrapper = str(Path("/eval/boltz_wrapper.py"))
    cmd = [
        sys.executable, wrapper, "predict",
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
    ]

    # MSA handling: prefer a cached directory to avoid network variance
    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Matmul precision and compile flags — handled by the wrapper
    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])
    if config.get("compile_pairformer"):
        cmd.append("--compile_pairformer")
    if config.get("compile_structure"):
        cmd.append("--compile_structure")
    if config.get("compile_confidence"):
        cmd.append("--compile_confidence")
    if config.get("compile_msa"):
        cmd.append("--compile_msa")

    result: dict[str, Any] = {
        "wall_time_s": None,
        "quality": {},
        "error": None,
    }

    try:
        # wall-clock timing: subprocess.run() blocks until the child exits,
        # so perf_counter() brackets are correct without any CUDA sync.
        t_start = time.perf_counter()

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30-minute timeout per complex
        )

        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"boltz predict exited with code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse quality metrics from confidence JSON
        quality = _parse_confidence(out_dir, input_yaml)
        result["quality"] = quality

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 1800s"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Unexpected error: {exc}"

    return result


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    """Parse Boltz confidence summary JSON from the prediction output.

    Returns
    -------
    dict with keys like complex_plddt, iptm, confidence_score, etc.
    """
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    quality: dict[str, Any] = {}

    if not results_dir.exists():
        # Try to find any prediction directory
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Prediction directory not found: {results_dir}"}

    # Find confidence JSON files
    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON files found"}

    # Use best model (model_0)
    conf_path = confidence_files[0]
    with conf_path.open() as f:
        conf = json.load(f)

    # Extract key metrics
    for key in [
        "confidence_score",
        "ptm",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
        "complex_pde",
        "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]

    return quality


# ---------------------------------------------------------------------------
# Modal function: evaluate a configuration on the full test set
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    mounts=[modal.Mount.from_local_dir(str(EVAL_DIR), remote_path="/eval")],
)
def evaluate(config_json: str, sanity_check: bool = False, num_runs: int = 1) -> str:
    """Run evaluation on the test set and return JSON results.

    Parameters
    ----------
    config_json : str
        JSON-encoded configuration dict.
    sanity_check : bool
        If True, only run the smallest test case with minimal steps.
    num_runs : int
        Number of times to run each test case. Median wall time is reported.
        For baseline runs, use num_runs=3 to get stable timing.

    Returns
    -------
    str
        JSON-encoded results.
    """
    import statistics

    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    # --- Input validation ---
    steps = merged.get("sampling_steps", 200)
    if not isinstance(steps, int) or steps <= 0:
        return json.dumps({"error": f"Invalid sampling_steps: {steps}. Must be positive integer."})
    recycle = merged.get("recycling_steps", 3)
    if not isinstance(recycle, int) or recycle < 0:
        return json.dumps({"error": f"Invalid recycling_steps: {recycle}. Must be non-negative integer."})
    if not isinstance(num_runs, int) or num_runs < 1:
        return json.dumps({"error": f"Invalid num_runs: {num_runs}. Must be >= 1."})

    # Load eval config
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    if sanity_check:
        # Only run smallest test case with reduced steps for speed
        test_cases = [test_cases[0]] if test_cases else []
        merged["sampling_steps"] = min(merged.get("sampling_steps", 200), 10)
        merged["diffusion_samples"] = 1

    results: dict[str, Any] = {
        "config": merged,
        "sanity_check": sanity_check,
        "num_runs": num_runs,
        "per_complex": [],
        "aggregate": {},
    }

    sanity_ok = {
        "evaluator_runs": False,
        "timing_positive": True,
        "quality_in_range": True,
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml_rel = tc["yaml"]
        tc_yaml = Path("/eval") / tc_yaml_rel

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
            # UUID-based work directory prevents stale results from prior runs
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[eval] Running {tc_name} run {run_idx + 1}/{num_runs} with config: "
                f"steps={merged['sampling_steps']}, "
                f"recycle={merged['recycling_steps']}"
            )

            pred_result = _run_boltz_prediction(tc_yaml, work_dir, merged)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                break  # no point retrying on a hard error

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
            run_qualities.append(pred_result["quality"])

        # Aggregate across runs: median time, mean quality
        if last_error is not None:
            entry: dict[str, Any] = {
                "name": tc_name,
                "wall_time_s": None,
                "quality": {},
                "error": last_error,
            }
        else:
            median_time = statistics.median(run_times) if run_times else None
            # Mean pLDDT across runs (sanity: should be nearly deterministic)
            all_plddts = [
                q["complex_plddt"] for q in run_qualities if "complex_plddt" in q
            ]
            mean_plddt_runs = (sum(all_plddts) / len(all_plddts)) if all_plddts else None

            merged_quality: dict[str, Any] = {}
            if run_qualities:
                merged_quality = dict(run_qualities[-1])  # keep all keys from last run
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

        # Sanity check validation
        if entry["error"] is not None:
            sanity_ok["timing_positive"] = False
            sanity_ok["quality_in_range"] = False
        else:
            sanity_ok["evaluator_runs"] = True

            if entry["wall_time_s"] is not None and entry["wall_time_s"] <= 0:
                sanity_ok["timing_positive"] = False

            plddt = entry["quality"].get("complex_plddt")
            if plddt is not None and not (0.0 <= plddt <= 1.0):
                sanity_ok["quality_in_range"] = False

    # Compute aggregates across successful runs
    successful = [
        r for r in results["per_complex"]
        if r["error"] is None and r["wall_time_s"] is not None
    ]

    # All test cases must succeed — no cherry-picking fast cases.
    # (Skip this check during sanity_check mode, which intentionally runs only 1 case.)
    if not sanity_check and len(successful) < len(test_cases):
        failed_names = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        results["aggregate"] = {
            "error": f"Not all test cases succeeded. Failed: {failed_names}",
            "num_successful": len(successful),
            "num_total": len(test_cases),
            "speedup": 0,
            "passes_quality_gate": False,
        }
        if sanity_check:
            results["sanity_checks"] = sanity_ok
            results["sanity_all_pass"] = all(sanity_ok.values())
        return json.dumps(results, indent=2)

    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts_raw = [
            r["quality"]["complex_plddt"]
            for r in successful
            if "complex_plddt" in r["quality"]
        ]
        # Filter out NaN/Inf/out-of-range before computing mean
        plddts = [
            p for p in plddts_raw
            if p is not None
            and isinstance(p, (int, float))
            and not math.isnan(p)
            and not math.isinf(p)
            and 0.0 <= p <= 1.0
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

        # Reject NaN/Inf/None/out-of-range pLDDT values
        invalid_plddts = [
            p for p in plddts_raw
            if p is None
            or not isinstance(p, (int, float))
            or (isinstance(p, float) and (math.isnan(p) or math.isinf(p)))
            or not (0.0 <= p <= 1.0)
        ]
        if invalid_plddts:
            results["aggregate"]["error"] = (
                f"Invalid pLDDT values detected: {invalid_plddts}"
            )
            results["aggregate"]["passes_quality_gate"] = False
        else:
            # Compare against baseline if available
            baseline = eval_config.get("baseline")
            if baseline is not None and baseline:
                baseline_time = baseline.get("mean_wall_time_s")
                baseline_plddt = baseline.get("mean_plddt")
                if baseline_time and mean_time > 0:
                    results["aggregate"]["speedup"] = baseline_time / mean_time
                if baseline_plddt is not None and plddts:
                    mean_plddt = sum(plddts) / len(plddts)
                    results["aggregate"]["plddt_delta_pp"] = (
                        (mean_plddt - baseline_plddt) * 100.0
                    )

                    # Quality gate: mean pLDDT regression <= 2 pp
                    regression = (baseline_plddt - mean_plddt) * 100.0
                    results["aggregate"]["passes_quality_gate"] = regression <= 2.0

                    # Per-complex quality floor: no single case may regress by more than 5 pp
                    if baseline.get("per_complex"):
                        baseline_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                        per_complex_violations = {}
                        for r in successful:
                            bl_case = baseline_by_name.get(r["name"])
                            if bl_case and bl_case.get("complex_plddt") is not None:
                                case_plddt = r["quality"].get("complex_plddt")
                                if case_plddt is None:
                                    # Missing pLDDT for a successful complex — fail the gate
                                    results["aggregate"]["passes_quality_gate"] = False
                                    per_complex_violations[r["name"]] = "missing pLDDT"
                                else:
                                    case_regression = (bl_case["complex_plddt"] - case_plddt) * 100.0
                                    if case_regression > 5.0:
                                        results["aggregate"]["passes_quality_gate"] = False
                                        per_complex_violations[r["name"]] = f"-{case_regression:.1f}pp (limit: 5pp)"
                        if per_complex_violations:
                            results["aggregate"]["per_complex_regression"] = per_complex_violations

    if sanity_check:
        results["sanity_checks"] = sanity_ok
        all_pass = all(sanity_ok.values())
        results["sanity_all_pass"] = all_pass

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Modal function: run baseline and produce baseline data
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    mounts=[modal.Mount.from_local_dir(str(EVAL_DIR), remote_path="/eval")],
)
def run_baseline() -> str:
    """Run the default 200-step baseline and return results + updated config.

    Returns
    -------
    str
        JSON with "results" (full eval output) and "updated_config_yaml"
        (string to write back to config.yaml).
    """
    # Run evaluation with default config, 3 runs for stable timing
    results_json = evaluate.local(json.dumps(DEFAULT_CONFIG), sanity_check=False, num_runs=3)
    results = json.loads(results_json)

    # Build baseline record for config.yaml
    agg = results.get("aggregate", {})
    baseline_record = {
        "config": DEFAULT_CONFIG,
        "mean_wall_time_s": agg.get("mean_wall_time_s"),
        "total_wall_time_s": agg.get("total_wall_time_s"),
        "mean_plddt": agg.get("mean_plddt"),
        "mean_iptm": agg.get("mean_iptm"),
        "per_complex": [
            {
                "name": r["name"],
                "wall_time_s": r["wall_time_s"],
                "complex_plddt": r["quality"].get("complex_plddt"),
                "iptm": r["quality"].get("iptm"),
            }
            for r in results.get("per_complex", [])
            if r["error"] is None
        ],
    }

    # Update config.yaml
    eval_config = _load_config_yaml()
    eval_config["baseline"] = baseline_record
    updated_yaml = _save_config_yaml(eval_config)

    output = {
        "results": results,
        "baseline_record": baseline_record,
        "updated_config_yaml": updated_yaml,
    }

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Local entrypoint (CLI)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(
        description="Boltz-2 inference evaluation harness",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a minimal test to verify the evaluator works.",
    )
    group.add_argument(
        "--baseline",
        action="store_true",
        help="Run with default settings and save results as the baseline.",
    )
    group.add_argument(
        "--config",
        type=str,
        help="JSON string of configuration overrides.",
    )
    group.add_argument(
        "--config-file",
        type=str,
        help="Path to a JSON file with configuration overrides.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of repeated runs per test case; median wall time is reported.",
    )

    # Parse only known args; Modal may inject its own
    args, _unknown = parser.parse_known_args()

    if args.sanity_check:
        print("[evaluator] Running sanity check ...")
        result_json = evaluate.remote(json.dumps(DEFAULT_CONFIG), sanity_check=True)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

        if result.get("sanity_all_pass"):
            print("\n[evaluator] Sanity check PASSED.")
        else:
            print("\n[evaluator] Sanity check FAILED.")
            sys.exit(1)

    elif args.baseline:
        print("[evaluator] Running baseline evaluation ...")
        result_json = run_baseline.remote()
        result = json.loads(result_json)

        # Save updated config.yaml locally
        updated_yaml = result.get("updated_config_yaml")
        if updated_yaml:
            config_path = EVAL_DIR / "config.yaml"
            with config_path.open("w") as f:
                f.write(updated_yaml)
            print(f"[evaluator] Baseline saved to {config_path}")

        print(json.dumps(result["results"], indent=2))

    else:
        # Load config from --config or --config-file
        if args.config_file:
            with open(args.config_file) as f:
                config = json.load(f)
        else:
            config = json.loads(args.config)

        num_runs = args.num_runs
        print(f"[evaluator] Evaluating config: {json.dumps(config)} (num_runs={num_runs})")
        result_json = evaluate.remote(json.dumps(config), sanity_check=False, num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

        # Print summary
        agg = result.get("aggregate", {})
        speedup = agg.get("speedup")
        plddt_delta = agg.get("plddt_delta_pp")
        passes = agg.get("passes_quality_gate")

        if speedup is not None:
            print(f"\n[evaluator] Speedup: {speedup:.2f}x")
        if plddt_delta is not None:
            print(f"[evaluator] pLDDT delta: {plddt_delta:+.2f} pp")
        if passes is not None:
            status = "PASS" if passes else "FAIL"
            print(f"[evaluator] Quality gate: {status}")


if __name__ == "__main__":
    # Allow running directly for local testing (without Modal)
    print("Use 'modal run research/eval/evaluator.py' to run on GPU.")
    print("Or import and call evaluate() / run_baseline() directly.")
