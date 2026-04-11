"""Baseline configurations for Boltz-2 inference speedup evaluation.

Defines a set of standard configurations and runs them through the evaluator
to establish performance baselines at various quality/speed tradeoffs.

Usage:
    modal run research/eval/baselines.py
    modal run research/eval/baselines.py --only default_200step
    modal run research/eval/baselines.py --only default_50step,default_20step
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

# Import the evaluator app's image and functions.
# Insert the directory containing this file into sys.path so the import works
# both when invoked via 'modal run' from the repo root and when imported
# from any other working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluator import DEFAULT_CONFIG, app, evaluate  # noqa: E402

# ---------------------------------------------------------------------------
# Baseline configurations
# ---------------------------------------------------------------------------

BASELINES: dict[str, dict[str, Any]] = {
    "default_200step": {
        "sampling_steps": 200,
        "recycling_steps": 3,
        "matmul_precision": "highest",
        "compile_pairformer": False,
        "compile_structure": False,
        "compile_confidence": False,
        "compile_msa": False,
        "diffusion_samples": 1,
        "seed": 42,
    },
    "default_50step": {
        "sampling_steps": 50,
        "recycling_steps": 3,
        "matmul_precision": "highest",
        "compile_pairformer": False,
        "compile_structure": False,
        "compile_confidence": False,
        "compile_msa": False,
        "diffusion_samples": 1,
        "seed": 42,
    },
    "default_20step": {
        "sampling_steps": 20,
        "recycling_steps": 3,
        "matmul_precision": "highest",
        "compile_pairformer": False,
        "compile_structure": False,
        "compile_confidence": False,
        "compile_msa": False,
        "diffusion_samples": 1,
        "seed": 42,
    },
    "reduced_recycling": {
        "sampling_steps": 200,
        "recycling_steps": 1,
        "matmul_precision": "highest",
        "compile_pairformer": False,
        "compile_structure": False,
        "compile_confidence": False,
        "compile_msa": False,
        "diffusion_samples": 1,
        "seed": 42,
    },
    "fast_matmul_200step": {
        "sampling_steps": 200,
        "recycling_steps": 3,
        "matmul_precision": "high",
        "compile_pairformer": False,
        "compile_structure": False,
        "compile_confidence": False,
        "compile_msa": False,
        "diffusion_samples": 1,
        "seed": 42,
    },
    "aggressive_fast": {
        "sampling_steps": 20,
        "recycling_steps": 1,
        "matmul_precision": "medium",
        "compile_pairformer": False,
        "compile_structure": False,
        "compile_confidence": False,
        "compile_msa": False,
        "diffusion_samples": 1,
        "seed": 42,
    },
}


def _format_summary_table(all_results: dict[str, dict]) -> str:
    """Format baseline results as a readable comparison table.

    Parameters
    ----------
    all_results : dict
        Mapping from baseline name to evaluation results dict.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    header = (
        f"{'Baseline':<25s} {'Steps':>5s} {'Recycle':>7s} "
        f"{'Time(s)':>8s} {'pLDDT':>7s} {'iPTM':>7s} {'Speedup':>8s} {'Gate':>5s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Use 200-step as reference for speedup
    ref_time = None
    ref_result = all_results.get("default_200step")
    if ref_result:
        ref_agg = ref_result.get("aggregate", {})
        ref_time = ref_agg.get("mean_wall_time_s")
        ref_plddt = ref_agg.get("mean_plddt")

    for name, result in all_results.items():
        config = result.get("config", {})
        agg = result.get("aggregate", {})

        steps = config.get("sampling_steps", "?")
        recycle = config.get("recycling_steps", "?")
        mean_time = agg.get("mean_wall_time_s")
        mean_plddt = agg.get("mean_plddt")
        mean_iptm = agg.get("mean_iptm")

        time_str = f"{mean_time:.1f}" if mean_time else "ERR"
        plddt_str = f"{mean_plddt:.4f}" if mean_plddt is not None else "ERR"
        iptm_str = f"{mean_iptm:.4f}" if mean_iptm is not None else "ERR"

        speedup_str = ""
        gate_str = ""
        if ref_time and mean_time and mean_time > 0:
            speedup = ref_time / mean_time
            speedup_str = f"{speedup:.2f}x"

            if ref_plddt is not None and mean_plddt is not None:
                regression_pp = (ref_plddt - mean_plddt) * 100.0
                gate_str = "PASS" if regression_pp <= 2.0 else "FAIL"

        line = (
            f"{name:<25s} {steps:>5} {recycle:>7} "
            f"{time_str:>8s} {plddt_str:>7s} {iptm_str:>7s} "
            f"{speedup_str:>8s} {gate_str:>5s}"
        )
        lines.append(line)

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(
        description="Run baseline configurations through the Boltz evaluator.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of baseline names to run. Runs all if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save full JSON results.",
    )

    args, _unknown = parser.parse_known_args()

    # Select baselines to run
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        selected = {n: BASELINES[n] for n in names if n in BASELINES}
        missing = [n for n in names if n not in BASELINES]
        if missing:
            print(f"[baselines] WARNING: Unknown baselines skipped: {missing}")
    else:
        selected = BASELINES

    print(f"[baselines] Running {len(selected)} baseline configurations ...")
    print(f"[baselines] Configurations: {list(selected.keys())}")

    all_results: dict[str, dict] = {}

    for name, config in selected.items():
        print(f"\n{'='*60}")
        print(f"[baselines] Running: {name}")
        print(f"{'='*60}")

        t0 = time.perf_counter()

        try:
            result_json = evaluate.remote(json.dumps(config), sanity_check=False, num_runs=3)
            result = json.loads(result_json)
            all_results[name] = result
        except Exception as exc:  # noqa: BLE001
            print(f"[baselines] ERROR running {name}: {exc}")
            all_results[name] = {"config": config, "error": str(exc)}

        elapsed = time.perf_counter() - t0
        print(f"[baselines] {name} completed in {elapsed:.1f}s")

    # Print summary table
    print(f"\n{'='*60}")
    print("[baselines] Summary")
    print(f"{'='*60}\n")
    print(_format_summary_table(all_results))

    # Save full results
    output_path = args.output or str(
        Path(__file__).resolve().parent / "baseline_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[baselines] Full results saved to {output_path}")


if __name__ == "__main__":
    print("Use 'modal run research/eval/baselines.py' to run baselines on GPU.")
