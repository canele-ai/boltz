"""Evaluate TF32 matmul precision on L40S with 20 steps / 0 recycling.

Tests whether switching matmul_precision from "highest" to "high" (enabling
TF32 on Ada Lovelace) gives end-to-end speedup while maintaining quality.

Runs 3 seeds in parallel via Modal .map() for statistical reliability.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import modal

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
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_dir(str(REPO_ROOT / "research" / "eval"), remote_path="/research_eval")
)

app = modal.App("boltz-l40s-tf32-eval", image=boltz_image)


def _run_single_config(
    seed: int,
    sampling_steps: int,
    recycling_steps: int,
    matmul_precision: str,
    test_cases: list[dict],
) -> dict[str, Any]:
    """Run Boltz prediction for all test cases with a specific config + seed."""
    import torch
    torch.set_float32_matmul_precision(matmul_precision)

    results = {
        "seed": seed,
        "sampling_steps": sampling_steps,
        "recycling_steps": recycling_steps,
        "matmul_precision": matmul_precision,
        "per_complex": [],
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = f"/research_eval/test_cases/{tc['yaml'].split('/')[-1]}"

        work_dir = f"/tmp/boltz_tf32/{matmul_precision}_s{seed}_{tc_name}_{uuid.uuid4().hex[:8]}"
        os.makedirs(work_dir, exist_ok=True)

        cmd = [
            sys.executable, "/research_eval/boltz_wrapper.py",
            tc_yaml,
            "--out_dir", work_dir,
            "--sampling_steps", str(sampling_steps),
            "--recycling_steps", str(recycling_steps),
            "--diffusion_samples", "1",
            "--override",
            "--no_kernels",
            "--matmul_precision", matmul_precision,
            "--use_msa_server",
            "--seed", str(seed),
        ]

        try:
            t_start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            t_end = time.perf_counter()

            if proc.returncode == 0:
                # Parse confidence
                import glob
                conf_files = sorted(glob.glob(f"{work_dir}/boltz_results_*/predictions/*/confidence_*.json"))
                quality = {}
                if conf_files:
                    with open(conf_files[0]) as f:
                        conf = json.load(f)
                    quality = {k: conf.get(k) for k in [
                        "complex_plddt", "iptm", "confidence_score",
                        "ptm", "complex_iplddt", "complex_pde",
                    ] if k in conf}

                results["per_complex"].append({
                    "name": tc_name,
                    "wall_time_s": round(t_end - t_start, 2),
                    "quality": quality,
                    "error": None,
                })
            else:
                results["per_complex"].append({
                    "name": tc_name,
                    "wall_time_s": None,
                    "quality": {},
                    "error": proc.stderr[-500:] if proc.stderr else "unknown error",
                })
        except Exception as e:
            results["per_complex"].append({
                "name": tc_name,
                "wall_time_s": None,
                "quality": {},
                "error": str(e),
            })

    return results


@app.function(gpu="L40S", timeout=3600)
def eval_seed(config_json: str) -> str:
    """Evaluate a single seed on all test cases. Returns JSON."""
    config = json.loads(config_json)
    import yaml
    with open("/research_eval/config.yaml") as f:
        eval_config = yaml.safe_load(f)
    test_cases = eval_config.get("test_cases", [])

    result = _run_single_config(
        seed=config["seed"],
        sampling_steps=config["sampling_steps"],
        recycling_steps=config["recycling_steps"],
        matmul_precision=config["matmul_precision"],
        test_cases=test_cases,
    )
    return json.dumps(result, indent=2)


@app.local_entrypoint()
def main():
    """Run 3 seeds in parallel for two configs: highest vs high matmul precision."""
    import statistics

    seeds = [42, 123, 7]

    # Config 1: baseline 20s/0r with highest precision (matches step-reduction orbit)
    # Config 2: 20s/0r with TF32 (high precision)
    configs = []
    for precision in ["highest", "high"]:
        for seed in seeds:
            configs.append({
                "seed": seed,
                "sampling_steps": 20,
                "recycling_steps": 0,
                "matmul_precision": precision,
            })

    config_jsons = [json.dumps(c) for c in configs]

    print(f"[eval] Running {len(configs)} configs in parallel ({len(seeds)} seeds x 2 precisions)...")
    results_list = list(eval_seed.map(config_jsons))

    # Parse and aggregate
    all_results = [json.loads(r) for r in results_list]

    # Group by precision
    by_precision = {}
    for r in all_results:
        p = r["matmul_precision"]
        if p not in by_precision:
            by_precision[p] = []
        by_precision[p].append(r)

    # Baseline from config.yaml
    import yaml
    eval_dir = Path("research/eval")
    with open(eval_dir / "config.yaml") as f:
        eval_config = yaml.safe_load(f)
    baseline_time = eval_config["baseline"]["mean_wall_time_s"]
    baseline_plddt = eval_config["baseline"]["mean_plddt"]

    summary = {}
    for precision, runs in by_precision.items():
        all_times = []
        all_plddts = []
        per_complex_times = {}
        per_complex_plddts = {}

        for run in runs:
            run_time_total = 0
            run_plddts = []
            for tc in run["per_complex"]:
                if tc["error"] is None:
                    run_time_total += tc["wall_time_s"]
                    plddt = tc["quality"].get("complex_plddt")
                    if plddt is not None:
                        run_plddts.append(plddt)
                        if tc["name"] not in per_complex_plddts:
                            per_complex_plddts[tc["name"]] = []
                            per_complex_times[tc["name"]] = []
                        per_complex_plddts[tc["name"]].append(plddt)
                        per_complex_times[tc["name"]].append(tc["wall_time_s"])

            if run_plddts:
                all_times.append(run_time_total / len(run_plddts))  # mean time per complex
                all_plddts.append(sum(run_plddts) / len(run_plddts))

        if all_times:
            mean_time = statistics.mean(all_times)
            std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0
            mean_plddt = statistics.mean(all_plddts)
            std_plddt = statistics.stdev(all_plddts) if len(all_plddts) > 1 else 0
            speedup = baseline_time / mean_time if mean_time > 0 else 0
            plddt_delta = (mean_plddt - baseline_plddt) * 100

            summary[precision] = {
                "mean_time_s": round(mean_time, 2),
                "std_time_s": round(std_time, 2),
                "mean_plddt": round(mean_plddt, 4),
                "std_plddt": round(std_plddt, 4),
                "speedup": round(speedup, 2),
                "plddt_delta_pp": round(plddt_delta, 2),
                "passes_gate": plddt_delta >= -2.0,
                "per_complex": {},
            }

            for tc_name in per_complex_plddts:
                summary[precision]["per_complex"][tc_name] = {
                    "mean_time": round(statistics.mean(per_complex_times[tc_name]), 2),
                    "mean_plddt": round(statistics.mean(per_complex_plddts[tc_name]), 4),
                }

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBaseline: {baseline_time:.1f}s, pLDDT={baseline_plddt:.4f}")
    print()

    for precision, s in summary.items():
        print(f"--- matmul_precision={precision} ---")
        print(f"  Mean time: {s['mean_time_s']:.1f} +/- {s['std_time_s']:.1f}s")
        print(f"  Mean pLDDT: {s['mean_plddt']:.4f} +/- {s['std_plddt']:.4f}")
        print(f"  Speedup: {s['speedup']:.2f}x")
        print(f"  pLDDT delta: {s['plddt_delta_pp']:+.2f}pp")
        print(f"  Quality gate: {'PASS' if s['passes_gate'] else 'FAIL'}")
        for tc, d in s["per_complex"].items():
            print(f"    {tc}: time={d['mean_time']:.1f}s, pLDDT={d['mean_plddt']:.4f}")
        print()

    # Save full results
    output = {
        "all_runs": all_results,
        "summary": summary,
        "baseline": {"mean_time_s": baseline_time, "mean_plddt": baseline_plddt},
    }

    out_path = Path("orbits/l40s-kernels/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[eval] Full results saved to {out_path}")
