"""Sweep diffusion sampling steps in parallel via Modal.

Runs multiple configurations concurrently on separate L40S GPUs,
then collects results for Pareto analysis.

Usage:
    cd /home/liambai/code/boltz/.worktrees/step-reduction
    /home/liambai/code/boltz/.venv/bin/modal run orbits/step-reduction/sweep_steps.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Resolve EVAL_DIR at MODULE SCOPE — only works on the LOCAL machine.
# Inside the Modal container, the eval directory is mounted at /eval.
# ---------------------------------------------------------------------------

# Try relative path from script location first, then absolute fallback.
_script_dir = Path(__file__).resolve().parent
_candidates = [
    _script_dir / ".." / ".." / "research" / "eval",
    Path("/home/liambai/code/boltz/.worktrees/step-reduction/research/eval"),
    Path("/home/liambai/code/boltz/research/eval"),
]
EVAL_DIR = None
for c in _candidates:
    if c.resolve().exists():
        EVAL_DIR = c.resolve()
        break

if EVAL_DIR is None:
    # Last resort: we're inside the container, EVAL_DIR won't be used for image build
    EVAL_DIR = Path("/eval")

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
)

app = modal.App("boltz-step-sweep", image=boltz_image)


@app.function(gpu="L40S", timeout=7200)
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation for a single config on a remote GPU."""
    import json as _json
    import math
    import statistics
    import subprocess
    import time as _time
    import uuid

    import yaml

    DEFAULT_CONFIG = {
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

    config = _json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        eval_config = yaml.safe_load(f)

    test_cases = eval_config.get("test_cases", [])

    results = {
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
                "error": f"YAML not found: {tc_yaml}",
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
                f"[sweep] {tc_name} run {run_idx+1}/{num_runs} "
                f"steps={merged['sampling_steps']} recycle={merged['recycling_steps']}"
            )

            wrapper = "/eval/boltz_wrapper.py"
            cmd = [
                sys.executable, wrapper,
                str(tc_yaml),
                "--out_dir", str(work_dir),
                "--sampling_steps", str(merged.get("sampling_steps", 200)),
                "--recycling_steps", str(merged.get("recycling_steps", 3)),
                "--diffusion_samples", str(merged.get("diffusion_samples", 1)),
                "--override",
                "--no_kernels",
            ]

            msa_dir = merged.get("msa_directory")
            if msa_dir:
                cmd.extend(["--msa_directory", str(msa_dir)])
            else:
                cmd.append("--use_msa_server")

            seed = merged.get("seed")
            if seed is not None:
                cmd.extend(["--seed", str(seed)])

            precision = merged.get("matmul_precision", "highest")
            cmd.extend(["--matmul_precision", precision])

            try:
                t_start = _time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                t_end = _time.perf_counter()
                wall_time = t_end - t_start

                if proc.returncode != 0:
                    last_error = (
                        f"Exit code {proc.returncode}: "
                        f"{proc.stderr[-2000:] if proc.stderr else ''}"
                    )
                    break

                run_times.append(wall_time)

                # Parse confidence
                target_name = tc_yaml.stem
                results_dir = (
                    work_dir / f"boltz_results_{target_name}" / "predictions" / target_name
                )
                if not results_dir.exists():
                    pred_base = work_dir / f"boltz_results_{target_name}" / "predictions"
                    if pred_base.exists():
                        subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
                        if subdirs:
                            results_dir = subdirs[0]

                quality = {}
                if results_dir.exists():
                    conf_files = sorted(results_dir.glob("confidence_*.json"))
                    if conf_files:
                        with conf_files[0].open() as f:
                            conf = _json.load(f)
                        for key in [
                            "confidence_score", "ptm", "iptm", "ligand_iptm",
                            "protein_iptm", "complex_plddt", "complex_iplddt",
                            "complex_pde", "complex_ipde",
                        ]:
                            if key in conf:
                                quality[key] = conf[key]

                run_qualities.append(quality)

            except subprocess.TimeoutExpired:
                last_error = "Timed out after 1800s"
                break
            except Exception as exc:
                last_error = f"Error: {exc}"
                break

        if last_error:
            entry = {
                "name": tc_name,
                "wall_time_s": None,
                "quality": {},
                "error": last_error,
            }
        else:
            median_time = statistics.median(run_times) if run_times else None
            all_plddts = [
                q.get("complex_plddt")
                for q in run_qualities
                if q.get("complex_plddt") is not None
            ]
            mean_plddt = sum(all_plddts) / len(all_plddts) if all_plddts else None
            merged_quality = dict(run_qualities[-1]) if run_qualities else {}
            if mean_plddt is not None:
                merged_quality["complex_plddt"] = mean_plddt
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
        failed = [r["name"] for r in results["per_complex"] if r.get("error")]
        results["aggregate"] = {
            "error": f"Failed: {failed}",
            "speedup": 0,
            "passes_quality_gate": False,
        }
    elif successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [
            r["quality"]["complex_plddt"]
            for r in successful
            if "complex_plddt" in r["quality"]
        ]
        iptms = [
            r["quality"]["iptm"]
            for r in successful
            if "iptm" in r["quality"]
        ]

        agg = {
            "num_successful": len(successful),
            "total_wall_time_s": total_time,
            "mean_wall_time_s": mean_time,
            "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
            "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
        }

        # Compare to baseline
        baseline = eval_config.get("baseline", {})
        if baseline:
            bl_time = baseline.get("mean_wall_time_s")
            bl_plddt = baseline.get("mean_plddt")
            if bl_time and mean_time > 0:
                agg["speedup"] = bl_time / mean_time
            if bl_plddt and plddts:
                mean_p = sum(plddts) / len(plddts)
                agg["plddt_delta_pp"] = (mean_p - bl_plddt) * 100.0
                regression = (bl_plddt - mean_p) * 100.0
                agg["passes_quality_gate"] = regression <= 2.0

                # Per-complex quality floor
                if baseline.get("per_complex"):
                    bl_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                    violations = {}
                    for r in successful:
                        bl_case = bl_by_name.get(r["name"])
                        if bl_case and bl_case.get("complex_plddt") is not None:
                            case_plddt = r["quality"].get("complex_plddt")
                            if case_plddt is not None:
                                case_reg = (bl_case["complex_plddt"] - case_plddt) * 100.0
                                if case_reg > 5.0:
                                    agg["passes_quality_gate"] = False
                                    violations[r["name"]] = f"-{case_reg:.1f}pp"
                    if violations:
                        agg["per_complex_regression"] = violations

        results["aggregate"] = agg

    return _json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    """Run parallel sweep of sampling steps and recycling steps."""

    # Phase 1: Sweep sampling steps with recycling_steps=3
    step_configs = [
        {"sampling_steps": 100, "recycling_steps": 3},
        {"sampling_steps": 50, "recycling_steps": 3},
        {"sampling_steps": 20, "recycling_steps": 3},
        {"sampling_steps": 10, "recycling_steps": 3},
    ]

    print("=" * 60)
    print("PHASE 1: Sampling steps sweep (recycling_steps=3)")
    print("=" * 60)

    config_jsons = [json.dumps(c) for c in step_configs]
    num_runs_list = [1] * len(step_configs)

    results_phase1 = []
    for result_json in evaluate_config.map(config_jsons, num_runs_list):
        result = json.loads(result_json)
        cfg = result["config"]
        agg = result.get("aggregate", {})
        print(f"\n--- steps={cfg['sampling_steps']}, recycle={cfg['recycling_steps']} ---")
        print(f"  Speedup: {agg.get('speedup', 'N/A')}")
        print(f"  Mean pLDDT: {agg.get('mean_plddt', 'N/A')}")
        print(f"  pLDDT delta: {agg.get('plddt_delta_pp', 'N/A')} pp")
        print(f"  Quality gate: {agg.get('passes_quality_gate', 'N/A')}")
        print(f"  Mean wall time: {agg.get('mean_wall_time_s', 'N/A')}")
        results_phase1.append(result)

    # Determine best step count that passes quality gate
    passing = [
        r for r in results_phase1
        if r.get("aggregate", {}).get("passes_quality_gate")
    ]
    if passing:
        best = max(passing, key=lambda r: r["aggregate"].get("speedup", 0))
        best_steps = best["config"]["sampling_steps"]
        print(
            f"\n[Phase 1] Best passing config: steps={best_steps}, "
            f"speedup={best['aggregate']['speedup']:.2f}x"
        )
    else:
        best_steps = 50  # conservative fallback
        print(
            f"\n[Phase 1] No config passed quality gate. "
            f"Using steps={best_steps} for Phase 2."
        )

    # Phase 2: Sweep recycling steps with the best step count
    recycle_configs = [
        {"sampling_steps": best_steps, "recycling_steps": 1},
        {"sampling_steps": best_steps, "recycling_steps": 0},
    ]

    print("\n" + "=" * 60)
    print(f"PHASE 2: Recycling sweep (sampling_steps={best_steps})")
    print("=" * 60)

    config_jsons2 = [json.dumps(c) for c in recycle_configs]
    num_runs_list2 = [1] * len(recycle_configs)

    results_phase2 = []
    for result_json in evaluate_config.map(config_jsons2, num_runs_list2):
        result = json.loads(result_json)
        cfg = result["config"]
        agg = result.get("aggregate", {})
        print(f"\n--- steps={cfg['sampling_steps']}, recycle={cfg['recycling_steps']} ---")
        print(f"  Speedup: {agg.get('speedup', 'N/A')}")
        print(f"  Mean pLDDT: {agg.get('mean_plddt', 'N/A')}")
        print(f"  pLDDT delta: {agg.get('plddt_delta_pp', 'N/A')} pp")
        print(f"  Quality gate: {agg.get('passes_quality_gate', 'N/A')}")
        print(f"  Mean wall time: {agg.get('mean_wall_time_s', 'N/A')}")
        results_phase2.append(result)

    # Save all results
    all_results = {
        "phase1_step_sweep": results_phase1,
        "phase2_recycle_sweep": results_phase2,
        "best_steps": best_steps,
    }

    outfile = Path("orbits/step-reduction/sweep_results.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[sweep] All results saved to {outfile}")
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(
        f"{'Steps':>6} {'Recycle':>7} {'Time(s)':>8} {'pLDDT':>7} "
        f"{'Delta(pp)':>9} {'Speedup':>8} {'Gate':>5}"
    )
    print("-" * 60)

    # Baseline row
    print(
        f"{'200':>6} {'3':>7} {'70.37':>8} {'0.7107':>7} "
        f"{'0.00':>9} {'1.00x':>8} {'PASS':>5}"
    )

    for r in results_phase1 + results_phase2:
        cfg = r["config"]
        agg = r.get("aggregate", {})
        steps = cfg["sampling_steps"]
        recycle = cfg["recycling_steps"]
        time_s = (
            f"{agg.get('mean_wall_time_s', 0):.1f}"
            if agg.get("mean_wall_time_s") else "ERR"
        )
        plddt = (
            f"{agg.get('mean_plddt', 0):.4f}"
            if agg.get("mean_plddt") else "ERR"
        )
        delta = (
            f"{agg.get('plddt_delta_pp', 0):+.2f}"
            if agg.get("plddt_delta_pp") is not None else "ERR"
        )
        speedup = (
            f"{agg.get('speedup', 0):.2f}x"
            if agg.get("speedup") else "ERR"
        )
        gate = "PASS" if agg.get("passes_quality_gate") else "FAIL"
        print(
            f"{steps:>6} {recycle:>7} {time_s:>8} {plddt:>7} "
            f"{delta:>9} {speedup:>8} {gate:>5}"
        )
