"""Validation runs (3 runs each) for promising ODE sampler configs.

Runs selected configs with num_runs=3 for stable timing comparison.
Each config gets its own L40S GPU and runs all 3 test cases x 3 runs.

Usage:
    cd /home/liambai/code/boltz/.worktrees/ode-sampler
    python orbits/ode-sampler/run_validate.py
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

app = modal.App("boltz-ode-validate", image=boltz_image)


def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


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


def _run_boltz_prediction(input_yaml, out_dir, config):
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

    result = {"wall_time_s": None, "quality": {}, "error": None}
    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start
        if proc.returncode != 0:
            result["error"] = f"Exit code {proc.returncode}.\nSTDERR: {proc.stderr[-2000:]}"
            return result
        result["quality"] = _parse_confidence(out_dir, input_yaml)
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout 1800s"
    except Exception as exc:
        result["error"] = str(exc)
    return result


@app.function(gpu="L40S", timeout=7200)
def evaluate_config(config_json: str) -> str:
    """Evaluate a single config with multiple runs."""
    import statistics
    config = json.loads(config_json)
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    num_runs = config.pop("num_runs", 3)

    results = {"config": config, "num_runs": num_runs, "per_complex": [], "aggregate": {}}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]
        if not tc_yaml.exists():
            results["per_complex"].append({"name": tc_name, "error": f"YAML not found", "wall_time_s": None, "quality": {}})
            continue

        run_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)
            label = "ODE" if config.get("gamma_0", 0.8) == 0.0 else "SDE"
            print(f"[validate] {label} steps={config['sampling_steps']} {tc_name} run {run_idx+1}/{num_runs}")
            pred_result = _run_boltz_prediction(tc_yaml, work_dir, config)
            if pred_result["error"]:
                last_error = pred_result["error"]
                break
            if pred_result["wall_time_s"]:
                run_times.append(pred_result["wall_time_s"])
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
            entry = {"name": tc_name, "wall_time_s": median_time, "quality": merged_quality, "error": None, "run_times": run_times}
        results["per_complex"].append(entry)

    successful = [r for r in results["per_complex"] if r["error"] is None and r["wall_time_s"] is not None]
    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [r["quality"]["complex_plddt"] for r in successful if "complex_plddt" in r["quality"] and isinstance(r["quality"]["complex_plddt"], (int, float))]
        iptms = [r["quality"]["iptm"] for r in successful if "iptm" in r["quality"]]
        results["aggregate"] = {
            "num_successful": len(successful),
            "mean_wall_time_s": mean_time,
            "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
            "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
        }
        baseline = eval_config.get("baseline")
        if baseline:
            bt = baseline.get("mean_wall_time_s")
            bp = baseline.get("mean_plddt")
            if bt and mean_time > 0:
                results["aggregate"]["speedup"] = bt / mean_time
            if bp and plddts:
                mp = sum(plddts) / len(plddts)
                results["aggregate"]["plddt_delta_pp"] = (mp - bp) * 100.0
                regression = (bp - mp) * 100.0
                results["aggregate"]["passes_quality_gate"] = regression <= 2.0
                if baseline.get("per_complex"):
                    baseline_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                    violations = {}
                    for r in successful:
                        bl = baseline_by_name.get(r["name"])
                        if bl and bl.get("complex_plddt") is not None:
                            cp = r["quality"].get("complex_plddt")
                            if cp is None:
                                results["aggregate"]["passes_quality_gate"] = False
                                violations[r["name"]] = "missing"
                            else:
                                reg = (bl["complex_plddt"] - cp) * 100.0
                                if reg > 5.0:
                                    results["aggregate"]["passes_quality_gate"] = False
                                    violations[r["name"]] = f"-{reg:.1f}pp"
                    if violations:
                        results["aggregate"]["per_complex_regression"] = violations
    else:
        results["aggregate"] = {"error": "No successful test cases"}

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    """Validate promising ODE configs with 3 runs each."""
    configs = [
        # Reference: stochastic 20-step with recycling=0 (the step-reduction best)
        {"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.8, "noise_scale": 1.003, "seed": 42, "num_runs": 3, "label": "SDE-20-r0"},
        # ODE 20-step with recycling=0
        {"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "noise_scale": 1.003, "seed": 42, "num_runs": 3, "label": "ODE-20-r0"},
        # ODE 10-step with recycling=0 (the key test)
        {"sampling_steps": 10, "recycling_steps": 0, "gamma_0": 0.0, "noise_scale": 1.003, "seed": 42, "num_runs": 3, "label": "ODE-10-r0"},
    ]

    labels = [c.pop("label") for c in configs]
    config_jsons = [json.dumps(c) for c in configs]

    print(f"[validate] Launching {len(configs)} validation runs in parallel (3 runs each)...")
    results = list(evaluate_config.map(config_jsons))

    print("\n" + "=" * 110)
    print(f"{'Config':<14} {'Steps':>5} {'gamma_0':>8} {'Time(s)':>8} {'pLDDT':>7} {'Delta(pp)':>10} {'Speedup':>8} {'Gate':>5}")
    print("-" * 110)

    all_results = []
    for label, result_json in zip(labels, results):
        r = json.loads(result_json)
        agg = r.get("aggregate", {})
        mt = agg.get("mean_wall_time_s")
        mp = agg.get("mean_plddt")
        dp = agg.get("plddt_delta_pp")
        sp = agg.get("speedup")
        passes = agg.get("passes_quality_gate")
        cfg = r.get("config", {})
        err = agg.get("error")

        if err:
            steps_str = str(cfg.get('sampling_steps', '?'))
            gamma_str = str(cfg.get('gamma_0', '?'))
            print(f"{label:<14} {steps_str:>5} {gamma_str:>8} {'ERROR':>8} -- {str(err)[:60]}")
        else:
            gate = "PASS" if passes else "FAIL" if passes is not None else "?"
            time_str = f"{mt:.1f}" if mt is not None else "?"
            plddt_str = f"{mp:.4f}" if mp is not None else "?"
            delta_str = f"{dp:+.2f}" if dp is not None else "?"
            speedup_str = f"{sp:.2f}x" if sp is not None else "?"
            steps_str = str(cfg.get('sampling_steps', '?'))
            gamma_str = str(cfg.get('gamma_0', '?'))
            print(f"{label:<14} {steps_str:>5} {gamma_str:>8} {time_str:>8} {plddt_str:>7} {delta_str:>10} {speedup_str:>8} {gate:>5}")

        # Print per-complex details
        for pc in r.get("per_complex", []):
            plddt = pc.get("quality", {}).get("complex_plddt", "?")
            wt = pc.get("wall_time_s", "?")
            times = pc.get("run_times", [])
            times_str = ", ".join(f"{t:.1f}" for t in times) if times else "?"
            wt_str = f"{wt:.1f}" if isinstance(wt, (int, float)) else str(wt)
            plddt_str = f"{plddt:.4f}" if isinstance(plddt, (int, float)) else str(plddt)
            print(f"  {pc['name']:<20} time={wt_str}s  pLDDT={plddt_str}  runs=[{times_str}]")

        all_results.append({"label": label, "result": r})

    output_path = Path(__file__).parent / "validate_results.json"
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[validate] Results saved to {output_path}")
