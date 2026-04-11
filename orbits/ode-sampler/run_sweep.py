"""Parallel sweep of ODE vs stochastic sampler configurations.

Runs multiple configs in parallel using Modal's .map() to maximize throughput.
Each config gets its own L40S GPU instance.

Usage:
    cd /home/liambai/code/boltz/.worktrees/ode-sampler
    python orbits/ode-sampler/run_sweep.py
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

app = modal.App("boltz-ode-sweep", image=boltz_image)


# ---------------------------------------------------------------------------
# Helpers (run inside Modal container)
# ---------------------------------------------------------------------------


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


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Modal function: evaluate a single config on the full test set
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=7200)
def evaluate_config(config_json: str) -> str:
    """Run evaluation for a single configuration."""
    import statistics

    config = json.loads(config_json)
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    num_runs = config.pop("num_runs", 1)

    results: dict[str, Any] = {
        "config": config,
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

            label = "ODE" if config.get("gamma_0", 0.8) == 0.0 else "SDE"
            print(
                f"[sweep] {label} steps={config['sampling_steps']} "
                f"gamma_0={config.get('gamma_0', 0.8)} "
                f"{tc_name} run {run_idx + 1}/{num_runs}"
            )

            pred_result = _run_boltz_prediction(tc_yaml, work_dir, config)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
            run_qualities.append(pred_result["quality"])

        if last_error is not None:
            entry = {
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
            mean_plddt = (sum(all_plddts) / len(all_plddts)) if all_plddts else None
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

    # Aggregate
    successful = [
        r for r in results["per_complex"]
        if r["error"] is None and r["wall_time_s"] is not None
    ]

    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [
            r["quality"]["complex_plddt"]
            for r in successful
            if "complex_plddt" in r["quality"]
            and isinstance(r["quality"]["complex_plddt"], (int, float))
            and not math.isnan(r["quality"]["complex_plddt"])
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
        if baseline:
            baseline_time = baseline.get("mean_wall_time_s")
            baseline_plddt = baseline.get("mean_plddt")
            if baseline_time and mean_time > 0:
                results["aggregate"]["speedup"] = baseline_time / mean_time
            if baseline_plddt and plddts:
                mp = sum(plddts) / len(plddts)
                results["aggregate"]["plddt_delta_pp"] = (mp - baseline_plddt) * 100.0
                regression = (baseline_plddt - mp) * 100.0
                results["aggregate"]["passes_quality_gate"] = regression <= 2.0
    else:
        results["aggregate"] = {
            "error": "No successful test cases",
            "speedup": 0,
            "passes_quality_gate": False,
        }

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Local entrypoint: sweep configs in parallel
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run parallel sweep of ODE vs stochastic sampler configurations."""

    # Define configurations to test
    configs = [
        # Stochastic baselines (gamma_0=0.8, the default) at recycling_steps=0
        {"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.8, "noise_scale": 1.003, "seed": 42, "label": "SDE-20"},
        {"sampling_steps": 10, "recycling_steps": 0, "gamma_0": 0.8, "noise_scale": 1.003, "seed": 42, "label": "SDE-10"},

        # Deterministic ODE (gamma_0=0) at various step counts
        {"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "noise_scale": 1.003, "seed": 42, "label": "ODE-20"},
        {"sampling_steps": 10, "recycling_steps": 0, "gamma_0": 0.0, "noise_scale": 1.003, "seed": 42, "label": "ODE-10"},
        {"sampling_steps": 5,  "recycling_steps": 0, "gamma_0": 0.0, "noise_scale": 1.003, "seed": 42, "label": "ODE-5"},

        # ODE with noise_scale=1.0 (pure ODE, no noise scaling)
        {"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "noise_scale": 1.0, "seed": 42, "label": "ODE-20-ns1"},
        {"sampling_steps": 10, "recycling_steps": 0, "gamma_0": 0.0, "noise_scale": 1.0, "seed": 42, "label": "ODE-10-ns1"},
    ]

    # Prepare configs as JSON strings (remove label key before sending)
    labels = [c.pop("label") for c in configs]
    config_jsons = [json.dumps(c) for c in configs]

    print(f"[sweep] Launching {len(configs)} configurations in parallel on L40S GPUs...")
    for label, cfg in zip(labels, configs):
        gamma = cfg.get("gamma_0", 0.8)
        ns = cfg.get("noise_scale", 1.003)
        print(f"  {label}: steps={cfg['sampling_steps']}, gamma_0={gamma}, noise_scale={ns}")

    # Run all configs in parallel via Modal .map()
    results = list(evaluate_config.map(config_jsons))

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<16} {'Steps':>5} {'gamma_0':>8} {'ns':>6} {'Time(s)':>8} {'pLDDT':>7} {'Delta(pp)':>10} {'Speedup':>8} {'Gate':>5}")
    print("-" * 100)

    all_results = []
    for label, result_json in zip(labels, results):
        r = json.loads(result_json)
        agg = r.get("aggregate", {})
        mean_time = agg.get("mean_wall_time_s")
        mean_plddt = agg.get("mean_plddt")
        delta_pp = agg.get("plddt_delta_pp")
        speedup = agg.get("speedup")
        passes = agg.get("passes_quality_gate")
        cfg = r.get("config", {})
        error = agg.get("error")

        if error:
            print(f"{label:<16} {cfg.get('sampling_steps', '?'):>5} {cfg.get('gamma_0', '?'):>8} {cfg.get('noise_scale', '?'):>6} {'ERROR':>8} -- {error[:60]}")
        else:
            gate_str = "PASS" if passes else "FAIL" if passes is not None else "?"
            time_str = f"{mean_time:.1f}" if mean_time else "?"
            plddt_str = f"{mean_plddt:.4f}" if mean_plddt else "?"
            delta_str = f"{delta_pp:+.2f}" if delta_pp is not None else "?"
            speedup_str = f"{speedup:.2f}x" if speedup else "?"
            print(f"{label:<16} {cfg.get('sampling_steps', '?'):>5} {cfg.get('gamma_0', '?'):>8} {cfg.get('noise_scale', '?'):>6} {time_str:>8} {plddt_str:>7} {delta_str:>10} {speedup_str:>8} {gate_str:>5}")

        all_results.append({"label": label, "result": r})

    # Save raw results
    output_path = Path(__file__).parent / "sweep_results.json"
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[sweep] Raw results saved to {output_path}")


if __name__ == "__main__":
    main()
