"""Evaluator for diffusion loop optimizations.

Tests multiple configurations:
1. ODE-20 (gamma_0=0, 20 steps, 0 recycles) - parent orbit's best config on eval-v2
2. ODE-10 (gamma_0=0, 10 steps, 0 recycles) - fewer steps with clean timing
3. ODE-20 + compile_structure - torch.compile on score model
4. ODE-20 + TF32 matmul precision

All run on eval-v2 (torch 2.6.0 + cuequivariance kernels).
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
REPO_ROOT = EVAL_DIR.parent.parent

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
    .add_local_file(str(ORBIT_DIR / "boltz_wrapper_opt.py"), remote_path="/eval/boltz_wrapper_opt.py")
)

app = modal.App("boltz-diffusion-loop-opt", image=boltz_image)


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
    """Run a single Boltz-2 prediction with custom wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_opt.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
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

    # Custom flags for our wrapper
    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    gamma_0 = config.get("gamma_0", 0.8)
    cmd.extend(["--gamma_0", str(gamma_0)])

    noise_scale = config.get("noise_scale", 1.003)
    cmd.extend(["--noise_scale", str(noise_scale)])

    if config.get("compile_structure"):
        cmd.append("--compile_structure")
    if config.get("compile_pairformer"):
        cmd.append("--compile_pairformer")

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

        # Parse quality metrics
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


@app.function(
    gpu="L40S",
    timeout=7200,
)
def evaluate_config(config_json: str, label: str = "") -> str:
    """Evaluate a single configuration on all test cases.

    Returns JSON with per-complex and aggregate results.
    """
    config = json.loads(config_json)
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results = {
        "label": label,
        "config": config,
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

        work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[eval] {label}: Running {tc_name} with steps={config.get('sampling_steps', 200)}, "
              f"recycle={config.get('recycling_steps', 3)}, gamma_0={config.get('gamma_0', 0.8)}")

        pred_result = _run_boltz_prediction(tc_yaml, work_dir, config)

        if pred_result["error"] is not None:
            results["per_complex"].append({
                "name": tc_name,
                "wall_time_s": pred_result["wall_time_s"],
                "quality": {},
                "error": pred_result["error"],
            })
        else:
            results["per_complex"].append({
                "name": tc_name,
                "wall_time_s": pred_result["wall_time_s"],
                "quality": pred_result["quality"],
                "error": None,
            })

    # Compute aggregates
    successful = [r for r in results["per_complex"]
                  if r["error"] is None and r["wall_time_s"] is not None]

    if successful and len(successful) == len(test_cases):
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [r["quality"]["complex_plddt"] for r in successful
                   if "complex_plddt" in r["quality"]]
        iptms = [r["quality"]["iptm"] for r in successful
                  if "iptm" in r["quality"]]

        results["aggregate"] = {
            "total_wall_time_s": total_time,
            "mean_wall_time_s": mean_time,
            "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
            "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
        }

        # Compare against baseline
        baseline = eval_config.get("baseline", {})
        baseline_time = baseline.get("mean_wall_time_s")
        baseline_plddt = baseline.get("mean_plddt")
        if baseline_time and mean_time > 0:
            results["aggregate"]["speedup"] = baseline_time / mean_time
        if baseline_plddt and plddts:
            mean_plddt = sum(plddts) / len(plddts)
            results["aggregate"]["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
            regression = (baseline_plddt - mean_plddt) * 100.0
            results["aggregate"]["passes_quality_gate"] = regression <= 2.0
    else:
        failed = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        results["aggregate"] = {
            "error": f"Failed test cases: {failed}",
            "passes_quality_gate": False,
        }

    return json.dumps(results, indent=2)


@app.function(
    gpu="L40S",
    timeout=14400,
)
def run_sweep(configs_json: str) -> str:
    """Run multiple configurations sequentially on the same GPU.

    This avoids GPU warmup overhead by keeping the same container.
    configs_json: JSON list of {config: {...}, label: "..."}
    """
    configs = json.loads(configs_json)
    all_results = []

    for entry in configs:
        config = entry["config"]
        label = entry.get("label", "unnamed")
        result_json = evaluate_config.local(json.dumps(config), label=label)
        result = json.loads(result_json)
        all_results.append(result)

        # Print summary
        agg = result.get("aggregate", {})
        speedup = agg.get("speedup", "N/A")
        plddt = agg.get("mean_plddt", "N/A")
        gate = agg.get("passes_quality_gate", "N/A")
        print(f"\n[sweep] {label}: speedup={speedup}, pLDDT={plddt}, gate={gate}\n")

    return json.dumps(all_results, indent=2)


@app.local_entrypoint()
def main(mode: str = "sweep"):
    """Run diffusion loop optimization experiments.

    Modes:
        sweep: Run all optimization configs (single run each for exploration)
        validate: Run best config with 3 runs for validation
    """
    if mode == "sweep":
        configs = [
            {
                "label": "ODE-20-r0 (eval-v2 baseline)",
                "config": {
                    "sampling_steps": 20,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "seed": 42,
                },
            },
            {
                "label": "ODE-10-r0",
                "config": {
                    "sampling_steps": 10,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "seed": 42,
                },
            },
            {
                "label": "ODE-20-r0-tf32",
                "config": {
                    "sampling_steps": 20,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "matmul_precision": "high",
                    "seed": 42,
                },
            },
            {
                "label": "ODE-20-r0-compile",
                "config": {
                    "sampling_steps": 20,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "compile_structure": True,
                    "seed": 42,
                },
            },
            {
                "label": "ODE-10-r0-tf32",
                "config": {
                    "sampling_steps": 10,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "matmul_precision": "high",
                    "seed": 42,
                },
            },
        ]

        print(f"[diffusion-loop-opt] Running sweep of {len(configs)} configurations...")
        results_json = run_sweep.remote(json.dumps(configs))
        results = json.loads(results_json)

        # Save results
        output_path = Path("orbits/diffusion-loop-opt/sweep_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[diffusion-loop-opt] Results saved to {output_path}")

        # Print summary table
        print("\n" + "=" * 80)
        print(f"{'Config':<30} {'Time(s)':<10} {'pLDDT':<10} {'Speedup':<10} {'Gate':<6}")
        print("-" * 80)
        for r in results:
            label = r.get("label", "?")
            agg = r.get("aggregate", {})
            t = agg.get("mean_wall_time_s", "N/A")
            p = agg.get("mean_plddt", "N/A")
            s = agg.get("speedup", "N/A")
            g = agg.get("passes_quality_gate", "N/A")
            t_str = f"{t:.1f}" if isinstance(t, float) else str(t)
            p_str = f"{p:.4f}" if isinstance(p, float) else str(p)
            s_str = f"{s:.2f}x" if isinstance(s, float) else str(s)
            g_str = "PASS" if g is True else ("FAIL" if g is False else str(g))
            print(f"{label:<30} {t_str:<10} {p_str:<10} {s_str:<10} {g_str:<6}")
        print("=" * 80)

    elif mode == "validate":
        # Load sweep results and validate the best config
        sweep_path = Path("orbits/diffusion-loop-opt/sweep_results.json")
        if not sweep_path.exists():
            print("[validate] No sweep results found. Run with mode=sweep first.")
            sys.exit(1)

        with sweep_path.open() as f:
            sweep_results = json.load(f)

        # Find best passing config
        best = None
        best_speedup = 0
        for r in sweep_results:
            agg = r.get("aggregate", {})
            if agg.get("passes_quality_gate") and agg.get("speedup", 0) > best_speedup:
                best = r
                best_speedup = agg["speedup"]

        if best is None:
            print("[validate] No passing configs found in sweep.")
            sys.exit(1)

        print(f"[validate] Best config: {best['label']} (speedup={best_speedup:.2f}x)")
        print(f"[validate] Running 3-seed validation...")

        base_config = best["config"]
        seeds = [42, 123, 7]

        # Run seeds in parallel using .map()
        configs_for_seeds = []
        for seed in seeds:
            cfg = dict(base_config)
            cfg["seed"] = seed
            configs_for_seeds.append((json.dumps(cfg), f"{best['label']}-seed{seed}"))

        results_list = []
        for config_json, label in configs_for_seeds:
            result_json = evaluate_config.remote(config_json, label=label)
            results_list.append(json.loads(result_json))

        # Compute mean +/- std
        speedups = [r["aggregate"].get("speedup", 0) for r in results_list if r["aggregate"].get("speedup")]
        plddts = [r["aggregate"].get("mean_plddt", 0) for r in results_list if r["aggregate"].get("mean_plddt")]
        times = [r["aggregate"].get("mean_wall_time_s", 0) for r in results_list if r["aggregate"].get("mean_wall_time_s")]

        import statistics

        validate_results = {
            "label": best["label"],
            "config": base_config,
            "seeds": seeds,
            "per_seed": results_list,
            "summary": {
                "mean_speedup": statistics.mean(speedups) if speedups else None,
                "std_speedup": statistics.stdev(speedups) if len(speedups) > 1 else 0,
                "mean_plddt": statistics.mean(plddts) if plddts else None,
                "std_plddt": statistics.stdev(plddts) if len(plddts) > 1 else 0,
                "mean_time": statistics.mean(times) if times else None,
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            },
        }

        output_path = Path("orbits/diffusion-loop-opt/validate_results.json")
        with output_path.open("w") as f:
            json.dump(validate_results, f, indent=2)

        summary = validate_results["summary"]
        print(f"\n[validate] {best['label']}")
        print(f"  Speedup: {summary['mean_speedup']:.2f}x +/- {summary['std_speedup']:.3f}")
        print(f"  pLDDT:   {summary['mean_plddt']:.4f} +/- {summary['std_plddt']:.4f}")
        print(f"  Time:    {summary['mean_time']:.1f}s +/- {summary['std_time']:.1f}s")

        # Check quality gate across all seeds
        all_pass = all(
            r["aggregate"].get("passes_quality_gate", False)
            for r in results_list
        )
        print(f"  Quality gate: {'ALL PASS' if all_pass else 'SOME FAIL'}")


if __name__ == "__main__":
    print("Use 'modal run orbits/diffusion-loop-opt/eval_diffusion_opt.py' to run on GPU.")
