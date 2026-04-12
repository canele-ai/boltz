"""Clean evaluation of diffusion loop optimizations.

Runs each config multiple times on the same warm GPU to get reliable timing.
Uses the same container for all configs to avoid container coldstart variance.

Strategy:
1. Run a warmup (ODE-20-r0) to populate MSA cache and warm GPU
2. Then run each config 3x to get stable timing
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

app = modal.App("boltz-diffusion-clean-eval", image=boltz_image)


def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _run_single_prediction(
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

    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    cmd.extend(["--matmul_precision", config.get("matmul_precision", "highest")])
    cmd.extend(["--gamma_0", str(config.get("gamma_0", 0.8))])
    cmd.extend(["--noise_scale", str(config.get("noise_scale", 1.003))])

    if config.get("compile_structure"):
        cmd.append("--compile_structure")
    if config.get("compile_pairformer"):
        cmd.append("--compile_pairformer")

    result = {"wall_time_s": None, "quality": {}, "error": None}

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = f"Exit code {proc.returncode}: {proc.stderr[-1500:] if proc.stderr else ''}"
            return result

        # Parse confidence
        target_name = input_yaml.stem
        results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name
        if not results_dir.exists():
            pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
            if pred_base.exists():
                subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
                if subdirs:
                    results_dir = subdirs[0]

        if results_dir.exists():
            confidence_files = sorted(results_dir.glob("confidence_*.json"))
            if confidence_files:
                with confidence_files[0].open() as f:
                    conf = json.load(f)
                for key in ["confidence_score", "ptm", "iptm", "ligand_iptm",
                            "protein_iptm", "complex_plddt", "complex_iplddt",
                            "complex_pde", "complex_ipde"]:
                    if key in conf:
                        result["quality"][key] = conf[key]
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout after 1800s"
    except Exception as exc:
        result["error"] = str(exc)

    return result


@app.function(gpu="L40S", timeout=14400)
def run_clean_comparison(configs_json: str) -> str:
    """Run configs on warm GPU with multiple seeds for reliable comparison.

    configs_json: JSON list of {label, config, seeds}
    """
    import statistics

    configs = json.loads(configs_json)
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    baseline = eval_config.get("baseline", {})

    # Step 1: Warmup run to cache MSAs
    print("[eval] WARMUP: Running first prediction to cache MSAs and warm GPU...")
    warmup_tc = test_cases[0]
    warmup_yaml = Path("/eval") / warmup_tc["yaml"]
    warmup_dir = Path(f"/tmp/boltz_eval/warmup_{uuid.uuid4().hex[:8]}")
    warmup_dir.mkdir(parents=True, exist_ok=True)
    warmup_config = {
        "sampling_steps": 20, "recycling_steps": 0,
        "gamma_0": 0.0, "seed": 42,
    }
    _run_single_prediction(warmup_yaml, warmup_dir, warmup_config)
    print("[eval] Warmup complete.\n")

    all_results = []

    for entry in configs:
        label = entry["label"]
        config = entry["config"]
        seeds = entry.get("seeds", [42, 123, 7])

        print(f"\n{'='*60}")
        print(f"[eval] Config: {label}")
        print(f"[eval] Seeds: {seeds}")
        print(f"{'='*60}")

        seed_results = []

        for seed in seeds:
            cfg = dict(config)
            cfg["seed"] = seed

            per_complex = []
            for tc in test_cases:
                tc_name = tc["name"]
                tc_yaml = Path("/eval") / tc["yaml"]

                if not tc_yaml.exists():
                    per_complex.append({"name": tc_name, "error": "YAML not found"})
                    continue

                work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
                work_dir.mkdir(parents=True, exist_ok=True)

                print(f"  [{label}] seed={seed}, {tc_name}...", end=" ", flush=True)
                result = _run_single_prediction(tc_yaml, work_dir, cfg)

                if result["error"]:
                    print(f"ERROR: {result['error'][:80]}")
                    per_complex.append({
                        "name": tc_name,
                        "wall_time_s": result["wall_time_s"],
                        "quality": {},
                        "error": result["error"],
                    })
                else:
                    plddt = result["quality"].get("complex_plddt", "?")
                    print(f"{result['wall_time_s']:.1f}s, pLDDT={plddt}")
                    per_complex.append({
                        "name": tc_name,
                        "wall_time_s": result["wall_time_s"],
                        "quality": result["quality"],
                        "error": None,
                    })

            # Compute seed-level aggregate
            successful = [r for r in per_complex if r["error"] is None and r["wall_time_s"] is not None]
            if successful and len(successful) == len(test_cases):
                mean_time = sum(r["wall_time_s"] for r in successful) / len(successful)
                plddts = [r["quality"]["complex_plddt"] for r in successful
                          if "complex_plddt" in r["quality"]]
                mean_plddt = sum(plddts) / len(plddts) if plddts else None
                speedup = baseline.get("mean_wall_time_s", 0) / mean_time if mean_time > 0 else None
            else:
                mean_time = None
                mean_plddt = None
                speedup = None

            seed_results.append({
                "seed": seed,
                "per_complex": per_complex,
                "mean_wall_time_s": mean_time,
                "mean_plddt": mean_plddt,
                "speedup": speedup,
            })

        # Compute cross-seed statistics
        times = [s["mean_wall_time_s"] for s in seed_results if s["mean_wall_time_s"] is not None]
        plddts_all = [s["mean_plddt"] for s in seed_results if s["mean_plddt"] is not None]
        speedups = [s["speedup"] for s in seed_results if s["speedup"] is not None]

        summary = {
            "mean_time": statistics.mean(times) if times else None,
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "mean_plddt": statistics.mean(plddts_all) if plddts_all else None,
            "std_plddt": statistics.stdev(plddts_all) if len(plddts_all) > 1 else 0,
            "mean_speedup": statistics.mean(speedups) if speedups else None,
            "std_speedup": statistics.stdev(speedups) if len(speedups) > 1 else 0,
            "num_seeds": len(seeds),
            "num_successful": len(times),
        }

        # Quality gate
        if summary["mean_plddt"] is not None and baseline.get("mean_plddt"):
            regression = (baseline["mean_plddt"] - summary["mean_plddt"]) * 100
            summary["plddt_delta_pp"] = -regression  # positive means better
            summary["passes_quality_gate"] = regression <= 2.0
        else:
            summary["passes_quality_gate"] = False

        config_result = {
            "label": label,
            "config": config,
            "seeds": seeds,
            "per_seed": seed_results,
            "summary": summary,
        }
        all_results.append(config_result)

        print(f"\n  SUMMARY: {label}")
        if summary["mean_speedup"] is not None:
            print(f"    Speedup: {summary['mean_speedup']:.2f}x +/- {summary['std_speedup']:.3f}")
        if summary["mean_plddt"] is not None:
            print(f"    pLDDT:   {summary['mean_plddt']:.4f} +/- {summary['std_plddt']:.4f}")
        if summary["mean_time"] is not None:
            print(f"    Time:    {summary['mean_time']:.1f}s +/- {summary['std_time']:.1f}s")
        print(f"    Gate:    {'PASS' if summary['passes_quality_gate'] else 'FAIL'}")

    return json.dumps(all_results, indent=2)


@app.local_entrypoint()
def main(mode: str = "full"):
    """Run clean comparison experiments.

    Modes:
        full: ODE-20 and ODE-10 with 3 seeds each
        compile: Test torch.compile with fixed wrapper
        tf32: Test TF32 matmul precision
    """
    if mode == "full":
        configs = [
            {
                "label": "ODE-20-r0",
                "config": {
                    "sampling_steps": 20,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                },
                "seeds": [42, 123, 7],
            },
            {
                "label": "ODE-10-r0",
                "config": {
                    "sampling_steps": 10,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                },
                "seeds": [42, 123, 7],
            },
            {
                "label": "ODE-20-r0-tf32",
                "config": {
                    "sampling_steps": 20,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "matmul_precision": "high",
                },
                "seeds": [42, 123, 7],
            },
            {
                "label": "ODE-10-r0-tf32",
                "config": {
                    "sampling_steps": 10,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "matmul_precision": "high",
                },
                "seeds": [42, 123, 7],
            },
        ]
    elif mode == "compile":
        configs = [
            {
                "label": "ODE-20-r0-compile",
                "config": {
                    "sampling_steps": 20,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "compile_structure": True,
                },
                "seeds": [42],
            },
        ]
    elif mode == "tf32":
        configs = [
            {
                "label": "ODE-20-r0-tf32",
                "config": {
                    "sampling_steps": 20,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "noise_scale": 1.003,
                    "matmul_precision": "high",
                },
                "seeds": [42, 123, 7],
            },
        ]
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    print(f"[clean-eval] Running {len(configs)} configs...")
    results_json = run_clean_comparison.remote(json.dumps(configs))
    results = json.loads(results_json)

    # Save results
    output_path = Path(f"orbits/diffusion-loop-opt/clean_results_{mode}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<25} {'Time(s)':<15} {'pLDDT':<15} {'Speedup':<15} {'Gate':<6}")
    print("-" * 90)
    for r in results:
        s = r["summary"]
        label = r["label"]
        t = f"{s['mean_time']:.1f} +/- {s['std_time']:.1f}" if s["mean_time"] else "N/A"
        p = f"{s['mean_plddt']:.4f} +/- {s['std_plddt']:.4f}" if s["mean_plddt"] else "N/A"
        sp = f"{s['mean_speedup']:.2f}x +/- {s['std_speedup']:.3f}" if s["mean_speedup"] else "N/A"
        g = "PASS" if s["passes_quality_gate"] else "FAIL"
        print(f"{label:<25} {t:<15} {p:<15} {sp:<15} {g:<6}")
    print("=" * 90)
