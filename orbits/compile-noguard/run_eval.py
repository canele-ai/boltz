"""Evaluation harness for compile-noguard orbit.

Runs the solution.py wrapper (ODE + TF32 + bf16 + torch.compile without guards)
on the standard test set using Modal L40S GPU.

Usage:
    # Single seed (fast iteration):
    modal run orbits/compile-noguard/run_eval.py --seed 42

    # Multiple seeds in parallel:
    modal run orbits/compile-noguard/run_eval.py --seeds '42,123,7'

    # Sanity check:
    modal run orbits/compile-noguard/run_eval.py --sanity-check

    # Control which modules to compile:
    modal run orbits/compile-noguard/run_eval.py --compile-targets pairformer,structure
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
# Paths
# ---------------------------------------------------------------------------
ORBIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORBIT_DIR.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
    )
    .pip_install("boltz==2.2.1")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(str(ORBIT_DIR / "solution.py"), remote_path="/eval/solution.py")
)

app = modal.App("compile-noguard-eval", image=boltz_image)

# ---------------------------------------------------------------------------
# Baseline values from config.yaml
# ---------------------------------------------------------------------------
BASELINE = {
    "mean_wall_time_s": 53.56664509866667,
    "mean_plddt": 0.7169803380966187,
    "per_complex": {
        "small_complex": {"wall_time_s": 42.805923546, "complex_plddt": 0.8344780802726746},
        "medium_complex": {"wall_time_s": 51.27919788899999, "complex_plddt": 0.5094943046569824},
        "large_complex": {"wall_time_s": 66.61481386100002, "complex_plddt": 0.806968629360199},
    },
}


def _run_prediction(input_yaml: Path, out_dir: Path, config: dict) -> dict:
    """Run a single Boltz-2 prediction using solution.py wrapper."""
    wrapper = "/eval/solution.py"
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
    ]

    # MSA
    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Stacked optimizations (parent orbit)
    cmd.extend(["--matmul_precision", config.get("matmul_precision", "high")])
    cmd.extend(["--gamma_0", str(config.get("gamma_0", 0.0))])
    cmd.extend(["--noise_scale", str(config.get("noise_scale", 1.003))])
    if config.get("bf16_trunk", True):
        cmd.append("--bf16_trunk")

    # Compile flags (THIS orbit's contribution)
    compile_targets = config.get("compile_targets", [])
    if "pairformer" in compile_targets:
        cmd.append("--compile_pairformer")
    if "msa" in compile_targets:
        cmd.append("--compile_msa")
    if "structure" in compile_targets:
        cmd.append("--compile_structure")
    if "confidence" in compile_targets:
        cmd.append("--compile_confidence")
    if compile_targets:
        cmd.extend(["--compile_mode", config.get("compile_mode", "default")])

    result = {"wall_time_s": None, "quality": {}, "error": None}

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"Exit code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-3000:] if proc.stderr else '(empty)'}\n"
                f"STDOUT: {proc.stdout[-1000:] if proc.stdout else '(empty)'}"
            )
            return result

        # Parse confidence
        result["quality"] = _parse_confidence(out_dir, input_yaml)
        result["stdout_tail"] = proc.stdout[-500:] if proc.stdout else ""

    except subprocess.TimeoutExpired:
        result["error"] = "Timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected: {exc}"

    return result


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict:
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Dir not found: {results_dir}"}

    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence files"}

    with confidence_files[0].open() as f:
        conf = json.load(f)

    quality = {}
    for key in ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]:
        if key in conf:
            quality[key] = conf[key]
    return quality


@app.function(gpu="L40S", timeout=7200)
def evaluate_seed(config_json: str) -> str:
    """Run evaluation on the full test set for a single seed."""
    import yaml

    config = json.loads(config_json)

    # Load test cases
    with open("/eval/config.yaml") as f:
        eval_config = yaml.safe_load(f)
    test_cases = eval_config.get("test_cases", [])

    results = {"config": config, "per_complex": [], "aggregate": {}}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name, "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None, "quality": {},
            })
            continue

        work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[eval] {tc_name} seed={config.get('seed')} "
              f"steps={config.get('sampling_steps')} compile={config.get('compile_targets')}")

        pred = _run_prediction(tc_yaml, work_dir, config)

        entry = {
            "name": tc_name,
            "wall_time_s": pred["wall_time_s"],
            "quality": pred["quality"],
            "error": pred["error"],
        }
        if "stdout_tail" in pred:
            entry["stdout_tail"] = pred["stdout_tail"]
        results["per_complex"].append(entry)

    # Aggregates
    successful = [r for r in results["per_complex"]
                  if r["error"] is None and r["wall_time_s"] is not None]

    if len(successful) < len(test_cases):
        failed = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        results["aggregate"] = {
            "error": f"Failed: {failed}",
            "speedup": 0, "passes_quality_gate": False,
        }
        return json.dumps(results, indent=2)

    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [r["quality"]["complex_plddt"] for r in successful
                  if "complex_plddt" in r["quality"]
                  and isinstance(r["quality"]["complex_plddt"], (int, float))
                  and not math.isnan(r["quality"]["complex_plddt"])]
        iptms = [r["quality"]["iptm"] for r in successful if "iptm" in r["quality"]]

        mean_plddt = sum(plddts) / len(plddts) if plddts else None

        agg = {
            "total_wall_time_s": total_time,
            "mean_wall_time_s": mean_time,
            "mean_plddt": mean_plddt,
            "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
        }

        if BASELINE["mean_wall_time_s"] and mean_time > 0:
            agg["speedup"] = BASELINE["mean_wall_time_s"] / mean_time

        if mean_plddt is not None:
            regression = (BASELINE["mean_plddt"] - mean_plddt) * 100.0
            agg["plddt_delta_pp"] = (mean_plddt - BASELINE["mean_plddt"]) * 100.0
            agg["passes_quality_gate"] = regression <= 2.0

            # Per-complex floor
            for r in successful:
                bl = BASELINE["per_complex"].get(r["name"])
                if bl and "complex_plddt" in r["quality"]:
                    case_regression = (bl["complex_plddt"] - r["quality"]["complex_plddt"]) * 100.0
                    if case_regression > 5.0:
                        agg["passes_quality_gate"] = False
                        agg.setdefault("per_complex_violations", {})[r["name"]] = f"-{case_regression:.1f}pp"

        results["aggregate"] = agg

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    sanity_check: bool = False,
    seed: int = 42,
    seeds: str = "",
    compile_targets: str = "pairformer,structure",
    compile_mode: str = "default",
    sampling_steps: int = 20,
    recycling_steps: int = 0,
    gamma_0: float = 0.0,
    noise_scale: float = 1.003,
    bf16_trunk: bool = True,
    matmul_precision: str = "high",
):
    """Run compile-noguard evaluation.

    Examples:
        modal run orbits/compile-noguard/run_eval.py --seed 42
        modal run orbits/compile-noguard/run_eval.py --seeds '42,123,7'
        modal run orbits/compile-noguard/run_eval.py --compile-targets 'pairformer,structure,msa'
    """
    targets = [t.strip() for t in compile_targets.split(",") if t.strip()]

    base_config = {
        "sampling_steps": sampling_steps,
        "recycling_steps": recycling_steps,
        "diffusion_samples": 1,
        "matmul_precision": matmul_precision,
        "gamma_0": gamma_0,
        "noise_scale": noise_scale,
        "bf16_trunk": bf16_trunk,
        "compile_targets": targets,
        "compile_mode": compile_mode,
    }

    if sanity_check:
        base_config["sampling_steps"] = 5
        base_config["seed"] = seed
        print(f"[compile-noguard] Sanity check: {json.dumps(base_config)}")
        result = evaluate_seed.remote(json.dumps(base_config))
        print(result)
        return

    # Determine seeds
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    else:
        seed_list = [seed]

    print(f"[compile-noguard] Config: {json.dumps(base_config, indent=2)}")
    print(f"[compile-noguard] Seeds: {seed_list}")
    print(f"[compile-noguard] Compile targets: {targets}")

    # Launch all seeds in parallel
    configs = []
    for s in seed_list:
        cfg = dict(base_config)
        cfg["seed"] = s
        configs.append(json.dumps(cfg))

    # Parallel execution via Modal .map()
    all_results = []
    for result_json in evaluate_seed.map(configs):
        result = json.loads(result_json)
        all_results.append(result)
        seed_val = result["config"]["seed"]
        agg = result.get("aggregate", {})
        speedup = agg.get("speedup", 0)
        plddt = agg.get("mean_plddt")
        gate = agg.get("passes_quality_gate")
        if plddt is not None:
            plddt_str = f"{plddt:.4f}"
        else:
            plddt_str = "N/A"
        gate_str = "PASS" if gate else "FAIL"
        print(f"\n[seed={seed_val}] speedup={speedup:.3f}x, "
              f"pLDDT={plddt_str}, gate={gate_str}")

        # Print per-complex details
        for pc in result.get("per_complex", []):
            if pc.get("error"):
                print(f"  {pc['name']}: ERROR: {pc['error'][:200]}")
            else:
                pc_plddt = pc['quality'].get('complex_plddt')
                if pc_plddt is not None:
                    pc_plddt_str = f"{pc_plddt:.4f}"
                else:
                    pc_plddt_str = "N/A"
                print(f"  {pc['name']}: {pc['wall_time_s']:.1f}s, pLDDT={pc_plddt_str}")

    # Aggregate across seeds
    if len(all_results) > 1:
        speedups = [r["aggregate"].get("speedup", 0) for r in all_results if "aggregate" in r]
        plddts = [r["aggregate"].get("mean_plddt") for r in all_results
                  if r.get("aggregate", {}).get("mean_plddt") is not None]
        gates = [r["aggregate"].get("passes_quality_gate", False) for r in all_results]

        if speedups:
            import statistics
            mean_speedup = statistics.mean(speedups)
            std_speedup = statistics.stdev(speedups) if len(speedups) > 1 else 0
            mean_plddt_all = statistics.mean(plddts) if plddts else None

            print(f"\n{'='*60}")
            print(f"AGGREGATE ({len(seed_list)} seeds)")
            print(f"  Speedup: {mean_speedup:.3f}x +/- {std_speedup:.3f}")
            if mean_plddt_all:
                print(f"  pLDDT:   {mean_plddt_all:.4f}")
            print(f"  Gate:    {'ALL PASS' if all(gates) else 'SOME FAIL'}")
            print(f"{'='*60}")


if __name__ == "__main__":
    print("Use: modal run orbits/compile-noguard/run_eval.py [--flags]")
