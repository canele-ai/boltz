"""ODE step count sweep on eval-v5 with bypass wrapper + 0 recycling.

Sweeps sampling_steps in {6, 8, 10, 12, 15} with 3 seeds each.
All configs: recycling=0, gamma_0=0.0, TF32, bf16_trunk, bypass wrapper, CUDA warmup.

Uses Modal .map() for parallel execution across all (steps, seed) combos.

Usage:
    modal run orbits/ode-steps-v5/eval_steps_sweep.py
"""

from __future__ import annotations

import json
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent
BYPASS_WRAPPER = REPO_ROOT / "orbits" / "bypass-lightning" / "boltz_bypass_wrapper.py"

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
    .add_local_file(
        str(BYPASS_WRAPPER),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
)

app = modal.App("boltz-ode-steps-v5-sweep", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

STEP_COUNTS = [6, 8, 10, 12, 15]
SEEDS = [42, 123, 7]

BASE_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "cuda_warmup": True,
    "diffusion_samples": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path):
    """Inject cached MSA paths into input YAML."""
    import yaml
    import shutil

    target_name = input_yaml.stem
    cache_dir = msa_cache_root / target_name
    if not cache_dir.exists():
        return None
    msa_files = sorted(cache_dir.glob("*.csv"))
    if not msa_files:
        return None

    with input_yaml.open() as f:
        data = yaml.safe_load(f)

    local_msa_dir = work_dir / "cached_msas"
    local_msa_dir.mkdir(parents=True, exist_ok=True)

    entity_msa_map = {}
    for msa_file in msa_files:
        local_path = local_msa_dir / msa_file.name
        shutil.copy2(msa_file, local_path)
        parts = msa_file.stem.split("_")
        if len(parts) >= 2:
            entity_msa_map[parts[-1]] = str(local_path)

    if "sequences" not in data:
        return None

    entity_idx = 0
    injected = 0
    for seq_entry in data["sequences"]:
        if "protein" in seq_entry:
            entity_key = str(entity_idx)
            if entity_key in entity_msa_map:
                seq_entry["protein"]["msa"] = entity_msa_map[entity_key]
                injected += 1
            entity_idx += 1

    if injected == 0:
        return None

    cached_yaml = work_dir / f"{target_name}_cached.yaml"
    with cached_yaml.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[sweep] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


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


def _run_boltz_bypass(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction using the bypass-lightning wrapper."""
    wrapper = str(Path("/eval/boltz_bypass_wrapper.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--gamma_0", str(config.get("gamma_0", 0.8)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
    ]

    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    if config.get("cuda_warmup", False):
        cmd.append("--cuda_warmup")

    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    result: dict[str, Any] = {
        "wall_time_s": None,
        "predict_only_s": None,
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
                f"STDOUT: {proc.stdout[-2000:] if proc.stdout else '(empty)'}\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        predict_start = None
        predict_end = None
        for line in proc.stderr.split("\n"):
            if "[PHASE] predict_start=" in line:
                predict_start = float(line.split("=")[1])
            elif "[PHASE] predict_end=" in line:
                predict_end = float(line.split("=")[1])

        if predict_start is not None and predict_end is not None:
            result["predict_only_s"] = predict_end - predict_start

        quality = _parse_confidence(out_dir, input_yaml)
        result["quality"] = quality

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Modal function: evaluate a single (steps, seed) combo on all test cases
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_cache},
)
def evaluate_single(steps: int, seed: int) -> str:
    """Evaluate one (steps, seed) combination on all 3 test cases.

    Returns JSON with per-complex results.
    """
    config = dict(BASE_CONFIG)
    config["sampling_steps"] = steps
    config["seed"] = seed

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    result: dict[str, Any] = {
        "steps": steps,
        "seed": seed,
        "per_complex": [],
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            result["per_complex"].append({
                "name": tc_name,
                "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None,
                "predict_only_s": None,
                "quality": {},
            })
            continue

        work_dir = Path(f"/tmp/boltz_eval/{tc_name}_s{steps}_seed{seed}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        run_config = dict(config)
        effective_yaml = tc_yaml
        if use_msa_cache:
            cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
            if cached_yaml is not None:
                effective_yaml = cached_yaml
                run_config["_msa_cached"] = True

        print(f"[sweep] steps={steps}, seed={seed}, complex={tc_name}")

        pred_result = _run_boltz_bypass(effective_yaml, work_dir, run_config)

        entry: dict[str, Any] = {
            "name": tc_name,
            "wall_time_s": pred_result["wall_time_s"],
            "predict_only_s": pred_result.get("predict_only_s"),
            "quality": pred_result["quality"],
            "error": pred_result["error"],
        }
        result["per_complex"].append(entry)

        if pred_result["error"]:
            print(f"[sweep] ERROR steps={steps}, seed={seed}, {tc_name}: {pred_result['error'][:300]}")
        else:
            plddt = pred_result["quality"].get("complex_plddt", "N/A")
            wt = pred_result["wall_time_s"]
            po = pred_result.get("predict_only_s", "N/A")
            print(f"[sweep] steps={steps}, seed={seed}, {tc_name}: "
                  f"wall={wt:.1f}s, predict_only={po}s, pLDDT={plddt}")

    return json.dumps(result)


# ---------------------------------------------------------------------------
# CLI entrypoint: launch all combos via .map()
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Launch ODE step sweep with parallel seeds via Modal .map()."""
    import math

    # Build all (steps, seed) combos
    combos = [(steps, seed) for steps in STEP_COUNTS for seed in SEEDS]
    steps_list = [c[0] for c in combos]
    seeds_list = [c[1] for c in combos]

    print(f"[sweep] Launching {len(combos)} evaluations: "
          f"{len(STEP_COUNTS)} step counts x {len(SEEDS)} seeds")
    print(f"[sweep] Step counts: {STEP_COUNTS}")
    print(f"[sweep] Seeds: {SEEDS}")
    print(f"[sweep] Config: recycling=0, gamma_0=0.0, TF32, bf16_trunk, bypass+warmup")

    # Launch all in parallel via .map()
    raw_results = list(evaluate_single.map(steps_list, seeds_list))

    # Parse results
    all_results = [json.loads(r) for r in raw_results]

    # Load baseline from config.yaml
    # Hardcode baseline values to avoid yaml import issues in Modal's local entrypoint
    eval_config = {
        "baseline": {
            "mean_wall_time_s": 53.56664509866667,
            "mean_plddt": 0.7169803380966187,
            "per_complex": [
                {"name": "small_complex", "complex_plddt": 0.8344780802726746},
                {"name": "medium_complex", "complex_plddt": 0.5094943046569824},
                {"name": "large_complex", "complex_plddt": 0.806968629360199},
            ],
        }
    }
    baseline = eval_config.get("baseline", {})
    baseline_mean_time = baseline.get("mean_wall_time_s", 53.567)
    baseline_mean_plddt = baseline.get("mean_plddt", 0.7170)
    baseline_per_complex = {pc["name"]: pc for pc in baseline.get("per_complex", [])}

    # Organize by step count
    by_steps: dict[int, list] = {}
    for r in all_results:
        steps = r["steps"]
        if steps not in by_steps:
            by_steps[steps] = []
        by_steps[steps].append(r)

    # Compute aggregates per step count
    summary = {}
    for steps in sorted(by_steps.keys()):
        runs = by_steps[steps]

        # Collect per-complex metrics across seeds
        complex_metrics: dict[str, dict] = {}
        for r in runs:
            for pc in r["per_complex"]:
                name = pc["name"]
                if name not in complex_metrics:
                    complex_metrics[name] = {
                        "wall_times": [], "predict_times": [],
                        "plddts": [], "iptms": [], "errors": [],
                    }
                if pc.get("error"):
                    complex_metrics[name]["errors"].append(pc["error"])
                else:
                    if pc.get("wall_time_s") is not None:
                        complex_metrics[name]["wall_times"].append(pc["wall_time_s"])
                    if pc.get("predict_only_s") is not None:
                        complex_metrics[name]["predict_times"].append(pc["predict_only_s"])
                    plddt = pc.get("quality", {}).get("complex_plddt")
                    if plddt is not None:
                        complex_metrics[name]["plddts"].append(plddt)
                    iptm = pc.get("quality", {}).get("iptm")
                    if iptm is not None:
                        complex_metrics[name]["iptms"].append(iptm)

        # Check if all complexes succeeded across all seeds
        all_ok = True
        per_complex_summary = {}
        for name, m in complex_metrics.items():
            if m["errors"]:
                all_ok = False
                per_complex_summary[name] = {"error": m["errors"][0]}
                continue

            n = len(m["plddts"])
            mean_plddt = sum(m["plddts"]) / n if n > 0 else None
            std_plddt = (sum((x - mean_plddt)**2 for x in m["plddts"]) / n)**0.5 if n > 1 else 0
            mean_wall = sum(m["wall_times"]) / len(m["wall_times"]) if m["wall_times"] else None
            mean_predict = sum(m["predict_times"]) / len(m["predict_times"]) if m["predict_times"] else None
            mean_iptm = sum(m["iptms"]) / len(m["iptms"]) if m["iptms"] else None

            # Per-complex quality gate
            bl_plddt = baseline_per_complex.get(name, {}).get("complex_plddt")
            regression_pp = None
            if bl_plddt is not None and mean_plddt is not None:
                regression_pp = (bl_plddt - mean_plddt) * 100.0

            per_complex_summary[name] = {
                "mean_plddt": mean_plddt,
                "std_plddt": std_plddt,
                "mean_wall_s": mean_wall,
                "mean_predict_s": mean_predict,
                "mean_iptm": mean_iptm,
                "n_seeds": n,
                "regression_pp": regression_pp,
            }

        # Aggregate across complexes
        all_plddts = [v["mean_plddt"] for v in per_complex_summary.values()
                      if isinstance(v.get("mean_plddt"), (int, float))]
        all_walls = [v["mean_wall_s"] for v in per_complex_summary.values()
                     if isinstance(v.get("mean_wall_s"), (int, float))]
        all_predicts = [v["mean_predict_s"] for v in per_complex_summary.values()
                        if isinstance(v.get("mean_predict_s"), (int, float))]

        mean_plddt_agg = sum(all_plddts) / len(all_plddts) if all_plddts else None
        mean_wall_agg = sum(all_walls) / len(all_walls) if all_walls else None
        mean_predict_agg = sum(all_predicts) / len(all_predicts) if all_predicts else None

        # Quality gates
        passes_mean_gate = True
        passes_per_complex_gate = True
        if mean_plddt_agg is not None:
            mean_regression = (baseline_mean_plddt - mean_plddt_agg) * 100.0
            if mean_regression > 2.0:
                passes_mean_gate = False
        else:
            passes_mean_gate = False

        for name, v in per_complex_summary.items():
            reg = v.get("regression_pp")
            if reg is not None and reg > 5.0:
                passes_per_complex_gate = False

        passes_all = all_ok and passes_mean_gate and passes_per_complex_gate

        speedup = baseline_mean_time / mean_wall_agg if mean_wall_agg and mean_wall_agg > 0 else None

        summary[steps] = {
            "all_ok": all_ok,
            "per_complex": per_complex_summary,
            "mean_plddt": mean_plddt_agg,
            "mean_wall_s": mean_wall_agg,
            "mean_predict_s": mean_predict_agg,
            "speedup": speedup,
            "passes_mean_gate": passes_mean_gate,
            "passes_per_complex_gate": passes_per_complex_gate,
            "passes_all": passes_all,
        }

    # Print results table
    print("\n" + "=" * 100)
    print("ODE STEP SWEEP RESULTS (bypass + warmup, recycling=0, gamma_0=0.0)")
    print("=" * 100)
    print(f"\nBaseline: mean_wall={baseline_mean_time:.1f}s, mean_pLDDT={baseline_mean_plddt:.4f}")
    print(f"Quality gates: mean pLDDT regression <= 2pp, per-complex <= 5pp\n")

    print(f"{'Steps':>5} | {'Wall(s)':>8} | {'Pred(s)':>8} | {'pLDDT':>8} | {'Regr(pp)':>8} | "
          f"{'Speedup':>8} | {'MeanGate':>8} | {'PCGate':>8} | {'Pass':>5}")
    print("-" * 100)

    for steps in sorted(summary.keys()):
        s = summary[steps]
        wall = f"{s['mean_wall_s']:.1f}" if s['mean_wall_s'] else "ERR"
        pred = f"{s['mean_predict_s']:.1f}" if s['mean_predict_s'] else "N/A"
        plddt = f"{s['mean_plddt']:.4f}" if s['mean_plddt'] else "ERR"

        regr = ""
        if s['mean_plddt'] is not None:
            regr_val = (baseline_mean_plddt - s['mean_plddt']) * 100.0
            regr = f"{regr_val:+.2f}"

        spd = f"{s['speedup']:.2f}x" if s['speedup'] else "ERR"
        mg = "PASS" if s['passes_mean_gate'] else "FAIL"
        pg = "PASS" if s['passes_per_complex_gate'] else "FAIL"
        pa = "YES" if s['passes_all'] else "NO"

        print(f"{steps:>5} | {wall:>8} | {pred:>8} | {plddt:>8} | {regr:>8} | "
              f"{spd:>8} | {mg:>8} | {pg:>8} | {pa:>5}")

    # Per-complex breakdown
    print("\n" + "=" * 100)
    print("PER-COMPLEX BREAKDOWN")
    print("=" * 100)

    for steps in sorted(summary.keys()):
        s = summary[steps]
        print(f"\n--- Steps={steps} ---")
        for name, v in s["per_complex"].items():
            if "error" in v:
                print(f"  {name}: ERROR - {v['error'][:100]}")
            else:
                bl_p = baseline_per_complex.get(name, {}).get("complex_plddt", 0)
                reg = v.get("regression_pp")
                reg_str = f"{reg:+.2f}pp" if reg is not None else "N/A"
                gate = "PASS" if (reg is None or reg <= 5.0) else "FAIL"
                print(f"  {name}: pLDDT={v['mean_plddt']:.4f}+/-{v['std_plddt']:.4f} "
                      f"(bl={bl_p:.4f}, {reg_str}) {gate} | "
                      f"wall={v['mean_wall_s']:.1f}s, pred={v['mean_predict_s']:.1f}s"
                      if v['mean_predict_s'] else
                      f"  {name}: pLDDT={v['mean_plddt']:.4f}+/-{v['std_plddt']:.4f} "
                      f"(bl={bl_p:.4f}, {reg_str}) {gate} | "
                      f"wall={v['mean_wall_s']:.1f}s")

    # Find best passing config
    print("\n" + "=" * 100)
    best_steps = None
    best_speedup = 0
    for steps in sorted(summary.keys()):
        s = summary[steps]
        if s["passes_all"] and s["speedup"] and s["speedup"] > best_speedup:
            best_speedup = s["speedup"]
            best_steps = steps

    if best_steps is not None:
        print(f"BEST PASSING CONFIG: ODE-{best_steps} with speedup={best_speedup:.2f}x")
    else:
        print("NO CONFIG PASSED ALL QUALITY GATES")
    print("=" * 100)

    # Save full results as JSON
    output = {
        "sweep_config": {
            "step_counts": STEP_COUNTS,
            "seeds": SEEDS,
            "base_config": BASE_CONFIG,
        },
        "baseline": {
            "mean_wall_time_s": baseline_mean_time,
            "mean_plddt": baseline_mean_plddt,
            "per_complex": dict(baseline_per_complex),
        },
        "summary": summary,
        "raw_results": all_results,
        "best_steps": best_steps,
        "best_speedup": best_speedup,
    }

    # Write to stdout for capture
    print("\n--- FULL JSON RESULTS ---")
    print(json.dumps(output, indent=2, default=str))
