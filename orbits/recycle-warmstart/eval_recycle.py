"""Evaluator for recycle-warmstart: compare recycling_steps=0 vs 1.

Tests whether recycling provides any quality benefit on eval-v5 test cases
when using the bypass-lightning wrapper with ODE-12 sampling.

Runs 3 seeds in parallel using Modal .map() for each config.

Usage:
    modal run orbits/recycle-warmstart/eval_recycle.py --mode compare
    modal run orbits/recycle-warmstart/eval_recycle.py --mode compare --validate
"""

from __future__ import annotations

import json
import math
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

app = modal.App("boltz-eval-recycle-warmstart", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)

# ---------------------------------------------------------------------------
# Configs to compare
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 7]

BASE_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
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
    print(f"[eval] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


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
            cmd, capture_output=True, text=True, timeout=1800,
        )
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"boltz predict exited with code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse phase timestamps
        predict_start = predict_end = None
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


# ---------------------------------------------------------------------------
# Modal function: evaluate a single (config, seed, test_case) combination
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_cache},
)
def evaluate_single(config_json: str) -> str:
    """Run a single (config, seed, test_case) and return results JSON."""
    config = json.loads(config_json)

    tc_name = config.pop("_test_case_name")
    tc_yaml_rel = config.pop("_test_case_yaml")
    label = config.pop("_label", "unknown")

    tc_yaml = Path("/eval") / tc_yaml_rel
    if not tc_yaml.exists():
        return json.dumps({
            "label": label,
            "test_case": tc_name,
            "seed": config.get("seed"),
            "error": f"Test case YAML not found: {tc_yaml}",
        })

    # MSA cache
    msa_cache_root = Path("/msa_cache")
    work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    effective_yaml = tc_yaml
    if msa_cache_root.exists() and any(msa_cache_root.iterdir()):
        cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
        if cached_yaml is not None:
            effective_yaml = cached_yaml
            config["_msa_cached"] = True

    print(f"[eval] Running {label} | {tc_name} | seed={config.get('seed')} | "
          f"recycle={config.get('recycling_steps')}")

    pred_result = _run_boltz_bypass(effective_yaml, work_dir, config)

    return json.dumps({
        "label": label,
        "test_case": tc_name,
        "seed": config.get("seed"),
        "recycling_steps": config.get("recycling_steps"),
        "wall_time_s": pred_result.get("wall_time_s"),
        "predict_only_s": pred_result.get("predict_only_s"),
        "quality": pred_result.get("quality", {}),
        "error": pred_result.get("error"),
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "compare",
    validate: bool = False,
):
    """Compare recycling_steps=0 vs 1 across 3 seeds.

    Usage:
        modal run orbits/recycle-warmstart/eval_recycle.py --mode compare
        modal run orbits/recycle-warmstart/eval_recycle.py --mode compare --validate
    """
    # Load config locally — pyyaml may not be installed locally, fall back to
    # simple parsing if needed
    eval_config = None
    try:
        import yaml
        config_path = EVAL_DIR / "config.yaml"
        with config_path.open() as f:
            eval_config = yaml.safe_load(f)
    except ImportError:
        # Fallback: read config.yaml and extract test_cases manually
        import re
        config_path = EVAL_DIR / "config.yaml"
        text = config_path.read_text()
        # Extract test cases from YAML manually
        eval_config = {
            "test_cases": [
                {"name": "small_complex", "yaml": "test_cases/small_complex.yaml"},
                {"name": "medium_complex", "yaml": "test_cases/medium_complex.yaml"},
                {"name": "large_complex", "yaml": "test_cases/large_complex.yaml"},
            ],
            "baseline": {
                "mean_wall_time_s": 53.57,
                "mean_plddt": 0.7170,
                "per_complex": [
                    {"name": "small_complex", "complex_plddt": 0.8345},
                    {"name": "medium_complex", "complex_plddt": 0.5095},
                    {"name": "large_complex", "complex_plddt": 0.8070},
                ],
            }
        }

    test_cases = eval_config.get("test_cases", [])

    # Build all jobs: 2 recycling configs x 3 test cases x 3 seeds = 18 jobs
    recycling_values = [0, 1]
    seeds = SEEDS

    jobs = []
    for recycle in recycling_values:
        label = f"recycle={recycle}"
        for tc in test_cases:
            for seed in seeds:
                cfg = dict(BASE_CONFIG)
                cfg["recycling_steps"] = recycle
                cfg["seed"] = seed
                cfg["_test_case_name"] = tc["name"]
                cfg["_test_case_yaml"] = tc["yaml"]
                cfg["_label"] = label
                jobs.append(json.dumps(cfg))

    print(f"[recycle-warmstart] Launching {len(jobs)} jobs ({len(recycling_values)} configs x "
          f"{len(test_cases)} test cases x {len(seeds)} seeds)")

    # Run all jobs in parallel via Modal .map()
    results_raw = list(evaluate_single.map(jobs))

    # Parse results
    all_results = [json.loads(r) for r in results_raw]

    # Organize by config
    by_config: dict[str, list] = {}
    for r in all_results:
        label = r["label"]
        if label not in by_config:
            by_config[label] = []
        by_config[label].append(r)

    # Print detailed results
    baseline_data = eval_config.get("baseline", {})
    baseline_time = baseline_data.get("mean_wall_time_s", 53.57)
    baseline_plddt = baseline_data.get("mean_plddt", 0.7170)

    print(f"\n{'='*90}")
    print("RECYCLING COMPARISON: recycling_steps=0 vs recycling_steps=1")
    print(f"{'='*90}")
    print(f"Baseline (200-step SDE, 3 recycle): mean_time={baseline_time:.1f}s, mean_pLDDT={baseline_plddt:.4f}")
    print()

    summary = {}

    for label in sorted(by_config.keys()):
        results = by_config[label]
        errors = [r for r in results if r.get("error")]
        if errors:
            print(f"\n{label}: {len(errors)} ERRORS")
            for e in errors:
                print(f"  {e['test_case']} seed={e['seed']}: {str(e['error'])[:200]}")
            continue

        print(f"\n--- {label} ---")

        # Group by test case
        by_tc: dict[str, list] = {}
        for r in results:
            tc = r["test_case"]
            if tc not in by_tc:
                by_tc[tc] = []
            by_tc[tc].append(r)

        tc_summaries = []
        for tc_name in sorted(by_tc.keys()):
            tc_results = by_tc[tc_name]
            times = [r["wall_time_s"] for r in tc_results if r.get("wall_time_s")]
            pred_times = [r["predict_only_s"] for r in tc_results if r.get("predict_only_s")]
            plddts = [r["quality"].get("complex_plddt") for r in tc_results
                      if r.get("quality", {}).get("complex_plddt") is not None]
            iptms = [r["quality"].get("iptm") for r in tc_results
                     if r.get("quality", {}).get("iptm") is not None]

            mean_time = sum(times) / len(times) if times else None
            mean_pred = sum(pred_times) / len(pred_times) if pred_times else None
            mean_plddt = sum(plddts) / len(plddts) if plddts else None
            mean_iptm = sum(iptms) / len(iptms) if iptms else None

            std_plddt = (sum((p - mean_plddt)**2 for p in plddts) / len(plddts))**0.5 if plddts and len(plddts) > 1 else 0

            tc_summaries.append({
                "name": tc_name,
                "mean_time": mean_time,
                "mean_pred": mean_pred,
                "mean_plddt": mean_plddt,
                "std_plddt": std_plddt,
                "mean_iptm": mean_iptm,
                "plddts": plddts,
                "times": times,
            })

            plddt_str = f"{mean_plddt:.4f} +/- {std_plddt:.4f}" if mean_plddt else "N/A"
            time_str = f"{mean_time:.1f}s" if mean_time else "N/A"
            pred_str = f" (pred={mean_pred:.1f}s)" if mean_pred else ""
            seeds_str = " | ".join(f"s={r['seed']}:{r['quality'].get('complex_plddt', 'N/A'):.4f}"
                                   for r in tc_results if r.get("quality", {}).get("complex_plddt"))
            iptm_str = f"{mean_iptm:.4f}" if mean_iptm else "N/A"
            print(f"  {tc_name}: time={time_str}{pred_str}, pLDDT={plddt_str}, iptm={iptm_str}")
            print(f"    per-seed: {seeds_str}")

        # Aggregate across test cases
        all_times = [s["mean_time"] for s in tc_summaries if s["mean_time"]]
        all_plddts = [s["mean_plddt"] for s in tc_summaries if s["mean_plddt"]]

        if all_times and all_plddts:
            mean_time_all = sum(all_times) / len(all_times)
            mean_plddt_all = sum(all_plddts) / len(all_plddts)
            speedup = baseline_time / mean_time_all if mean_time_all > 0 else 0
            plddt_delta = (mean_plddt_all - baseline_plddt) * 100

            summary[label] = {
                "mean_time": mean_time_all,
                "mean_plddt": mean_plddt_all,
                "speedup": speedup,
                "plddt_delta_pp": plddt_delta,
                "per_complex": tc_summaries,
            }

            print(f"  AGGREGATE: time={mean_time_all:.1f}s, pLDDT={mean_plddt_all:.4f}, "
                  f"speedup={speedup:.2f}x, pLDDT delta={plddt_delta:+.2f}pp")

    # Final comparison table
    print(f"\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"{'Config':<20} {'Time(s)':>8} {'pLDDT':>8} {'Speedup':>8} {'pLDDT delta':>12} {'Gate':>6}")
    print("-" * 70)

    for label in sorted(summary.keys()):
        s = summary[label]
        gate = "PASS" if s["plddt_delta_pp"] >= -2.0 else "FAIL"
        print(f"{label:<20} {s['mean_time']:>8.1f} {s['mean_plddt']:>8.4f} "
              f"{s['speedup']:>7.2f}x {s['plddt_delta_pp']:>+11.2f}pp {gate:>6}")

    # Print per-complex breakdown
    print(f"\n{'='*90}")
    print("PER-COMPLEX PLDDT COMPARISON")
    print(f"{'='*90}")

    tc_names = sorted(set(
        r["test_case"] for results in by_config.values() for r in results
    ))

    print(f"{'Test Case':<20}", end="")
    for label in sorted(summary.keys()):
        print(f" {label:>20}", end="")
    print(f" {'Baseline':>12}")
    print("-" * (20 + 20 * len(summary) + 12))

    baseline_per_complex = {pc["name"]: pc for pc in baseline_data.get("per_complex", [])}

    for tc_name in tc_names:
        print(f"{tc_name:<20}", end="")
        for label in sorted(summary.keys()):
            tc_data = next((t for t in summary[label]["per_complex"] if t["name"] == tc_name), None)
            if tc_data and tc_data["mean_plddt"]:
                print(f" {tc_data['mean_plddt']:>20.4f}", end="")
            else:
                print(f" {'N/A':>20}", end="")
        bl = baseline_per_complex.get(tc_name, {})
        bl_plddt = bl.get("complex_plddt")
        print(f" {bl_plddt:>12.4f}" if bl_plddt else f" {'N/A':>12}")

    # Full JSON dump
    print("\n\n--- FULL RESULTS JSON ---")
    print(json.dumps({
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_complex"}
                    for k, v in summary.items()},
        "raw": all_results,
    }, indent=2, default=str))
