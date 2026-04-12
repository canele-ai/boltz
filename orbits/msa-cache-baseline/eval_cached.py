"""Evaluator with MSA caching for clean GPU-only timing.

Phase 1: Generate MSAs once using the MSA server, save to a Modal Volume.
Phase 2: Run evaluations with pre-cached MSAs (no network calls).

This removes MSA server latency (~5-30s per complex) from timing measurements,
giving clean GPU-only baselines for all future optimization orbits.

Usage:
    # Phase 1: Generate and cache MSAs
    modal run orbits/msa-cache-baseline/eval_cached.py --phase cache-msas

    # Phase 2: Run baseline (200 steps, 3 recycles) with cached MSAs
    modal run orbits/msa-cache-baseline/eval_cached.py --phase eval-baseline --num-runs 3

    # Phase 2: Run ODE-20/0r winner with cached MSAs
    modal run orbits/msa-cache-baseline/eval_cached.py --phase eval-winner --num-runs 3

    # Phase 2: Run both baseline and winner
    modal run orbits/msa-cache-baseline/eval_cached.py --phase eval-both --num-runs 3
"""

from __future__ import annotations

import json
import math
import os
import shutil
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
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_ode.py"),
        remote_path="/eval/boltz_wrapper_ode.py",
    )
)

app = modal.App("boltz-eval-msa-cache", image=boltz_image)

# Persistent volume for MSA cache
msa_volume = modal.Volume.from_name("boltz-msa-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

BASELINE_CONFIG = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
    "gamma_0": 0.8,
    "noise_scale": 1.003,
}

WINNER_CONFIG = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
    "gamma_0": 0.0,    # deterministic ODE
    "noise_scale": 1.003,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
    use_msa_server: bool = False,
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction.

    If use_msa_server=False, the input YAML must already have msa fields
    pointing to cached MSA files.
    """
    # Use ODE wrapper to support gamma_0
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

    if use_msa_server:
        cmd.append("--use_msa_server")
    # If not using MSA server, the input YAML has msa: fields pointing to files

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
                f"STDOUT: {proc.stdout[-1000:] if proc.stdout else '(empty)'}\n"
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


def _create_cached_yaml(original_yaml: Path, msa_cache_dir: Path, work_dir: Path) -> Path:
    """Create a modified input YAML with msa fields pointing to cached MSA files.

    Boltz stores MSAs as {target_id}_{entity_id}.csv in the msa/ subdirectory.
    We need to set the msa field on each protein chain to point to the cached file.
    """
    import yaml

    with original_yaml.open() as f:
        data = yaml.safe_load(f)

    target_name = original_yaml.stem

    # Find cached MSA files for this target
    # Boltz names them like: {target_name}_{entity_id}.csv
    # entity_id is sequential (0, 1, 2, ...) for each unique sequence
    msa_files = sorted(msa_cache_dir.glob(f"{target_name}_*.csv"))
    print(f"[msa-cache] Found {len(msa_files)} cached MSA files for {target_name}: {[f.name for f in msa_files]}")

    if not msa_files:
        # Try listing all files in cache dir
        all_files = sorted(msa_cache_dir.glob("*.csv"))
        print(f"[msa-cache] All CSV files in cache: {[f.name for f in all_files]}")
        # Fall back to looking for any CSV files matching the pattern
        msa_files = all_files

    # Copy MSA files to work directory so paths are accessible
    local_msa_dir = work_dir / "cached_msas"
    local_msa_dir.mkdir(parents=True, exist_ok=True)

    # Map entity index to MSA file path
    entity_msa_map = {}
    for msa_file in msa_files:
        # Copy to local dir
        local_path = local_msa_dir / msa_file.name
        shutil.copy2(msa_file, local_path)
        # Extract entity id from filename pattern: {target}_{entity_id}.csv
        parts = msa_file.stem.split("_")
        if len(parts) >= 2:
            entity_id = parts[-1]
            entity_msa_map[entity_id] = str(local_path)

    print(f"[msa-cache] Entity MSA map: {entity_msa_map}")

    # Modify YAML to add msa field to each protein sequence
    if "sequences" in data:
        entity_idx = 0
        for seq_entry in data["sequences"]:
            if "protein" in seq_entry:
                entity_key = str(entity_idx)
                if entity_key in entity_msa_map:
                    seq_entry["protein"]["msa"] = entity_msa_map[entity_key]
                    print(f"[msa-cache] Set msa for protein entity {entity_idx}: {entity_msa_map[entity_key]}")
                else:
                    print(f"[msa-cache] WARNING: No cached MSA for protein entity {entity_idx}")
                entity_idx += 1

    # Write modified YAML
    cached_yaml = work_dir / f"{target_name}_cached.yaml"
    with cached_yaml.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[msa-cache] Created cached YAML: {cached_yaml}")
    with cached_yaml.open() as f:
        print(f"[msa-cache] YAML content:\n{f.read()}")

    return cached_yaml


# ---------------------------------------------------------------------------
# Phase 1: Generate and cache MSAs
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_volume},
)
def cache_msas() -> str:
    """Run baseline once with MSA server to generate and cache MSA files."""
    import yaml

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results = {"cached_targets": [], "errors": []}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["errors"].append(f"YAML not found: {tc_yaml}")
            continue

        print(f"\n[msa-cache] Generating MSAs for {tc_name}...")

        work_dir = Path(f"/tmp/boltz_msa_gen/{tc_name}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Run with MSA server to generate MSAs
        pred_result = _run_boltz_prediction(
            tc_yaml, work_dir, BASELINE_CONFIG, use_msa_server=True
        )

        if pred_result["error"]:
            results["errors"].append(f"{tc_name}: {pred_result['error']}")
            continue

        # Find generated MSA files
        target_name = tc_yaml.stem
        msa_dir = work_dir / f"boltz_results_{target_name}" / "msa"

        if not msa_dir.exists():
            # Search more broadly
            print(f"[msa-cache] MSA dir not found at {msa_dir}, searching...")
            for p in work_dir.rglob("*.csv"):
                print(f"[msa-cache] Found CSV: {p}")
            for p in work_dir.rglob("*.a3m"):
                print(f"[msa-cache] Found A3M: {p}")
            # Also check the msa subdirectory of the output
            alt_msa_dir = work_dir / "msa"
            if alt_msa_dir.exists():
                msa_dir = alt_msa_dir
                print(f"[msa-cache] Found MSA dir at {msa_dir}")
            else:
                results["errors"].append(f"{tc_name}: MSA directory not found")
                # List all dirs for debugging
                for p in work_dir.rglob("*"):
                    if p.is_dir():
                        print(f"[msa-cache] Dir: {p}")
                continue

        # Copy MSA files to volume
        cache_dir = Path(f"/msa_cache/{target_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        msa_files = list(msa_dir.glob("*.csv")) + list(msa_dir.glob("*.a3m"))
        for msa_file in msa_files:
            dest = cache_dir / msa_file.name
            shutil.copy2(msa_file, dest)
            print(f"[msa-cache] Cached: {msa_file.name} -> {dest}")

        results["cached_targets"].append({
            "name": tc_name,
            "msa_files": [f.name for f in msa_files],
            "wall_time_s": pred_result["wall_time_s"],
            "quality": pred_result["quality"],
        })

    # Commit the volume
    msa_volume.commit()

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Phase 2: Evaluate with cached MSAs
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_volume},
)
def evaluate_cached(config_json: str, num_runs: int = 3) -> str:
    """Run evaluation using pre-cached MSAs for clean GPU-only timing."""
    import statistics

    config = json.loads(config_json)
    merged = {**BASELINE_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": merged,
        "num_runs": num_runs,
        "msa_cached": True,
        "per_complex": [],
        "aggregate": {},
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]
        target_name = tc_yaml.stem

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None,
                "quality": {},
            })
            continue

        # Check for cached MSAs
        cache_dir = Path(f"/msa_cache/{target_name}")
        if not cache_dir.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"No cached MSAs found at {cache_dir}. Run --phase cache-msas first.",
                "wall_time_s": None,
                "quality": {},
            })
            continue

        cached_files = list(cache_dir.glob("*.csv")) + list(cache_dir.glob("*.a3m"))
        print(f"[eval-cached] {tc_name}: found {len(cached_files)} cached MSA files")

        run_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Create input YAML with cached MSA references
            cached_yaml = _create_cached_yaml(tc_yaml, cache_dir, work_dir)

            print(
                f"[eval-cached] {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={merged['sampling_steps']}, recycle={merged['recycling_steps']}, "
                f"gamma_0={merged.get('gamma_0', 0.8)}"
            )

            pred_result = _run_boltz_prediction(
                cached_yaml, work_dir, merged, use_msa_server=False
            )

            if pred_result["error"]:
                last_error = pred_result["error"]
                print(f"[eval-cached] ERROR: {last_error}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
            run_qualities.append(pred_result["quality"])

        if last_error is not None:
            entry: dict[str, Any] = {
                "name": tc_name,
                "wall_time_s": None,
                "quality": {},
                "error": last_error,
                "run_times": [],
            }
        else:
            median_time = statistics.median(run_times) if run_times else None
            all_plddts = [
                q["complex_plddt"] for q in run_qualities if "complex_plddt" in q
            ]
            mean_plddt = (sum(all_plddts) / len(all_plddts)) if all_plddts else None

            merged_quality: dict[str, Any] = {}
            if run_qualities:
                merged_quality = dict(run_qualities[-1])
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
        failed_names = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        results["aggregate"] = {
            "error": f"Not all test cases succeeded. Failed: {failed_names}",
            "num_successful": len(successful),
            "num_total": len(test_cases),
        }
        return json.dumps(results, indent=2)

    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [
            r["quality"]["complex_plddt"]
            for r in successful
            if "complex_plddt" in r["quality"]
            and r["quality"]["complex_plddt"] is not None
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

        # Compare to baseline
        baseline = eval_config.get("baseline")
        if baseline:
            baseline_time = baseline.get("mean_wall_time_s")
            baseline_plddt = baseline.get("mean_plddt")
            if baseline_time and mean_time > 0:
                results["aggregate"]["speedup_vs_msa_baseline"] = baseline_time / mean_time
            if baseline_plddt and plddts:
                mean_plddt_val = sum(plddts) / len(plddts)
                results["aggregate"]["plddt_delta_pp"] = (mean_plddt_val - baseline_plddt) * 100.0
                regression = (baseline_plddt - mean_plddt_val) * 100.0
                results["aggregate"]["passes_quality_gate"] = regression <= 2.0

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    phase: str = "eval-both",
    num_runs: int = 3,
):
    """MSA caching evaluator.

    Phases:
        cache-msas:     Generate and cache MSAs to Modal Volume
        eval-baseline:  Run 200s/3r baseline with cached MSAs
        eval-winner:    Run ODE-20/0r winner with cached MSAs
        eval-both:      Run both baseline and winner
    """

    if phase == "cache-msas":
        print("[msa-cache] Phase 1: Generating and caching MSAs...")
        result_json = cache_msas.remote()
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

        if result.get("errors"):
            print(f"\n[msa-cache] ERRORS: {result['errors']}")
        else:
            print(f"\n[msa-cache] Successfully cached MSAs for {len(result['cached_targets'])} targets")

    elif phase in ("eval-baseline", "eval-both"):
        print(f"\n[msa-cache] Running BASELINE (200s/3r) with cached MSAs, {num_runs} runs...")
        baseline_json = evaluate_cached.remote(json.dumps(BASELINE_CONFIG), num_runs=num_runs)
        baseline = json.loads(baseline_json)
        print(json.dumps(baseline, indent=2))

        agg = baseline.get("aggregate", {})
        print(f"\n[msa-cache] Baseline mean time: {agg.get('mean_wall_time_s', 'N/A'):.2f}s")
        print(f"[msa-cache] Baseline mean pLDDT: {agg.get('mean_plddt', 'N/A')}")

        # Save baseline results
        out_path = Path(__file__).parent / "baseline_cached_results.json"
        with out_path.open("w") as f:
            f.write(json.dumps(baseline, indent=2))
        print(f"[msa-cache] Saved to {out_path}")

        if phase == "eval-both":
            print(f"\n[msa-cache] Running WINNER (ODE-20/0r) with cached MSAs, {num_runs} runs...")
            winner_json = evaluate_cached.remote(json.dumps(WINNER_CONFIG), num_runs=num_runs)
            winner = json.loads(winner_json)
            print(json.dumps(winner, indent=2))

            agg_w = winner.get("aggregate", {})
            print(f"\n[msa-cache] Winner mean time: {agg_w.get('mean_wall_time_s', 'N/A'):.2f}s")
            print(f"[msa-cache] Winner mean pLDDT: {agg_w.get('mean_plddt', 'N/A')}")

            # Compute cached speedup
            baseline_time = agg.get("mean_wall_time_s")
            winner_time = agg_w.get("mean_wall_time_s")
            if baseline_time and winner_time and winner_time > 0:
                cached_speedup = baseline_time / winner_time
                print(f"\n[msa-cache] CACHED SPEEDUP: {cached_speedup:.2f}x")

            out_path_w = Path(__file__).parent / "winner_cached_results.json"
            with out_path_w.open("w") as f:
                f.write(json.dumps(winner, indent=2))
            print(f"[msa-cache] Saved to {out_path_w}")

    elif phase == "eval-winner":
        print(f"\n[msa-cache] Running WINNER (ODE-20/0r) with cached MSAs, {num_runs} runs...")
        winner_json = evaluate_cached.remote(json.dumps(WINNER_CONFIG), num_runs=num_runs)
        winner = json.loads(winner_json)
        print(json.dumps(winner, indent=2))

        agg_w = winner.get("aggregate", {})
        print(f"\n[msa-cache] Winner mean time: {agg_w.get('mean_wall_time_s', 'N/A'):.2f}s")
        print(f"[msa-cache] Winner mean pLDDT: {agg_w.get('mean_plddt', 'N/A')}")

        out_path_w = Path(__file__).parent / "winner_cached_results.json"
        with out_path_w.open("w") as f:
            f.write(json.dumps(winner, indent=2))
        print(f"[msa-cache] Saved to {out_path_w}")

    else:
        print(f"Unknown phase: {phase}. Use: cache-msas, eval-baseline, eval-winner, eval-both")
