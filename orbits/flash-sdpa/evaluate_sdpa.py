"""Evaluator for the flash-sdpa orbit.

This is a thin wrapper around the standard evaluator that:
1. Includes the SDPA patch files in the Modal container image
2. Swaps the default boltz_wrapper.py for boltz_wrapper_sdpa.py
3. Otherwise follows the same evaluation protocol

Usage:
    # Single run (iteration):
    modal run orbits/flash-sdpa/evaluate_sdpa.py --seed 42

    # Three seeds in parallel:
    modal run orbits/flash-sdpa/evaluate_sdpa.py --seeds '42,123,7'

    # Sanity check:
    modal run orbits/flash-sdpa/evaluate_sdpa.py --sanity-check --seed 42

    # No SDPA (baseline comparison):
    modal run orbits/flash-sdpa/evaluate_sdpa.py --no-sdpa --seed 42
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
# Modal image: same as base evaluator + orbit patch files
# ---------------------------------------------------------------------------

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "boltz==2.2.1",
    )
    # Mount the eval directory (test cases, config, base wrapper)
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    # Mount the orbit directory (SDPA patch + wrapper)
    .add_local_dir(str(ORBIT_DIR), remote_path="/orbit")
)

app = modal.App("boltz-eval-sdpa", image=boltz_image)

# ---------------------------------------------------------------------------
# Default config (same as base evaluator)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Helpers (run inside Modal container)
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    """Parse Boltz confidence summary JSON from the prediction output."""
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
    use_sdpa: bool = True,
    sdpa_bf16: bool = False,
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction with optional SDPA patches."""

    # Choose wrapper: SDPA or base
    if use_sdpa:
        wrapper = "/orbit/boltz_wrapper_sdpa.py"
    else:
        wrapper = "/eval/boltz_wrapper.py"

    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--no_kernels",
    ]

    # SDPA bf16 flag
    if use_sdpa and sdpa_bf16:
        cmd.append("--sdpa_bf16")

    # MSA handling
    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Matmul precision
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
                f"STDERR: {proc.stderr[-3000:] if proc.stderr else '(empty)'}\n"
                f"STDOUT: {proc.stdout[-1000:] if proc.stdout else '(empty)'}"
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
# Modal function: evaluate one seed on full test set
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=7200)
def evaluate_one_seed(
    config_json: str,
    sanity_check: bool = False,
    use_sdpa: bool = True,
    sdpa_bf16: bool = False,
) -> str:
    """Evaluate a single seed on the full test set. Returns JSON results."""
    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    if sanity_check:
        test_cases = [test_cases[0]] if test_cases else []
        merged["sampling_steps"] = min(merged.get("sampling_steps", 200), 10)

    results: dict[str, Any] = {
        "config": merged,
        "sanity_check": sanity_check,
        "use_sdpa": use_sdpa,
        "sdpa_bf16": sdpa_bf16,
        "per_complex": [],
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

        print(
            f"[eval-sdpa] Running {tc_name} seed={merged.get('seed')} "
            f"steps={merged['sampling_steps']} sdpa={use_sdpa} bf16={sdpa_bf16}"
        )

        pred_result = _run_boltz_prediction(
            tc_yaml, work_dir, merged, use_sdpa=use_sdpa, sdpa_bf16=sdpa_bf16
        )

        entry = {
            "name": tc_name,
            "wall_time_s": pred_result["wall_time_s"],
            "quality": pred_result["quality"],
            "error": pred_result["error"],
        }
        results["per_complex"].append(entry)

    # Compute aggregates
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
            and r["quality"]["complex_plddt"] is not None
            and isinstance(r["quality"]["complex_plddt"], (int, float))
            and not math.isnan(r["quality"]["complex_plddt"])
            and 0.0 <= r["quality"]["complex_plddt"] <= 1.0
        ]
        iptms = [
            r["quality"]["iptm"]
            for r in successful
            if "iptm" in r["quality"]
        ]

        baseline = eval_config.get("baseline", {})
        baseline_time = baseline.get("mean_wall_time_s")
        baseline_plddt = baseline.get("mean_plddt")

        agg: dict[str, Any] = {
            "num_successful": len(successful),
            "num_total": len(test_cases) if not sanity_check else 1,
            "total_wall_time_s": total_time,
            "mean_wall_time_s": mean_time,
            "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
            "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
        }

        if baseline_time and mean_time > 0:
            agg["speedup"] = baseline_time / mean_time
        if baseline_plddt is not None and plddts:
            mean_plddt = sum(plddts) / len(plddts)
            agg["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
            regression = (baseline_plddt - mean_plddt) * 100.0
            agg["passes_quality_gate"] = regression <= 2.0

            # Per-complex quality floor
            if baseline.get("per_complex"):
                baseline_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                for r in successful:
                    bl_case = baseline_by_name.get(r["name"])
                    if bl_case and bl_case.get("complex_plddt") is not None:
                        case_plddt = r["quality"].get("complex_plddt")
                        if case_plddt is not None:
                            case_regression = (bl_case["complex_plddt"] - case_plddt) * 100.0
                            if case_regression > 5.0:
                                agg["passes_quality_gate"] = False
                                agg.setdefault("per_complex_violations", {})[r["name"]] = f"-{case_regression:.1f}pp"

        results["aggregate"] = agg
    else:
        results["aggregate"] = {
            "error": "No successful test cases",
            "num_successful": 0,
            "num_total": len(test_cases),
        }

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    sanity_check: bool = False,
    seed: int = 42,
    seeds: str = "",
    no_sdpa: bool = False,
    sdpa_bf16: bool = False,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    matmul_precision: str = "highest",
):
    """Run SDPA evaluation.

    Examples:
        modal run orbits/flash-sdpa/evaluate_sdpa.py --sanity-check --seed 42
        modal run orbits/flash-sdpa/evaluate_sdpa.py --seeds '42,123,7'
        modal run orbits/flash-sdpa/evaluate_sdpa.py --no-sdpa --seeds '42,123,7'
        modal run orbits/flash-sdpa/evaluate_sdpa.py --sdpa-bf16 --seeds '42,123,7'
    """
    use_sdpa = not no_sdpa

    # Parse seeds
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    else:
        seed_list = [seed]

    mode = "bf16" if sdpa_bf16 else ("fp32" if use_sdpa else "disabled")
    print(f"[eval-sdpa] seeds={seed_list}, sdpa={mode}, steps={sampling_steps}")

    # Launch all seeds in parallel using Modal .map()
    configs = []
    for s in seed_list:
        cfg = {
            "sampling_steps": sampling_steps,
            "recycling_steps": recycling_steps,
            "matmul_precision": matmul_precision,
            "seed": s,
        }
        configs.append(json.dumps(cfg))

    # Parallel seed evaluation via Modal .map()
    # .map() takes one iterable per positional arg
    sanity_flags = [sanity_check] * len(configs)
    sdpa_flags = [use_sdpa] * len(configs)
    bf16_flags = [sdpa_bf16] * len(configs)

    results_by_seed = {}
    for result_json in evaluate_one_seed.map(
        configs,
        sanity_flags,
        sdpa_flags,
        bf16_flags,
    ):
        result = json.loads(result_json)
        seed_val = result["config"]["seed"]
        results_by_seed[seed_val] = result

    # Print per-seed results
    print("\n" + "=" * 70)
    print("PER-SEED RESULTS")
    print("=" * 70)
    for s in seed_list:
        r = results_by_seed.get(s, {})
        agg = r.get("aggregate", {})
        print(f"\nSeed {s}:")
        print(f"  Mean wall time: {agg.get('mean_wall_time_s', 'N/A'):.2f}s" if isinstance(agg.get('mean_wall_time_s'), (int, float)) else f"  Mean wall time: N/A")
        print(f"  Mean pLDDT:     {agg.get('mean_plddt', 'N/A')}")
        print(f"  Speedup:        {agg.get('speedup', 'N/A')}")
        print(f"  Quality gate:   {agg.get('passes_quality_gate', 'N/A')}")
        if agg.get("error"):
            print(f"  ERROR: {agg['error']}")
        for pc in r.get("per_complex", []):
            status = "OK" if pc["error"] is None else "FAIL"
            wall = f"{pc['wall_time_s']:.1f}s" if pc["wall_time_s"] else "N/A"
            plddt = pc["quality"].get("complex_plddt", "N/A")
            if isinstance(plddt, float):
                plddt = f"{plddt:.4f}"
            print(f"    {pc['name']:20s} {status:5s} time={wall:>8s} plddt={plddt}")

    # Aggregate across seeds
    if len(seed_list) > 1:
        print("\n" + "=" * 70)
        print("AGGREGATE ACROSS SEEDS")
        print("=" * 70)

        all_times = []
        all_plddts = []
        all_speedups = []
        for s in seed_list:
            r = results_by_seed.get(s, {})
            agg = r.get("aggregate", {})
            t = agg.get("mean_wall_time_s")
            p = agg.get("mean_plddt")
            sp = agg.get("speedup")
            if isinstance(t, (int, float)):
                all_times.append(t)
            if isinstance(p, (int, float)):
                all_plddts.append(p)
            if isinstance(sp, (int, float)):
                all_speedups.append(sp)

        if all_times:
            import statistics
            mean_t = statistics.mean(all_times)
            std_t = statistics.stdev(all_times) if len(all_times) > 1 else 0
            mean_p = statistics.mean(all_plddts) if all_plddts else None
            std_p = statistics.stdev(all_plddts) if len(all_plddts) > 1 else 0
            mean_sp = statistics.mean(all_speedups) if all_speedups else None
            std_sp = statistics.stdev(all_speedups) if len(all_speedups) > 1 else 0

            print(f"  Mean wall time: {mean_t:.2f} +/- {std_t:.2f}s")
            if mean_p is not None:
                print(f"  Mean pLDDT:     {mean_p:.4f} +/- {std_p:.4f}")
            if mean_sp is not None:
                print(f"  Mean speedup:   {mean_sp:.3f}x +/- {std_sp:.3f}")

    # Dump full JSON for archival
    print("\n" + "=" * 70)
    print("FULL JSON RESULTS")
    print("=" * 70)
    print(json.dumps(results_by_seed, indent=2))
