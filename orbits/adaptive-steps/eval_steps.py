"""Step-sweep evaluator for adaptive-steps orbit.

Tests ODE-10, ODE-12, ODE-15 with TF32+bf16 on eval-v3 (MSA-cached).
Based on eval-v2-winner's eval_stacked.py with MSA caching from the
standard evaluator.

Usage:
    # Sanity check
    modal run orbits/adaptive-steps/eval_steps.py --mode sanity

    # Single config
    modal run orbits/adaptive-steps/eval_steps.py --mode eval --steps 10

    # Full sweep with validation (3 runs each)
    modal run orbits/adaptive-steps/eval_steps.py --mode sweep --validate

    # Specific seed
    modal run orbits/adaptive-steps/eval_steps.py --mode eval --steps 10 --seed 123
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent
# The stacked wrapper lives in the eval-v2-winner solution
WINNER_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "solutions" / "eval-v2-winner"

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
        str(WINNER_DIR / "boltz_wrapper_stacked.py"),
        remote_path="/eval/boltz_wrapper_stacked.py",
    )
)

app = modal.App("boltz-eval-adaptive-steps", image=boltz_image)

# Persistent volume for pre-cached MSA files (eval-v3).
msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
    "gamma_0": 0.8,
    "noise_scale": 1.003,
    "enable_kernels": True,
    "bf16_trunk": False,
}

# The winning config uses ODE (gamma_0=0), recycling=0, TF32+bf16.
# We vary only sampling_steps.
SWEEP_CONFIGS = {
    "ODE-20/TF32+bf16": {
        "sampling_steps": 20,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
    },
    "ODE-15/TF32+bf16": {
        "sampling_steps": 15,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
    },
    "ODE-12/TF32+bf16": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
    },
    "ODE-10/TF32+bf16": {
        "sampling_steps": 10,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _inject_cached_msas(
    input_yaml: Path,
    msa_cache_root: Path,
    work_dir: Path,
) -> Optional[Path]:
    """Create a modified input YAML with msa: fields pointing to cached MSA files."""
    import yaml

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

    entity_msa_map: dict[str, str] = {}
    for msa_file in msa_files:
        local_path = local_msa_dir / msa_file.name
        shutil.copy2(msa_file, local_path)
        parts = msa_file.stem.split("_")
        if len(parts) >= 2:
            entity_id = parts[-1]
            entity_msa_map[entity_id] = str(local_path)

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


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction with the stacked wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_stacked.py"))
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

    # Kernel control
    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    # bf16 trunk
    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    # MSA handling: use cached MSAs when available
    if config.get("_msa_cached"):
        pass  # MSA fields already injected into the YAML
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
                f"STDOUT: {proc.stdout[-2000:] if proc.stdout else '(empty)'}\n"
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
    # Boltz names output after the input yaml stem, so try both the raw name
    # and the name with _cached stripped.
    candidates = [target_name]
    if target_name.endswith("_cached"):
        candidates.append(target_name[:-len("_cached")])

    results_dir = None
    for name in candidates:
        candidate_dir = out_dir / f"boltz_results_{name}" / "predictions" / name
        if candidate_dir.exists():
            results_dir = candidate_dir
            break
        # Also try finding any subdir under predictions
        pred_base = out_dir / f"boltz_results_{name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]
                break

    quality: dict[str, Any] = {}

    if results_dir is None:
        return {"error": f"Prediction directory not found under {out_dir} for {target_name}"}

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


def _compute_aggregates(results: dict, eval_config: dict) -> dict:
    """Compute aggregate metrics and quality gate."""
    successful = [
        r for r in results["per_complex"]
        if r["error"] is None and r["wall_time_s"] is not None
    ]

    test_cases = eval_config.get("test_cases", [])

    if len(successful) < len(test_cases):
        failed_names = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        return {
            "error": f"Not all test cases succeeded. Failed: {failed_names}",
            "num_successful": len(successful),
            "num_total": len(test_cases),
            "speedup": 0,
            "passes_quality_gate": False,
        }

    if not successful:
        return {"error": "No successful test cases"}

    total_time = sum(r["wall_time_s"] for r in successful)
    mean_time = total_time / len(successful)
    plddts_raw = [
        r["quality"]["complex_plddt"]
        for r in successful
        if "complex_plddt" in r["quality"]
    ]
    plddts = [
        p for p in plddts_raw
        if p is not None and isinstance(p, (int, float))
        and not math.isnan(p) and not math.isinf(p) and 0.0 <= p <= 1.0
    ]
    iptms = [
        r["quality"]["iptm"]
        for r in successful
        if "iptm" in r["quality"]
    ]

    agg = {
        "num_successful": len(successful),
        "num_total": len(test_cases),
        "total_wall_time_s": total_time,
        "mean_wall_time_s": mean_time,
        "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
        "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
    }

    baseline = eval_config.get("baseline")
    if baseline is not None and baseline:
        baseline_time = baseline.get("mean_wall_time_s")
        baseline_plddt = baseline.get("mean_plddt")
        if baseline_time and mean_time > 0:
            agg["speedup"] = baseline_time / mean_time
        if baseline_plddt is not None and plddts:
            mean_plddt = sum(plddts) / len(plddts)
            agg["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
            regression = (baseline_plddt - mean_plddt) * 100.0
            agg["passes_quality_gate"] = regression <= 2.0

            if baseline.get("per_complex"):
                baseline_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                per_complex_violations = {}
                for r in successful:
                    bl_case = baseline_by_name.get(r["name"])
                    if bl_case and bl_case.get("complex_plddt") is not None:
                        case_plddt = r["quality"].get("complex_plddt")
                        if case_plddt is None:
                            agg["passes_quality_gate"] = False
                            per_complex_violations[r["name"]] = "missing pLDDT"
                        else:
                            case_regression = (bl_case["complex_plddt"] - case_plddt) * 100.0
                            if case_regression > 5.0:
                                agg["passes_quality_gate"] = False
                                per_complex_violations[r["name"]] = f"-{case_regression:.1f}pp"
                if per_complex_violations:
                    agg["per_complex_regression"] = per_complex_violations

    return agg


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_volume},
)
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation for a single configuration with MSA caching."""
    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    # MSA caching
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = (
        msa_cache_root.exists()
        and any(msa_cache_root.iterdir())
        and not merged.get("force_msa_server", False)
    )

    results: dict[str, Any] = {
        "config": merged,
        "num_runs": num_runs,
        "msa_cached": use_msa_cache,
        "per_complex": [],
        "aggregate": {},
    }

    import torch
    results["env"] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import cuequivariance_torch
        results["env"]["cuequivariance_torch"] = cuequivariance_torch.__version__
    except ImportError:
        results["env"]["cuequivariance_torch"] = None

    if use_msa_cache:
        print("[eval] MSA cache detected -- using pre-cached MSAs (eval-v3)")
    else:
        print("[eval] No MSA cache -- using MSA server (adds network latency)")

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

        run_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Try to inject cached MSAs
            run_config = dict(merged)
            effective_yaml = tc_yaml
            if use_msa_cache:
                cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached_yaml is not None:
                    effective_yaml = cached_yaml
                    run_config["_msa_cached"] = True

            print(
                f"[eval] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={run_config['sampling_steps']}, recycle={run_config['recycling_steps']}, "
                f"gamma_0={run_config.get('gamma_0', 0.8)}, "
                f"TF32={run_config.get('matmul_precision') == 'high'}, "
                f"bf16_trunk={run_config.get('bf16_trunk', False)}"
                f"{' (MSA cached)' if run_config.get('_msa_cached') else ''}"
            )

            pred_result = _run_boltz_prediction(effective_yaml, work_dir, run_config)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                print(f"[eval] {tc_name} run {run_idx + 1}: "
                      f"{pred_result['wall_time_s']:.1f}s, "
                      f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")
            run_qualities.append(pred_result["quality"])

        if last_error is not None:
            entry: dict[str, Any] = {
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
            mean_plddt_runs = (sum(all_plddts) / len(all_plddts)) if all_plddts else None

            merged_quality: dict[str, Any] = {}
            if run_qualities:
                merged_quality = dict(run_qualities[-1])
                if mean_plddt_runs is not None:
                    merged_quality["complex_plddt"] = mean_plddt_runs

            entry = {
                "name": tc_name,
                "wall_time_s": median_time,
                "quality": merged_quality,
                "error": None,
                "run_times": run_times,
            }

        results["per_complex"].append(entry)

    results["aggregate"] = _compute_aggregates(results, eval_config)
    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    steps: int = 0,
    seed: int = 42,
    num_runs: int = 1,
    validate: bool = False,
    config_name: str = "",
):
    """Adaptive-steps evaluation harness.

    Modes:
        sanity  - Quick sanity check (ODE-10, 1 test case)
        eval    - Run single config by --steps N or --config-name
        sweep   - Run all step counts (20, 15, 12, 10)

    Usage:
        modal run orbits/adaptive-steps/eval_steps.py --mode sanity
        modal run orbits/adaptive-steps/eval_steps.py --mode eval --steps 10
        modal run orbits/adaptive-steps/eval_steps.py --mode eval --steps 10 --validate
        modal run orbits/adaptive-steps/eval_steps.py --mode sweep --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        # Quick sanity: ODE-10, 1 run, smallest test case
        cfg = dict(SWEEP_CONFIGS["ODE-10/TF32+bf16"])
        cfg["seed"] = seed
        print(f"[adaptive-steps] Sanity check: ODE-10/TF32+bf16, seed={seed}")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=1)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "eval":
        if config_name and config_name in SWEEP_CONFIGS:
            cfg = dict(SWEEP_CONFIGS[config_name])
        elif steps > 0:
            name = f"ODE-{steps}/TF32+bf16"
            if name in SWEEP_CONFIGS:
                cfg = dict(SWEEP_CONFIGS[name])
            else:
                # Build config for arbitrary step count
                cfg = {
                    "sampling_steps": steps,
                    "recycling_steps": 0,
                    "gamma_0": 0.0,
                    "matmul_precision": "high",
                    "bf16_trunk": True,
                    "enable_kernels": True,
                }
        else:
            cfg = dict(SWEEP_CONFIGS["ODE-20/TF32+bf16"])

        cfg["seed"] = seed
        print(f"[adaptive-steps] Evaluating: steps={cfg['sampling_steps']}, seed={seed}, "
              f"num_runs={num_runs}")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        print(f"[adaptive-steps] Running sweep of {len(SWEEP_CONFIGS)} configs "
              f"(num_runs={num_runs}, seed={seed})")

        config_names = list(SWEEP_CONFIGS.keys())
        config_jsons = []
        for name in config_names:
            cfg = dict(SWEEP_CONFIGS[name])
            cfg["seed"] = seed
            config_jsons.append(json.dumps(cfg))
        num_runs_list = [num_runs] * len(config_names)

        all_results = {}
        for name, result_json in zip(
            config_names,
            evaluate_config.map(config_jsons, num_runs_list)
        ):
            result = json.loads(result_json)
            all_results[name] = result
            print(f"\n{'='*60}")
            print(f"CONFIG: {name}")
            print(f"{'='*60}")
            _print_summary(result)

        # Print comparison table
        print(f"\n{'='*60}")
        print("STEP SWEEP COMPARISON (eval-v3, MSA cached)")
        print(f"{'='*60}")
        print(f"{'Config':<25} {'Time(s)':>8} {'pLDDT':>8} {'Delta(pp)':>10} {'Speedup':>8} {'Gate':>6}")
        print("-" * 70)
        for name in config_names:
            result = all_results[name]
            agg = result.get("aggregate", {})
            t = agg.get("mean_wall_time_s")
            p = agg.get("mean_plddt")
            d = agg.get("plddt_delta_pp")
            s = agg.get("speedup")
            g = agg.get("passes_quality_gate")
            t_str = f"{t:.1f}" if t else "ERR"
            p_str = f"{p:.4f}" if p else "ERR"
            d_str = f"{d:+.2f}" if d is not None else "N/A"
            s_str = f"{s:.2f}x" if s else "ERR"
            g_str = "PASS" if g else "FAIL"
            print(f"{name:<25} {t_str:>8} {p_str:>8} {d_str:>10} {s_str:>8} {g_str:>6}")

        # Full JSON dump
        print("\n--- FULL SWEEP RESULTS ---")
        print(json.dumps(all_results, indent=2))


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    mean_time = agg.get("mean_wall_time_s")
    mean_plddt = agg.get("mean_plddt")
    speedup = agg.get("speedup")
    plddt_delta = agg.get("plddt_delta_pp")
    passes = agg.get("passes_quality_gate")

    if mean_time is not None:
        print(f"  Mean time: {mean_time:.1f}s")
    if mean_plddt is not None:
        print(f"  Mean pLDDT: {mean_plddt:.4f}")
    if speedup is not None:
        print(f"  Speedup: {speedup:.2f}x")
    if plddt_delta is not None:
        print(f"  pLDDT delta: {plddt_delta:+.2f} pp")
    if passes is not None:
        print(f"  Quality gate: {'PASS' if passes else 'FAIL'}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:100]}")
        else:
            t = pc.get("wall_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_times", [])
            times_str = ", ".join(f"{x:.1f}s" for x in times) if times else ""
            t_str = f"{t:.1f}s" if t is not None else "N/A"
            p_str = f"{p:.4f}" if p is not None else "N/A"
            print(f"  {pc['name']}: {t_str}, pLDDT={p_str}" + (f" [{times_str}]" if times_str else ""))
