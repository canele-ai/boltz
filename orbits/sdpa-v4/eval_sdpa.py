"""Evaluator for SDPA attention A/B test on eval-v4.

Clean A/B comparison:
- Control: ODE-12/0r + TF32 + bf16 (current best config)
- Test: Same + SDPA attention replacement

Uses eval-v4 infrastructure:
- Pre-cached MSAs from boltz-msa-cache-v3 (injected into YAML, not --msa_directory)
- Phase timestamps for predict_only_s (excludes model loading)
- Custom boltz_wrapper_sdpa.py with --sdpa flag

Usage:
    modal run orbits/sdpa-v4/eval_sdpa.py --mode sanity
    modal run orbits/sdpa-v4/eval_sdpa.py --mode ab --validate
"""

from __future__ import annotations

import json
import math
import shutil
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
        str(ORBIT_DIR / "boltz_wrapper_sdpa.py"),
        remote_path="/eval/boltz_wrapper_sdpa.py",
    )
)

app = modal.App("boltz-eval-sdpa-v4", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
    "gamma_0": None,
    "noise_scale": None,
    "bf16_trunk": False,
    "sdpa": False,
}

# A/B test configurations
CONTROL_CONFIG = {
    "sampling_steps": 12,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "sdpa": False,
}

SDPA_CONFIG = {
    "sampling_steps": 12,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "sdpa": True,
}

AB_CONFIGS = {
    "control (ODE-12+TF32+bf16)": CONTROL_CONFIG,
    "test (ODE-12+TF32+bf16+SDPA)": SDPA_CONFIG,
}


# ---------------------------------------------------------------------------
# Helpers (run inside Modal container)
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
    """Create a modified input YAML with msa: fields pointing to cached MSA files.

    Copied from eval-v4 evaluator.py — injects pre-computed MSA CSV paths
    into protein entries so Boltz skips the MMseqs2 server call.
    """
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

    # Copy MSA files to work_dir so Boltz can read them
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

    print(f"[eval-sdpa] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction with the SDPA wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_sdpa.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
    ]

    # MSA handling: if MSAs are injected into YAML, don't use server.
    # Otherwise fall back to --use_msa_server.
    if config.get("_msa_cached"):
        pass  # MSA fields already injected
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    # ODE / diffusion params
    if config.get("gamma_0") is not None:
        cmd.extend(["--gamma_0", str(config["gamma_0"])])
    if config.get("noise_scale") is not None:
        cmd.extend(["--noise_scale", str(config["noise_scale"])])

    # bf16 trunk
    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    # SDPA attention
    if config.get("sdpa", False):
        cmd.append("--sdpa")

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

        # Parse phase timestamps from stderr (eval-v4)
        predict_start = None
        predict_end = None
        if proc.stderr:
            for line in proc.stderr.split('\n'):
                if '[PHASE] predict_start=' in line:
                    predict_start = float(line.split('=')[1])
                elif '[PHASE] predict_end=' in line:
                    predict_end = float(line.split('=')[1])

        if predict_start is not None and predict_end is not None:
            result["predict_only_s"] = predict_end - predict_start

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
    # Handle both original and _cached YAML names
    if target_name.endswith("_cached"):
        target_name = target_name[:-7]
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    quality: dict[str, Any] = {}

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    # Also try with _cached suffix
    if not results_dir.exists():
        cached_name = input_yaml.stem
        results_dir = out_dir / f"boltz_results_{cached_name}" / "predictions" / cached_name
        if not results_dir.exists():
            pred_base = out_dir / f"boltz_results_{cached_name}" / "predictions"
            if pred_base.exists():
                subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
                if subdirs:
                    results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Prediction directory not found under {out_dir}"}

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
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_cache},
)
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation for a single configuration."""
    import statistics

    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    # MSA caching
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = (
        msa_cache_root.exists()
        and any(msa_cache_root.iterdir())
    )
    if use_msa_cache:
        print("[eval-sdpa] MSA cache detected -- using pre-cached MSAs")
    else:
        print("[eval-sdpa] No MSA cache -- using MSA server")

    results: dict[str, Any] = {
        "config": merged,
        "eval_version": "eval-v4",
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

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"Test case YAML not found: {tc_yaml}",
                "wall_time_s": None,
                "predict_only_s": None,
                "quality": {},
            })
            continue

        run_times = []
        run_predict_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Inject cached MSAs
            run_config = dict(merged)
            effective_yaml = tc_yaml
            if use_msa_cache:
                cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached_yaml is not None:
                    effective_yaml = cached_yaml
                    run_config["_msa_cached"] = True

            sdpa_label = "SDPA" if run_config.get("sdpa") else "no-SDPA"
            print(
                f"[eval-sdpa] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={run_config['sampling_steps']}, "
                f"gamma_0={run_config.get('gamma_0')}, "
                f"bf16={run_config.get('bf16_trunk')}, "
                f"{sdpa_label}"
                f"{' (MSA cached)' if run_config.get('_msa_cached') else ''}"
            )

            pred_result = _run_boltz_prediction(effective_yaml, work_dir, run_config)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-sdpa] ERROR: {last_error[:500]}")
                break

            # Use predict_only_s (eval-v4: excludes model loading)
            predict_s = pred_result.get("predict_only_s")
            wall_s = pred_result.get("wall_time_s")
            timing = predict_s if predict_s is not None else wall_s

            if timing is not None:
                run_times.append(timing)
            if predict_s is not None:
                run_predict_times.append(predict_s)

            plddt = pred_result['quality'].get('complex_plddt', 'N/A')
            print(f"[eval-sdpa] {tc_name} run {run_idx + 1}: "
                  f"predict_only={predict_s:.2f}s, wall={wall_s:.1f}s, "
                  f"pLDDT={plddt}")
            run_qualities.append(pred_result["quality"])

        if last_error is not None:
            entry: dict[str, Any] = {
                "name": tc_name,
                "wall_time_s": None,
                "predict_only_s": None,
                "quality": {},
                "error": last_error,
            }
        else:
            median_time = statistics.median(run_times) if run_times else None
            median_predict = statistics.median(run_predict_times) if run_predict_times else None
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
                "predict_only_s": median_predict,
                "quality": merged_quality,
                "error": None,
                "run_times": run_times,
                "run_predict_times": run_predict_times,
            }

        results["per_complex"].append(entry)

    # Compute aggregates
    results["aggregate"] = _compute_aggregates(results, eval_config)
    return json.dumps(results, indent=2)


def _compute_aggregates(results: dict, eval_config: dict) -> dict:
    """Compute aggregate metrics using predict_only_s (eval-v4)."""
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

    # Use predict_only_s when available, fall back to wall_time_s
    def get_time(r):
        return r.get("predict_only_s") or r.get("wall_time_s")

    total_time = sum(get_time(r) for r in successful)
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

    agg: dict[str, Any] = {
        "num_successful": len(successful),
        "num_total": len(test_cases),
        "total_predict_only_s": total_time,
        "mean_predict_only_s": mean_time,
        "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
        "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
    }

    baseline = eval_config.get("baseline")
    if baseline is not None and baseline:
        baseline_time = baseline.get("mean_wall_time_s")
        baseline_plddt = baseline.get("mean_plddt")
        if baseline_time and mean_time > 0:
            agg["speedup_vs_baseline"] = baseline_time / mean_time
        if baseline_plddt is not None and plddts:
            mean_plddt = sum(plddts) / len(plddts)
            agg["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
            regression = (baseline_plddt - mean_plddt) * 100.0
            agg["passes_quality_gate"] = regression <= 2.0

    return agg


@app.function(gpu="L40S", timeout=600, volumes={"/msa_cache": msa_cache})
def sanity_check() -> str:
    """Verify environment, wrapper, and MSA cache."""
    import torch
    results: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)

    try:
        import cuequivariance_torch
        results["cuequivariance_torch"] = cuequivariance_torch.__version__
    except ImportError:
        results["cuequivariance_torch"] = None

    try:
        import boltz
        results["boltz"] = getattr(boltz, "__version__", "unknown")
    except ImportError:
        results["boltz"] = None

    import importlib.util
    wrapper_path = "/eval/boltz_wrapper_sdpa.py"
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    results["wrapper_found"] = bool(spec and spec.loader)

    msa_path = Path("/msa_cache")
    if msa_path.exists():
        msa_files = list(msa_path.rglob("*"))
        results["msa_cache_found"] = True
        results["msa_cache_files"] = len(msa_files)
    else:
        results["msa_cache_found"] = False

    results["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "ab",
    config: str = "",
    num_runs: int = 1,
    validate: bool = False,
    seeds: str = "42,123,456",
):
    """SDPA attention A/B test on eval-v4.

    Modes:
        sanity  - Verify environment
        eval    - Run single config
        ab      - Run A/B test (control vs SDPA)

    Usage:
        modal run orbits/sdpa-v4/eval_sdpa.py --mode sanity
        modal run orbits/sdpa-v4/eval_sdpa.py --mode ab --validate
        modal run orbits/sdpa-v4/eval_sdpa.py --mode eval --config '{"sdpa": true, ...}'
    """
    if validate:
        num_runs = 3

    seed_list = [int(s.strip()) for s in seeds.split(",")]

    if mode == "sanity":
        print("[eval-sdpa] Running sanity check...")
        result_json = sanity_check.remote()
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

    elif mode == "eval":
        cfg = json.loads(config) if config else dict(SDPA_CONFIG)
        print(f"[eval-sdpa] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "ab":
        print(f"[eval-sdpa] Running A/B test with {len(seed_list)} seeds, "
              f"num_runs={num_runs}")

        all_results: dict[str, dict[str, Any]] = {}

        for seed in seed_list:
            for config_name, base_cfg in AB_CONFIGS.items():
                cfg = dict(base_cfg)
                cfg["seed"] = seed
                label = f"{config_name} seed={seed}"

                print(f"\n{'='*60}")
                print(f"[eval-sdpa] {label}")
                print(f"{'='*60}")

                result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
                result = json.loads(result_json)
                all_results[label] = result
                _print_summary(result)

        # Compute A/B summary
        _print_ab_summary(all_results)

        print("\n--- FULL A/B RESULTS ---")
        print(json.dumps(all_results, indent=2))


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    mean_time = agg.get("mean_predict_only_s")
    mean_plddt = agg.get("mean_plddt")
    speedup = agg.get("speedup_vs_baseline")
    plddt_delta = agg.get("plddt_delta_pp")
    passes = agg.get("passes_quality_gate")

    if mean_time is not None:
        print(f"  Mean predict_only_s: {mean_time:.2f}s")
    if mean_plddt is not None:
        print(f"  Mean pLDDT: {mean_plddt:.4f}")
    if speedup is not None:
        print(f"  Speedup vs baseline: {speedup:.2f}x")
    if plddt_delta is not None:
        print(f"  pLDDT delta: {plddt_delta:+.2f} pp")
    if passes is not None:
        print(f"  Quality gate: {'PASS' if passes else 'FAIL'}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:100]}")
        else:
            t = pc.get("predict_only_s") or pc.get("wall_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_predict_times") or pc.get("run_times", [])
            times_str = ", ".join(f"{x:.2f}s" for x in times) if times else ""
            t_str = f"{t:.2f}s" if t is not None else "N/A"
            p_str = f"{p:.4f}" if p is not None else "N/A"
            extra = f" [{times_str}]" if times_str else ""
            print(f"  {pc['name']}: {t_str}, pLDDT={p_str}{extra}")


def _print_ab_summary(all_results: dict):
    print(f"\n{'='*60}")
    print("A/B COMPARISON (predict_only_s)")
    print(f"{'='*60}")
    print(f"{'Label':<40} {'Time(s)':>8} {'pLDDT':>8}")
    print("-" * 60)

    control_times = []
    sdpa_times = []
    control_plddts = []
    sdpa_plddts = []

    for label, result in all_results.items():
        agg = result.get("aggregate", {})
        t = agg.get("mean_predict_only_s")
        p = agg.get("mean_plddt")
        t_str = f"{t:.2f}" if t else "ERR"
        p_str = f"{p:.4f}" if p else "ERR"
        print(f"{label:<40} {t_str:>8} {p_str:>8}")

        if t is not None:
            if "control" in label:
                control_times.append(t)
                if p is not None:
                    control_plddts.append(p)
            else:
                sdpa_times.append(t)
                if p is not None:
                    sdpa_plddts.append(p)

    if control_times and sdpa_times:
        mean_control = sum(control_times) / len(control_times)
        mean_sdpa = sum(sdpa_times) / len(sdpa_times)
        sdpa_vs_control = mean_control / mean_sdpa
        delta_pct = ((mean_control - mean_sdpa) / mean_control) * 100

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Control mean predict_only_s:  {mean_control:.2f}s ({len(control_times)} seeds)")
        print(f"SDPA mean predict_only_s:     {mean_sdpa:.2f}s ({len(sdpa_times)} seeds)")
        print(f"SDPA vs control:              {sdpa_vs_control:.3f}x")
        print(f"Time delta:                   {delta_pct:+.1f}%")

        if control_plddts and sdpa_plddts:
            mean_ctrl_plddt = sum(control_plddts) / len(control_plddts)
            mean_sdpa_plddt = sum(sdpa_plddts) / len(sdpa_plddts)
            plddt_delta = (mean_sdpa_plddt - mean_ctrl_plddt) * 100
            print(f"Control mean pLDDT:           {mean_ctrl_plddt:.4f}")
            print(f"SDPA mean pLDDT:              {mean_sdpa_plddt:.4f}")
            print(f"pLDDT delta:                  {plddt_delta:+.2f} pp")
