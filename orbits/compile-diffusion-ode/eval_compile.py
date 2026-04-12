"""Evaluator for compile-diffusion-ode: torch.compile on ODE diffusion score model.

Tests torch.compile applied to the score network called 12x in the ODE sampling loop.
Builds on bypass-lightning (monkey-patched Trainer.predict, ODE-12, TF32, bf16 trunk).

Usage:
    # Sanity check
    modal run orbits/compile-diffusion-ode/eval_compile.py --mode sanity

    # Single eval with compile (default mode)
    modal run orbits/compile-diffusion-ode/eval_compile.py --mode eval

    # Sweep: baseline (no compile) vs default vs reduce-overhead vs max-autotune
    modal run orbits/compile-diffusion-ode/eval_compile.py --mode sweep

    # Validation run (3 seeds)
    modal run orbits/compile-diffusion-ode/eval_compile.py --mode eval --seed 42 --validate
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
        str(ORBIT_DIR / "boltz_compile_wrapper.py"),
        remote_path="/eval/boltz_compile_wrapper.py",
    )
)

app = modal.App("boltz-eval-compile-diffusion-ode", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "recycling_steps": 0,
    "matmul_precision": "high",
    "diffusion_samples": 1,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "enable_kernels": True,
    "bf16_trunk": True,
    "cuda_warmup": True,
    "compile_score": False,
    "compile_transformer": False,
    "compile_mode": "default",
}

SWEEP_CONFIGS = {
    "bypass-only (no compile)": {
        **BASE_CONFIG,
        "compile_score": False,
        "compile_transformer": False,
    },
    "compile-score-default": {
        **BASE_CONFIG,
        "compile_score": True,
        "compile_mode": "default",
    },
    "compile-score-reduce-overhead": {
        **BASE_CONFIG,
        "compile_score": True,
        "compile_mode": "reduce-overhead",
    },
    "compile-transformer-default": {
        **BASE_CONFIG,
        "compile_transformer": True,
        "compile_mode": "default",
    },
    "compile-transformer-reduce-overhead": {
        **BASE_CONFIG,
        "compile_transformer": True,
        "compile_mode": "reduce-overhead",
    },
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
    print(f"[eval-compile] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _run_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction using the compile wrapper."""
    wrapper = str(Path("/eval/boltz_compile_wrapper.py"))
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

    # CUDA warmup
    if config.get("cuda_warmup", False):
        cmd.append("--cuda_warmup")

    # torch.compile flags
    if config.get("compile_score", False):
        cmd.append("--compile_score")
        cmd.extend(["--compile_mode", config.get("compile_mode", "default")])
    elif config.get("compile_transformer", False):
        cmd.append("--compile_transformer")
        cmd.extend(["--compile_mode", config.get("compile_mode", "default")])

    # MSA handling
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
                f"STDERR: {proc.stderr[-3000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse phase timestamps
        predict_start = None
        predict_end = None
        for line in proc.stderr.split("\n"):
            if "[PHASE] predict_start=" in line:
                predict_start = float(line.split("=")[1])
            elif "[PHASE] predict_end=" in line:
                predict_end = float(line.split("=")[1])

        if predict_start is not None and predict_end is not None:
            result["predict_only_s"] = predict_end - predict_start

        # Also capture warmup and compile timing from stderr
        for line in proc.stderr.split("\n"):
            if "Warmup complete in" in line:
                try:
                    warmup_s = float(line.split("in ")[1].split("s")[0])
                    result["warmup_s"] = warmup_s
                except (IndexError, ValueError):
                    pass

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

    predict_times = [r.get("predict_only_s") for r in successful if r.get("predict_only_s")]
    mean_predict_only = sum(predict_times) / len(predict_times) if predict_times else None

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
        "mean_predict_only_s": mean_predict_only,
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
    volumes={"/msa_cache": msa_cache},
)
def evaluate_config(
    config_json: str,
    num_runs: int = 1,
) -> str:
    """Run evaluation for a single configuration."""
    import statistics

    config = json.loads(config_json)
    merged = {**BASE_CONFIG, **config}

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())
    if use_msa_cache:
        print(f"[eval-compile] Using cached MSAs from {msa_cache_root}")
    else:
        print("[eval-compile] WARNING: No cached MSAs found, using MSA server")

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": merged,
        "num_runs": num_runs,
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

            run_config = dict(merged)
            effective_yaml = tc_yaml
            if use_msa_cache:
                cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached_yaml is not None:
                    effective_yaml = cached_yaml
                    run_config["_msa_cached"] = True

            compile_str = f"compile={run_config.get('compile_mode', 'off')}" if run_config.get("compile_score") else "no-compile"
            print(
                f"[eval-compile] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"{compile_str}, steps={run_config['sampling_steps']}, "
                f"recycle={run_config['recycling_steps']}"
                f"{' (MSA cached)' if run_config.get('_msa_cached') else ''}"
            )

            pred_result = _run_prediction(effective_yaml, work_dir, run_config)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-compile] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                if pred_result.get("predict_only_s") is not None:
                    run_predict_times.append(pred_result["predict_only_s"])
                print(f"[eval-compile] {tc_name} run {run_idx + 1}: "
                      f"{pred_result['wall_time_s']:.1f}s"
                      f" (predict_only: {pred_result.get('predict_only_s', 'N/A')}s"
                      f", warmup: {pred_result.get('warmup_s', 'N/A')}s), "
                      f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")
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
                "predict_times": run_predict_times,
            }

        results["per_complex"].append(entry)

    results["aggregate"] = _compute_aggregates(results, eval_config)
    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=600, volumes={"/msa_cache": msa_cache})
def sanity_check() -> str:
    """Verify environment."""
    import torch
    results = {
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

    msa_cache_path = Path("/msa_cache")
    if msa_cache_path.exists():
        msa_files = list(msa_cache_path.glob("*"))
        results["msa_cache_files"] = len(msa_files)
    else:
        results["msa_cache_files"] = 0

    # Check compile wrapper
    import importlib.util
    wrapper_path = "/eval/boltz_compile_wrapper.py"
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    results["compile_wrapper_found"] = bool(spec and spec.loader)

    # Check torch.compile availability
    results["torch_compile_available"] = hasattr(torch, "compile")
    results["dynamo_available"] = hasattr(torch, "_dynamo")

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    config: str = "",
    num_runs: int = 1,
    validate: bool = False,
    seed: int = 42,
    compile_mode: str = "default",
):
    """Compile-diffusion-ODE evaluation harness.

    Modes:
        sanity       - Verify environment
        eval         - Run single config (default: compile-default)
        sweep        - Compare no-compile vs all compile modes
        sweep-seeds  - Run sweep across 3 seeds for validation

    Usage:
        modal run orbits/compile-diffusion-ode/eval_compile.py --mode sanity
        modal run orbits/compile-diffusion-ode/eval_compile.py --mode eval
        modal run orbits/compile-diffusion-ode/eval_compile.py --mode eval --compile-mode reduce-overhead
        modal run orbits/compile-diffusion-ode/eval_compile.py --mode sweep
        modal run orbits/compile-diffusion-ode/eval_compile.py --mode sweep-seeds
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-compile] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "eval":
        if config:
            cfg = json.loads(config)
        else:
            cfg_name = f"compile-score-{compile_mode}"
            if cfg_name not in SWEEP_CONFIGS:
                cfg_name = list(SWEEP_CONFIGS.keys())[1]  # first compile config
            cfg = {
                **SWEEP_CONFIGS[cfg_name],
                "seed": seed,
            }

        print(f"[eval-compile] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        print(f"[eval-compile] Running sweep (num_runs={num_runs}, seed={seed})")
        all_results = {}

        # Run all configs in parallel using .map()
        config_names = list(SWEEP_CONFIGS.keys())
        config_jsons = [
            json.dumps({**SWEEP_CONFIGS[name], "seed": seed})
            for name in config_names
        ]

        results_iter = evaluate_config.map(
            config_jsons,
            [num_runs] * len(config_names),
        )

        for name, result_json in zip(config_names, results_iter):
            result = json.loads(result_json)
            all_results[name] = result
            print(f"\n{'='*60}")
            print(f"CONFIG: {name}")
            print(f"{'='*60}")
            _print_summary(result)

        _print_comparison(all_results)

    elif mode == "sweep-seeds":
        # Run sweep across 3 seeds in parallel
        seeds = [42, 123, 7]
        print(f"[eval-compile] Running sweep-seeds across seeds={seeds}")

        # Build all (config, seed) combinations
        config_names = list(SWEEP_CONFIGS.keys())
        all_config_jsons = []
        all_labels = []
        for name in config_names:
            for s in seeds:
                all_config_jsons.append(json.dumps({**SWEEP_CONFIGS[name], "seed": s}))
                all_labels.append((name, s))

        # Run all in parallel via .map()
        results_iter = evaluate_config.map(
            all_config_jsons,
            [1] * len(all_config_jsons),
        )

        # Collect results
        seed_results: dict[str, dict[int, Any]] = {name: {} for name in config_names}
        for (name, s), result_json in zip(all_labels, results_iter):
            result = json.loads(result_json)
            seed_results[name][s] = result

        # Compute mean +/- std across seeds
        print(f"\n{'='*70}")
        print("SWEEP-SEEDS SUMMARY (mean +/- std across seeds)")
        print(f"{'='*70}")
        print(f"{'Config':<35} {'Wall(s)':>8} {'Pred(s)':>8} {'pLDDT':>8} {'Speedup':>10} {'Gate':>6}")
        print("-" * 85)

        import statistics
        for name in config_names:
            times = []
            pred_times = []
            plddts = []
            speedups = []
            gate = True
            for s in seeds:
                r = seed_results[name][s]
                agg = r.get("aggregate", {})
                if agg.get("mean_wall_time_s"):
                    times.append(agg["mean_wall_time_s"])
                if agg.get("mean_predict_only_s"):
                    pred_times.append(agg["mean_predict_only_s"])
                if agg.get("mean_plddt"):
                    plddts.append(agg["mean_plddt"])
                if agg.get("speedup"):
                    speedups.append(agg["speedup"])
                if not agg.get("passes_quality_gate", False):
                    gate = False

            def fmt_mean_std(vals):
                if not vals:
                    return "N/A"
                m = statistics.mean(vals)
                if len(vals) > 1:
                    s = statistics.stdev(vals)
                    return f"{m:.1f}+/-{s:.1f}"
                return f"{m:.1f}"

            def fmt_mean_std_plddt(vals):
                if not vals:
                    return "N/A"
                m = statistics.mean(vals)
                if len(vals) > 1:
                    s = statistics.stdev(vals)
                    return f"{m:.4f}+/-{s:.4f}"
                return f"{m:.4f}"

            def fmt_mean_std_speedup(vals):
                if not vals:
                    return "N/A"
                m = statistics.mean(vals)
                if len(vals) > 1:
                    s = statistics.stdev(vals)
                    return f"{m:.2f}x+/-{s:.2f}"
                return f"{m:.2f}x"

            t_str = fmt_mean_std(times)
            p_str = fmt_mean_std(pred_times)
            plddt_str = fmt_mean_std_plddt(plddts)
            su_str = fmt_mean_std_speedup(speedups)
            g_str = "PASS" if gate else "FAIL"

            print(f"{name:<35} {t_str:>8} {p_str:>8} {plddt_str:>8} {su_str:>10} {g_str:>6}")

        # Per-seed detail
        for name in config_names:
            print(f"\n--- {name} ---")
            for s in seeds:
                r = seed_results[name][s]
                agg = r.get("aggregate", {})
                t = agg.get("mean_wall_time_s", "ERR")
                po = agg.get("mean_predict_only_s", "N/A")
                p = agg.get("mean_plddt", "ERR")
                su = agg.get("speedup", "ERR")
                print(f"  seed={s}: wall={t:.1f}s, predict={po}s, pLDDT={p}, speedup={su:.2f}x"
                      if isinstance(t, float) else f"  seed={s}: ERROR")

        # Dump full results
        print("\n--- FULL RESULTS ---")
        print(json.dumps(seed_results, indent=2))


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    for key in ["mean_wall_time_s", "mean_predict_only_s", "mean_plddt",
                "speedup", "plddt_delta_pp", "passes_quality_gate"]:
        val = agg.get(key)
        if val is not None:
            if key == "passes_quality_gate":
                print(f"  {key}: {'PASS' if val else 'FAIL'}")
            elif "plddt" in key and isinstance(val, float) and abs(val) < 1:
                print(f"  {key}: {val:.4f}")
            elif isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:200]}")
        else:
            t = pc.get("wall_time_s")
            po = pc.get("predict_only_s")
            p = pc.get("quality", {}).get("complex_plddt")
            po_str = f", predict={po:.1f}s" if po else ""
            print(f"  {pc['name']}: {t:.1f}s{po_str}, pLDDT={p:.4f}" if t and p else f"  {pc['name']}: incomplete")


def _print_comparison(all_results: dict):
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'Time(s)':>8} {'Pred(s)':>8} {'pLDDT':>8} {'Speedup':>8} {'Gate':>6}")
    print("-" * 80)
    for name, result in all_results.items():
        agg = result.get("aggregate", {})
        t = agg.get("mean_wall_time_s")
        po = agg.get("mean_predict_only_s")
        p = agg.get("mean_plddt")
        s = agg.get("speedup")
        g = agg.get("passes_quality_gate")
        t_str = f"{t:.1f}" if t else "ERR"
        po_str = f"{po:.1f}" if po else "N/A"
        p_str = f"{p:.4f}" if p else "ERR"
        s_str = f"{s:.2f}x" if s else "ERR"
        g_str = "PASS" if g else "FAIL"
        print(f"{name:<35} {t_str:>8} {po_str:>8} {p_str:>8} {s_str:>8} {g_str:>6}")
