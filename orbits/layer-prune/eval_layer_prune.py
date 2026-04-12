"""Evaluator for layer pruning experiment.

Sweeps DiffusionTransformer layer counts (K=24,20,16,12,8) and optionally
Pairformer block counts, all on top of ODE-20/0r + TF32 + bf16 (the current
best stacked config at 1.34x).

Each configuration runs 3 seeds on Modal L40S with pre-cached MSAs (eval-v3).

Usage:
    # Sanity check
    modal run orbits/layer-prune/eval_layer_prune.py --mode sanity

    # Single K value (1 run for fast iteration)
    modal run orbits/layer-prune/eval_layer_prune.py --mode eval --diff-k 16

    # Full sweep with 3 runs each (for validation)
    modal run orbits/layer-prune/eval_layer_prune.py --mode sweep --validate

    # Pairformer sweep
    modal run orbits/layer-prune/eval_layer_prune.py --mode pf-sweep --validate
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

ORBIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORBIT_DIR.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"

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
        str(ORBIT_DIR / "boltz_wrapper_layer_prune.py"),
        remote_path="/eval/boltz_wrapper_layer_prune.py",
    )
)

app = modal.App("boltz-eval-layer-prune", image=boltz_image)

msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base config: ODE-20/0r + TF32 + bf16 (current best = 1.34x)
BASE_CONFIG: dict[str, Any] = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "matmul_precision": "high",
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "bf16_trunk": True,
    "enable_kernels": True,
    "diffusion_samples": 1,
    "seed": 42,
}

# DiffusionTransformer layer sweep
DIFF_SWEEP_CONFIGS = {
    "DT-K24 (baseline)": {"diff_transformer_k": 24},
    "DT-K20": {"diff_transformer_k": 20},
    "DT-K16": {"diff_transformer_k": 16},
    "DT-K12": {"diff_transformer_k": 12},
    "DT-K8": {"diff_transformer_k": 8},
}

# Pairformer block sweep (run separately)
# Boltz2 uses PairformerArgsV2 with 64 blocks, not 48.
# Only the trunk pairformer is pruned; confidence head is untouched.
PF_SWEEP_CONFIGS = {
    "PF-K64 (baseline)": {"pairformer_k": 64},
    "PF-K48": {"pairformer_k": 48},
    "PF-K32": {"pairformer_k": 32},
}

# Combined sweep: best DT-K with best PF-K (run after individual sweeps)
# Will be configured dynamically based on results.


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
    """Run a single Boltz-2 prediction with the layer-pruning wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_layer_prune.py"))
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
        "--matmul_precision", str(config.get("matmul_precision", "highest")),
    ]

    # Kernel control
    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    # bf16 trunk
    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    # Layer pruning
    if config.get("diff_transformer_k") is not None:
        cmd.extend(["--diff_transformer_k", str(config["diff_transformer_k"])])
    if config.get("pairformer_k") is not None:
        cmd.extend(["--pairformer_k", str(config["pairformer_k"])])

    # MSA handling
    if config.get("_msa_cached"):
        pass  # MSA fields already in YAML
    elif config.get("msa_directory"):
        cmd.extend(["--msa_directory", str(config["msa_directory"])])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

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
    # Handle _cached suffix in yaml name
    if target_name.endswith("_cached"):
        target_name = target_name[:-7]
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    quality: dict[str, Any] = {}

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if not pred_base.exists():
            # Try without stripping _cached
            orig_name = input_yaml.stem
            pred_base = out_dir / f"boltz_results_{orig_name}" / "predictions"
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
    import statistics

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

    agg: dict[str, Any] = {
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
    """Run evaluation for a single configuration."""
    import statistics

    config = json.loads(config_json)
    merged = {**BASE_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    # MSA caching
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = (
        msa_cache_root.exists()
        and any(msa_cache_root.iterdir())
        and not merged.get("force_msa_server", False)
    )
    if use_msa_cache:
        print("[eval] MSA cache detected (eval-v3)")

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

            run_config = dict(merged)
            effective_yaml = tc_yaml
            if use_msa_cache:
                cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached_yaml is not None:
                    effective_yaml = cached_yaml
                    run_config["_msa_cached"] = True

            dtk = run_config.get("diff_transformer_k", "all")
            pfk = run_config.get("pairformer_k", "all")
            print(
                f"[eval] {tc_name} run {run_idx + 1}/{num_runs} "
                f"DT-K={dtk}, PF-K={pfk}, "
                f"steps={run_config['sampling_steps']}, recycle={run_config['recycling_steps']}"
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


@app.function(gpu="L40S", timeout=600, volumes={"/msa_cache": msa_volume})
def sanity_check() -> str:
    """Verify environment and wrapper."""
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

    # Test wrapper import
    import importlib.util
    wrapper_path = "/eval/boltz_wrapper_layer_prune.py"
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    results["wrapper_found"] = bool(spec and spec.loader)

    # Verify MSA cache
    msa_root = Path("/msa_cache")
    if msa_root.exists():
        entries = list(msa_root.iterdir())
        results["msa_cache_entries"] = [e.name for e in entries]
    else:
        results["msa_cache_entries"] = []

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    diff_k: int = 0,
    pf_k: int = 0,
    num_runs: int = 1,
    validate: bool = False,
):
    """Layer pruning evaluation harness.

    Modes:
        sanity   - Verify environment
        eval     - Run single config with --diff-k and/or --pf-k
        sweep    - Sweep DiffusionTransformer K values
        pf-sweep - Sweep Pairformer K values
        full     - Run both DT and PF sweeps

    Usage:
        modal run orbits/layer-prune/eval_layer_prune.py --mode sanity
        modal run orbits/layer-prune/eval_layer_prune.py --mode eval --diff-k 16
        modal run orbits/layer-prune/eval_layer_prune.py --mode sweep --validate
        modal run orbits/layer-prune/eval_layer_prune.py --mode pf-sweep --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[layer-prune] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "eval":
        cfg: dict[str, Any] = {}
        if diff_k > 0:
            cfg["diff_transformer_k"] = diff_k
        if pf_k > 0:
            cfg["pairformer_k"] = pf_k
        print(f"[layer-prune] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        _run_sweep(DIFF_SWEEP_CONFIGS, num_runs, "DiffusionTransformer")

    elif mode == "pf-sweep":
        _run_sweep(PF_SWEEP_CONFIGS, num_runs, "Pairformer")

    elif mode == "full":
        print("[layer-prune] === DiffusionTransformer Sweep ===")
        _run_sweep(DIFF_SWEEP_CONFIGS, num_runs, "DiffusionTransformer")
        print("\n[layer-prune] === Pairformer Sweep ===")
        _run_sweep(PF_SWEEP_CONFIGS, num_runs, "Pairformer")


def _run_sweep(configs: dict[str, dict], num_runs: int, label: str):
    """Run a sweep of configurations in parallel."""
    config_names = list(configs.keys())
    config_jsons = [json.dumps(configs[name]) for name in config_names]
    num_runs_list = [num_runs] * len(config_names)

    print(f"[layer-prune] Running {label} sweep: {len(configs)} configs x {num_runs} runs")

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

    # Comparison table
    print(f"\n{'='*70}")
    print(f"{label} SWEEP COMPARISON")
    print(f"{'='*70}")
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

    print("\n--- FULL RESULTS ---")
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
            p_str = f"{p:.4f}" if p is not None else "N/A"
            t_str = f"{t:.1f}s" if t is not None else "N/A"
            print(f"  {pc['name']}: {t_str}, pLDDT={p_str}" + (f" [{times_str}]" if times_str else ""))
