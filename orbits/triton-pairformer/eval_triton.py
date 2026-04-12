"""Evaluator for Triton triangle multiplication kernels.

Runs end-to-end Boltz-2 predictions with Triton triangle mul replacing
cuequivariance, stacked on top of ODE + TF32 + bf16 optimizations.

Configurations tested:
- ODE-12/0r + TF32 + bf16 + Triton triangle mul (matmul variant)
- ODE-12/0r + TF32 + bf16 + Triton triangle mul (Triton kernel variant)
- ODE-12/0r + TF32 + bf16 + cuequivariance (baseline comparison)

Usage:
    modal run orbits/triton-pairformer/eval_triton.py --mode sanity
    modal run orbits/triton-pairformer/eval_triton.py --mode eval --config-name matmul
    modal run orbits/triton-pairformer/eval_triton.py --mode sweep --validate
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

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "triton>=2.2.0",
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
        str(ORBIT_DIR / "boltz_wrapper_triton.py"),
        remote_path="/eval/boltz_wrapper_triton.py",
    )
    .add_local_file(
        str(ORBIT_DIR / "triton_triangle_mul.py"),
        remote_path="/eval/triton_triangle_mul.py",
    )
)

app = modal.App("boltz-eval-triton-pairformer", image=boltz_image)

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
    "bf16_trunk": False,
    "use_triton": False,
    "use_matmul": False,
    "no_triton": True,
    "enable_kernels": True,
}

SWEEP_CONFIGS = {
    # Current best from eval-v2-winner: ODE-12 + TF32 + bf16 + cuequivariance
    "baseline-ODE12-TF32-bf16-cueq": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "no_triton": True,
        "enable_kernels": True,
    },
    # Replace cuequivariance with batched matmul (traceable)
    "ODE12-TF32-bf16-matmul": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "use_matmul": True,
        "no_triton": False,
        "enable_kernels": False,
    },
    # Replace cuequivariance with Triton kernels (traceable)
    "ODE12-TF32-bf16-triton": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "use_triton": True,
        "use_matmul": False,
        "no_triton": False,
        "enable_kernels": False,
    },
    # No kernels at all (pure PyTorch einsum), for comparison
    "ODE12-TF32-bf16-einsum": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "no_triton": True,
        "enable_kernels": False,
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


def _inject_cached_msas(input_yaml, msa_cache_root, work_dir):
    """Inject cached MSAs into test case YAML."""
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

    return cached_yaml


def _run_boltz_prediction(input_yaml, out_dir, config):
    """Run a single Boltz-2 prediction with the Triton wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_triton.py"))
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
        "--matmul_precision", config.get("matmul_precision", "highest"),
    ]

    if config.get("bf16_trunk"):
        cmd.append("--bf16_trunk")

    if config.get("use_matmul"):
        cmd.append("--use_matmul")

    if config.get("no_triton"):
        cmd.append("--no_triton")

    if config.get("enable_kernels"):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    # MSA handling
    if config.get("_msa_cached"):
        pass  # MSA fields already injected
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    result = {
        "wall_time_s": None,
        "quality": {},
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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


def _parse_confidence(out_dir, input_yaml):
    target_name = input_yaml.stem
    # Handle _cached suffix
    if target_name.endswith("_cached"):
        target_name = target_name[:-7]

    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    quality = {}

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    # Also check without _cached suffix in directory name
    if not results_dir.exists():
        for suffix in ["_cached", ""]:
            name = input_yaml.stem
            pred_base = out_dir / f"boltz_results_{name}" / "predictions"
            if pred_base.exists():
                subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
                if subdirs:
                    results_dir = subdirs[0]
                    break

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


def _compute_aggregates(results, eval_config):
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
    """Run evaluation for a single configuration."""
    import statistics

    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

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

    # MSA caching
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = (
        msa_cache_root.exists()
        and any(msa_cache_root.iterdir())
    )
    if use_msa_cache:
        print("[eval-triton] MSA cache detected")
    else:
        print("[eval-triton] No MSA cache")

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"YAML not found: {tc_yaml}",
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

            desc = (
                f"steps={run_config['sampling_steps']}, "
                f"recycle={run_config['recycling_steps']}, "
                f"triton={'matmul' if run_config.get('use_matmul') else 'triton' if not run_config.get('no_triton') else 'off'}"
            )
            print(f"[eval-triton] Running {tc_name} run {run_idx + 1}/{num_runs} ({desc})")

            pred_result = _run_boltz_prediction(effective_yaml, work_dir, run_config)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-triton] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                print(f"[eval-triton] {tc_name} run {run_idx + 1}: "
                      f"{pred_result['wall_time_s']:.1f}s, "
                      f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")
            run_qualities.append(pred_result["quality"])

        if last_error is not None:
            entry = {
                "name": tc_name,
                "wall_time_s": None,
                "quality": {},
                "error": last_error,
            }
        else:
            median_time = statistics.median(run_times) if run_times else None
            all_plddts = [q["complex_plddt"] for q in run_qualities if "complex_plddt" in q]
            mean_plddt_runs = (sum(all_plddts) / len(all_plddts)) if all_plddts else None

            merged_quality = {}
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


@app.function(gpu="L40S", timeout=600)
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
        import triton
        results["triton_version"] = triton.__version__
    except ImportError:
        results["triton_version"] = None

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

    # Test Triton kernel import
    import sys
    sys.path.insert(0, "/eval")
    try:
        from triton_triangle_mul import triton_triangle_mul_outgoing
        # Quick correctness test
        a = torch.randn(1, 32, 32, 16, device="cuda")
        b = torch.randn(1, 32, 32, 16, device="cuda")
        ref = torch.einsum("bikd,bjkd->bijd", a, b)
        out = triton_triangle_mul_outgoing(a, b)
        err = (ref - out).abs().max().item()
        results["triton_kernel_test"] = {
            "max_abs_error": err,
            "pass": err < 1e-3,
        }
    except Exception as e:
        results["triton_kernel_test"] = {"error": str(e)}

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    config: str = "",
    config_name: str = "",
    num_runs: int = 1,
    validate: bool = False,
):
    """Triton Pairformer evaluation harness.

    Modes:
        sanity  - Verify environment and Triton kernels
        eval    - Run single config
        sweep   - Run all predefined configurations

    Usage:
        modal run orbits/triton-pairformer/eval_triton.py --mode sanity
        modal run orbits/triton-pairformer/eval_triton.py --mode eval --config-name ODE12-TF32-bf16-triton
        modal run orbits/triton-pairformer/eval_triton.py --mode sweep --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-triton] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "eval":
        if config_name and config_name in SWEEP_CONFIGS:
            cfg = SWEEP_CONFIGS[config_name]
        elif config:
            cfg = json.loads(config)
        else:
            cfg = SWEEP_CONFIGS["ODE12-TF32-bf16-matmul"]

        print(f"[eval-triton] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        print(f"[eval-triton] Running sweep of {len(SWEEP_CONFIGS)} configs (num_runs={num_runs})")

        config_names = list(SWEEP_CONFIGS.keys())
        config_jsons = [json.dumps(SWEEP_CONFIGS[name]) for name in config_names]
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
        print("SWEEP COMPARISON")
        print(f"{'='*60}")
        print(f"{'Config':<35} {'Time(s)':>8} {'pLDDT':>8} {'Delta(pp)':>10} {'Speedup':>8} {'Gate':>6}")
        print("-" * 80)
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
            print(f"{name:<35} {t_str:>8} {p_str:>8} {d_str:>10} {s_str:>8} {g_str:>6}")

        print("\n--- FULL SWEEP RESULTS ---")
        print(json.dumps(all_results, indent=2))


def _print_summary(result):
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
            print(f"  {pc['name']}: ERROR - {pc['error'][:200]}")
        else:
            t = pc.get("wall_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_times", [])
            times_str = ", ".join(f"{x:.1f}s" for x in times) if times else ""
            t_str = f"{t:.1f}s" if t else "N/A"
            p_str = f"{p:.4f}" if p else "N/A"
            print(f"  {pc['name']}: {t_str}, pLDDT={p_str}" + (f" [{times_str}]" if times_str else ""))
