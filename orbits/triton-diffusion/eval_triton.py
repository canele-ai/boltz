"""Evaluator for Triton fused attention + ODE + TF32 + bf16 trunk.

Runs the full eval-v3 test suite with a custom Triton kernel that fuses
Q@K^T + scaling + pair_bias + masking + softmax + @V in a single tiled
kernel, eliminating the full S*S attention matrix materialization.

Stacks on top of eval-v2-winner optimizations:
- ODE sampling (gamma_0=0), 20 steps, recycling=0
- TF32 matmul precision
- bf16 trunk (no .float() upcast)

Usage:
    # Sanity check
    modal run orbits/triton-diffusion/eval_triton.py --mode sanity

    # Single eval (1 run)
    modal run orbits/triton-diffusion/eval_triton.py --mode eval

    # Validated eval (3 runs, median timing)
    modal run orbits/triton-diffusion/eval_triton.py --mode eval --validate

    # Comparison: with and without Triton
    modal run orbits/triton-diffusion/eval_triton.py --mode compare --validate
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
    .pip_install(
        "triton>=2.2.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_triton.py"),
        remote_path="/eval/boltz_wrapper_triton.py",
    )
    .add_local_file(
        str(ORBIT_DIR / "triton_attention.py"),
        remote_path="/eval/triton_attention.py",
    )
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_sdpa.py"),
        remote_path="/eval/boltz_wrapper_sdpa.py",
    )
)

app = modal.App("boltz-eval-triton-diffusion", image=boltz_image)

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
    "triton_attention": False,
}

# The target configuration: Triton kernel + all optimizations stacked
TRITON_CONFIG = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "triton_attention": True,
}

# SDPA (PyTorch built-in flash attention) + all optimizations
SDPA_CONFIG = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "sdpa_attention": True,
}

# Comparison: same as winner but without Triton/SDPA
BASELINE_FAST_CONFIG = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "triton_attention": False,
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
    """Run a single Boltz-2 prediction with the appropriate wrapper."""
    # Choose wrapper based on config
    if config.get("sdpa_attention", False):
        wrapper = str(Path("/eval/boltz_wrapper_sdpa.py"))
    else:
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
    ]

    # Triton attention (only for triton wrapper)
    if config.get("triton_attention", False):
        cmd.append("--triton_attention")

    # SDPA attention (only for sdpa wrapper)
    if config.get("sdpa_attention", False):
        cmd.append("--sdpa_attention")

    # Kernel control
    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    # bf16 trunk
    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    # MSA handling
    if config.get("_msa_cached"):
        pass
    elif config.get("force_msa_server"):
        cmd.append("--use_msa_server")
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    # Set PYTHONPATH so triton_attention can be imported
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "/eval:" + env.get("PYTHONPATH", "")

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
            env=env,
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
    # Handle _cached suffix: Boltz uses the yaml stem as the output dir name
    # Try both with and without _cached suffix
    names_to_try = [target_name]
    if target_name.endswith("_cached"):
        names_to_try.append(target_name[:-7])
    else:
        names_to_try.append(target_name + "_cached")

    quality: dict[str, Any] = {}
    results_dir = None

    for name in names_to_try:
        candidate = out_dir / f"boltz_results_{name}" / "predictions" / name
        if candidate.exists():
            results_dir = candidate
            break
        # Also try any subdirectory under predictions
        pred_base = out_dir / f"boltz_results_{name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]
                break

    if results_dir is None:
        # Last resort: scan all boltz_results_* dirs
        for d in out_dir.glob("boltz_results_*"):
            pred_base = d / "predictions"
            if pred_base.exists():
                subdirs = [sd for sd in pred_base.iterdir() if sd.is_dir()]
                if subdirs:
                    results_dir = subdirs[0]
                    break

    if results_dir is None:
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

    # MSA caching
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = (
        msa_cache_root.exists()
        and any(msa_cache_root.iterdir())
        and not merged.get("force_msa_server", False)
    )
    if use_msa_cache:
        print("[eval] MSA cache detected -- using pre-cached MSAs (eval-v3)")

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
        import triton
        results["env"]["triton_version"] = triton.__version__
    except ImportError:
        results["env"]["triton_version"] = None

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

            print(
                f"[eval-triton] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={run_config['sampling_steps']}, recycle={run_config['recycling_steps']}, "
                f"gamma_0={run_config.get('gamma_0', 0.8)}, TF32={run_config.get('matmul_precision') == 'high'}, "
                f"bf16_trunk={run_config.get('bf16_trunk', False)}, "
                f"triton_attn={run_config.get('triton_attention', False)}"
                f"{' (MSA cached)' if run_config.get('_msa_cached') else ''}"
            )

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
    """Verify environment: torch, triton, boltz, cuequivariance, wrapper."""
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

    # Test Triton kernel correctness
    if results.get("triton_version") and results["cuda_available"]:
        import sys
        sys.path.insert(0, "/eval")
        try:
            from triton_attention import triton_attention_pair_bias, reference_attention_pair_bias

            B, S, H, D = 1, 64, 8, 48
            torch.manual_seed(42)
            q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
            k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
            v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
            bias = torch.randn(B, H, S, S, device="cuda", dtype=torch.float32) * 0.1
            mask = torch.ones(B, S, device="cuda", dtype=torch.float32)

            ref = reference_attention_pair_bias(q, k, v, bias, mask)
            tri = triton_attention_pair_bias(q, k, v, bias, mask)

            max_diff = (ref - tri).abs().max().item()
            results["triton_kernel_max_diff"] = max_diff
            results["triton_kernel_correct"] = max_diff < 1e-2
        except Exception as e:
            results["triton_kernel_error"] = str(e)

    # Check MSA cache
    msa_cache_root = Path("/msa_cache")
    results["msa_cache_exists"] = msa_cache_root.exists()
    if msa_cache_root.exists():
        results["msa_cache_entries"] = len(list(msa_cache_root.iterdir()))

    # Check wrapper
    import importlib.util
    wrapper_path = "/eval/boltz_wrapper_triton.py"
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    results["wrapper_found"] = spec is not None and spec.loader is not None

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
):
    """Triton fused attention evaluation harness.

    Modes:
        sanity  - Verify environment and kernel correctness
        eval    - Run triton+ODE+TF32+bf16 config
        compare - Run both with and without Triton for comparison

    Usage:
        modal run orbits/triton-diffusion/eval_triton.py --mode sanity
        modal run orbits/triton-diffusion/eval_triton.py --mode eval
        modal run orbits/triton-diffusion/eval_triton.py --mode eval --validate
        modal run orbits/triton-diffusion/eval_triton.py --mode compare --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-triton] Running sanity check...")
        result_json = sanity_check.remote()
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))

        if result.get("triton_kernel_correct"):
            print("\nTriton kernel correctness: PASS")
        else:
            print("\nTriton kernel correctness: FAIL or NOT TESTED")

    elif mode == "eval":
        if config:
            cfg = json.loads(config)
        else:
            cfg = TRITON_CONFIG

        print(f"[eval-triton] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary("Triton+ODE+TF32+bf16", result)

    elif mode == "sdpa":
        cfg = SDPA_CONFIG
        print(f"[eval-triton] Evaluating SDPA: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary("SDPA+ODE+TF32+bf16", result)

    elif mode == "compare":
        print(f"[eval-triton] Comparing configurations (num_runs={num_runs})")

        # Launch all in parallel
        configs = {
            "ODE-20+TF32+bf16 (baseline)": BASELINE_FAST_CONFIG,
            "ODE-20+TF32+bf16+SDPA": SDPA_CONFIG,
        }

        config_names = list(configs.keys())
        config_jsons = [json.dumps(configs[name]) for name in config_names]
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
            _print_summary(name, result)

        # Comparison table
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Config':<35} {'Time(s)':>8} {'pLDDT':>8} {'Delta(pp)':>10} {'Speedup':>8} {'Gate':>6}")
        print("-" * 80)
        for name in config_names:
            agg = all_results[name].get("aggregate", {})
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

        print("\n--- FULL RESULTS ---")
        print(json.dumps(all_results, indent=2))


def _print_summary(name: str, result: dict):
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
