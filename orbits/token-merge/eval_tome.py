"""Evaluator for Token Merging + stacked optimizations.

Runs ToMe-patched Boltz-2 on the eval-v3 test set. Builds on the
ODE+TF32+bf16 stacked baseline and adds token merging at various
merge ratios.

Usage:
    # Sanity check
    modal run orbits/token-merge/eval_tome.py --mode sanity

    # Single config (conservative 10% merge)
    modal run orbits/token-merge/eval_tome.py --mode eval \
        --config '{"tome_ratio": 0.1}'

    # Sweep merge ratios
    modal run orbits/token-merge/eval_tome.py --mode sweep
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
        str(ORBIT_DIR / "boltz_wrapper_tome.py"),
        remote_path="/eval/boltz_wrapper_tome.py",
    )
    .add_local_file(
        str(ORBIT_DIR / "token_merge.py"),
        remote_path="/eval/token_merge.py",
    )
)

app = modal.App("boltz-eval-tome", image=boltz_image)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base config: ODE-12 + recycling=0 + TF32 + bf16 (current best: 1.47x)
BASE_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "recycling_steps": 0,
    "matmul_precision": "high",
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "bf16_trunk": True,
    "enable_kernels": True,
    "diffusion_samples": 1,
    "seed": 42,
    "tome_ratio": 0.0,
    "tome_merge_after_layer": 0,
}

# Sweep: test various merge configurations
# Key insight from first experiment: merging before all layers (-17pp quality drop)
# is too aggressive. Partial-layer merging (merge_after_layer > 0) should help.
SWEEP_CONFIGS = {
    # Baseline: no merging
    "base-no-tome": {
        **BASE_CONFIG,
        "tome_ratio": 0.0,
    },
    # Conservative: merge 20% after layer 32 (half layers at full res)
    "r20-after32": {
        **BASE_CONFIG,
        "tome_ratio": 0.2,
        "tome_merge_after_layer": 32,
    },
    # Moderate: merge 30% after layer 32
    "r30-after32": {
        **BASE_CONFIG,
        "tome_ratio": 0.3,
        "tome_merge_after_layer": 32,
    },
    # Aggressive: merge 50% after layer 32
    "r50-after32": {
        **BASE_CONFIG,
        "tome_ratio": 0.5,
        "tome_merge_after_layer": 32,
    },
    # Very late merge: merge 30% after layer 48 (only 16 layers merged)
    "r30-after48": {
        **BASE_CONFIG,
        "tome_ratio": 0.3,
        "tome_merge_after_layer": 48,
    },
    # Early + conservative: merge 20% after layer 16
    "r20-after16": {
        **BASE_CONFIG,
        "tome_ratio": 0.2,
        "tome_merge_after_layer": 16,
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


def _run_boltz_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a single Boltz-2 prediction with ToMe wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_tome.py"))
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

    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    tome_ratio = config.get("tome_ratio", 0.0)
    if tome_ratio > 0:
        cmd.extend(["--tome_ratio", str(tome_ratio)])
        merge_after = config.get("tome_merge_after_layer", 0)
        cmd.extend(["--tome_merge_after_layer", str(merge_after)])

    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
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
    plddts = [
        r["quality"]["complex_plddt"]
        for r in successful
        if "complex_plddt" in r["quality"]
        and r["quality"]["complex_plddt"] is not None
        and isinstance(r["quality"]["complex_plddt"], (int, float))
        and 0.0 <= r["quality"]["complex_plddt"] <= 1.0
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
    if baseline:
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
                violations = {}
                for r in successful:
                    bl = baseline_by_name.get(r["name"])
                    if bl and bl.get("complex_plddt") is not None:
                        case_plddt = r["quality"].get("complex_plddt")
                        if case_plddt is None:
                            agg["passes_quality_gate"] = False
                            violations[r["name"]] = "missing pLDDT"
                        else:
                            case_reg = (bl["complex_plddt"] - case_plddt) * 100.0
                            if case_reg > 5.0:
                                agg["passes_quality_gate"] = False
                                violations[r["name"]] = f"-{case_reg:.1f}pp"
                if violations:
                    agg["per_complex_regression"] = violations

    return agg


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=7200)
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation for a single configuration."""
    import statistics
    import torch

    config = json.loads(config_json)
    merged_cfg = {**BASE_CONFIG, **config}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": merged_cfg,
        "num_runs": num_runs,
        "per_complex": [],
        "aggregate": {},
    }

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

            print(
                f"[eval-tome] Running {tc_name} run {run_idx + 1}/{num_runs} "
                f"steps={merged_cfg['sampling_steps']}, recycle={merged_cfg['recycling_steps']}, "
                f"tome_ratio={merged_cfg.get('tome_ratio', 0.0)}"
            )

            pred_result = _run_boltz_prediction(tc_yaml, work_dir, merged_cfg)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-tome] ERROR: {last_error[:500]}")
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
                print(f"[eval-tome] {tc_name} run {run_idx + 1}: "
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


@app.function(gpu="L40S", timeout=600)
def sanity_check() -> str:
    """Verify environment and token_merge module."""
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

    # Test token_merge module loads
    import importlib.util
    tome_path = "/eval/token_merge.py"
    spec = importlib.util.spec_from_file_location("token_merge", tome_path)
    if spec and spec.loader:
        tome_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tome_mod)
        results["token_merge_loaded"] = True
        results["token_merge_functions"] = [
            x for x in dir(tome_mod) if not x.startswith("_")
        ]
    else:
        results["token_merge_loaded"] = False

    # Quick numerical test: create dummy tensors and run merge/unmerge
    if torch.cuda.is_available():
        try:
            B, N, D_s, D_z = 1, 20, 64, 32
            s = torch.randn(B, N, D_s, device="cuda")
            z = torch.randn(B, N, N, D_z, device="cuda")
            mask = torch.ones(B, N, device="cuda")
            pair_mask = mask[:, :, None] * mask[:, None, :]

            def identity_fn(s_, z_, m_, pm_, uk_):
                return s_, z_

            s_out, z_out = tome_mod.tome_merge_and_run(
                s, z, mask, pair_mask, 0.2, identity_fn
            )
            results["numerical_test"] = {
                "input_shape": [B, N, D_s],
                "output_shape": list(s_out.shape),
                "z_input_shape": [B, N, N, D_z],
                "z_output_shape": list(z_out.shape),
                "shapes_match": list(s_out.shape) == [B, N, D_s] and list(z_out.shape) == [B, N, N, D_z],
            }
        except Exception as e:
            results["numerical_test"] = {"error": str(e)}

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
    """Token Merging evaluation harness.

    Modes:
        sanity  - Verify environment
        eval    - Run single config
        sweep   - Run all merge ratio configurations
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-tome] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "eval":
        if config_name and config_name in SWEEP_CONFIGS:
            cfg = SWEEP_CONFIGS[config_name]
        elif config:
            cfg = json.loads(config)
        else:
            cfg = SWEEP_CONFIGS["tome-10"]

        print(f"[eval-tome] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        print(f"[eval-tome] Running sweep of {len(SWEEP_CONFIGS)} configs (num_runs={num_runs})")

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
        print(f"{'Config':<20} {'Time(s)':>8} {'pLDDT':>8} {'Delta(pp)':>10} {'Speedup':>8} {'Gate':>6}")
        print("-" * 65)
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
            print(f"{name:<20} {t_str:>8} {p_str:>8} {d_str:>10} {s_str:>8} {g_str:>6}")

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
            t_str = f"{t:.1f}s" if t else "N/A"
            p_str = f"{p:.4f}" if p else "N/A"
            print(f"  {pc['name']}: {t_str}, pLDDT={p_str}" + (f" [{times_str}]" if times_str else ""))
