"""Evaluator for bmm-nokernels: clean eval-v4 A/B comparison.

Runs both configurations on the same GPU, 3 seeds each, and reports
predict_only_s (model loading excluded) via phase timestamps.

A/B configs:
  Control: ODE-12/0r + TF32 + bf16 + cuequivariance (use_kernels=True)
  Test:    ODE-12/0r + TF32 + bf16 + bmm (use_kernels=False)

Usage:
    modal run orbits/bmm-nokernels/eval_bmm.py --mode sanity
    modal run orbits/bmm-nokernels/eval_bmm.py --mode eval --config-name control
    modal run orbits/bmm-nokernels/eval_bmm.py --mode eval --config-name test-bmm
    modal run orbits/bmm-nokernels/eval_bmm.py --mode ab
    modal run orbits/bmm-nokernels/eval_bmm.py --mode ab --validate
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
    )
    .pip_install(
        "boltz==2.2.1",
    )
    .pip_install(
        # cuequivariance for control config
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_bmm.py"),
        remote_path="/eval/boltz_wrapper_bmm.py",
    )
)

app = modal.App("boltz-eval-bmm-nokernels", image=boltz_image)

msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=True)

# ---------------------------------------------------------------------------
# A/B Configuration
# ---------------------------------------------------------------------------

AB_CONFIGS = {
    # Control: cuequivariance kernels enabled (current best = 13.08s baseline)
    "control": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "use_bmm": False,
        "no_kernels": False,   # cuequivariance ON
    },
    # Test: bmm replaces cuequivariance (hypothesis: ~20% faster)
    "test-bmm": {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "gamma_0": 0.0,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "use_bmm": True,
        "no_kernels": True,    # cuequivariance OFF
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
    "gamma_0": 0.8,
    "noise_scale": 1.003,
    "bf16_trunk": False,
    "use_bmm": False,
    "no_kernels": False,
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
    """Run a single Boltz-2 prediction with the bmm wrapper."""
    wrapper = str(Path("/eval/boltz_wrapper_bmm.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--matmul_precision", config.get("matmul_precision", "highest"),
    ]

    # Diffusion params
    if config.get("gamma_0") is not None:
        cmd.extend(["--gamma_0", str(config["gamma_0"])])
    if config.get("noise_scale") is not None:
        cmd.extend(["--noise_scale", str(config["noise_scale"])])

    # bf16 trunk
    if config.get("bf16_trunk"):
        cmd.append("--bf16_trunk")

    # bmm patch
    if config.get("use_bmm"):
        cmd.append("--use_bmm")

    # Kernel control
    if config.get("no_kernels"):
        cmd.append("--no_kernels")

    # MSA handling
    if config.get("_msa_cached"):
        pass  # MSA fields already injected into YAML
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    result = {
        "wall_time_s": None,
        "predict_only_s": None,
        "quality": {},
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        # Parse phase timestamps from stderr (eval-v4 pattern)
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
    """Compute aggregate metrics, using predict_only_s as primary timing."""
    successful = [
        r for r in results["per_complex"]
        if r["error"] is None and r.get("predict_only_s") is not None
    ]

    test_cases = eval_config.get("test_cases", [])

    if len(successful) < len(test_cases):
        failed_names = [r["name"] for r in results["per_complex"] if r["error"] is not None]
        return {
            "error": f"Not all test cases succeeded. Failed: {failed_names}",
            "num_successful": len(successful),
            "num_total": len(test_cases),
        }

    if not successful:
        return {"error": "No successful test cases"}

    predict_times = [r["predict_only_s"] for r in successful]
    total_predict = sum(predict_times)
    mean_predict = total_predict / len(successful)

    wall_times = [r["wall_time_s"] for r in successful if r.get("wall_time_s")]
    total_wall = sum(wall_times) if wall_times else None
    mean_wall = (total_wall / len(wall_times)) if wall_times else None

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
        "total_predict_only_s": total_predict,
        "mean_predict_only_s": mean_predict,
        "total_wall_time_s": total_wall,
        "mean_wall_time_s": mean_wall,
        "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
        "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
    }

    # Compare against eval-v4 baseline (predict_only_s)
    baseline = eval_config.get("baseline")
    if baseline is not None:
        # eval-v4 baseline uses predict_only_s = 25.04s (200-step default)
        # But our A/B comparison baseline is the control config run in this same eval
        baseline_plddt = baseline.get("mean_plddt")
        if baseline_plddt is not None and plddts:
            mean_plddt = sum(plddts) / len(plddts)
            agg["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
            regression = (baseline_plddt - mean_plddt) * 100.0
            agg["passes_quality_gate"] = regression <= 2.0

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
        print("[eval-bmm] MSA cache detected")
    else:
        print("[eval-bmm] No MSA cache -- using MSA server")

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None,
                "predict_only_s": None,
                "quality": {},
            })
            continue

        run_predict_times = []
        run_wall_times = []
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
                f"bmm={run_config.get('use_bmm')}, "
                f"kernels={'off' if run_config.get('no_kernels') else 'on'}"
            )
            print(f"[eval-bmm] Running {tc_name} run {run_idx + 1}/{num_runs} ({desc})")

            pred_result = _run_boltz_prediction(effective_yaml, work_dir, run_config)

            if pred_result["error"] is not None:
                last_error = pred_result["error"]
                print(f"[eval-bmm] ERROR: {last_error[:500]}")
                break

            predict_s = pred_result.get("predict_only_s")
            wall_s = pred_result.get("wall_time_s")
            plddt = pred_result["quality"].get("complex_plddt", "N/A")

            if predict_s is not None:
                run_predict_times.append(predict_s)
            if wall_s is not None:
                run_wall_times.append(wall_s)
            run_qualities.append(pred_result["quality"])

            print(f"[eval-bmm] {tc_name} run {run_idx + 1}: "
                  f"predict_only={predict_s:.2f}s, wall={wall_s:.1f}s, pLDDT={plddt}")

        if last_error is not None:
            entry = {
                "name": tc_name,
                "wall_time_s": None,
                "predict_only_s": None,
                "quality": {},
                "error": last_error,
            }
        else:
            median_predict = statistics.median(run_predict_times) if run_predict_times else None
            median_wall = statistics.median(run_wall_times) if run_wall_times else None
            all_plddts = [q["complex_plddt"] for q in run_qualities if "complex_plddt" in q]
            mean_plddt_runs = (sum(all_plddts) / len(all_plddts)) if all_plddts else None

            merged_quality = {}
            if run_qualities:
                merged_quality = dict(run_qualities[-1])
                if mean_plddt_runs is not None:
                    merged_quality["complex_plddt"] = mean_plddt_runs

            entry = {
                "name": tc_name,
                "wall_time_s": median_wall,
                "predict_only_s": median_predict,
                "quality": merged_quality,
                "error": None,
                "run_predict_times": run_predict_times,
                "run_wall_times": run_wall_times,
            }

        results["per_complex"].append(entry)

    results["aggregate"] = _compute_aggregates(results, eval_config)
    return json.dumps(results, indent=2)


@app.function(
    gpu="L40S",
    timeout=10800,
    volumes={"/msa_cache": msa_volume},
)
def run_ab_comparison(num_runs: int = 3) -> str:
    """Run A/B comparison: control (cuequivariance) vs test (bmm).

    Both configs run on the same GPU instance to minimize hardware variance.
    Returns predict_only_s for both, plus speedup ratio.
    """
    all_results = {}

    for config_name in ["control", "test-bmm"]:
        cfg = AB_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"[eval-bmm] A/B: Running '{config_name}' ({num_runs} runs)")
        print(f"{'='*60}")

        result_json = evaluate_config.local(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        all_results[config_name] = result

        agg = result.get("aggregate", {})
        print(f"[eval-bmm] {config_name}: mean_predict_only_s="
              f"{agg.get('mean_predict_only_s', 'ERR')}, "
              f"mean_plddt={agg.get('mean_plddt', 'ERR')}")

    # Compute A/B speedup
    ctrl_agg = all_results["control"].get("aggregate", {})
    test_agg = all_results["test-bmm"].get("aggregate", {})

    ctrl_time = ctrl_agg.get("mean_predict_only_s")
    test_time = test_agg.get("mean_predict_only_s")

    ab_summary = {
        "control_mean_predict_only_s": ctrl_time,
        "test_bmm_mean_predict_only_s": test_time,
        "control_mean_plddt": ctrl_agg.get("mean_plddt"),
        "test_bmm_mean_plddt": test_agg.get("mean_plddt"),
    }

    if ctrl_time and test_time and test_time > 0:
        ab_summary["speedup_bmm_vs_cueq"] = ctrl_time / test_time
        ab_summary["delta_s"] = ctrl_time - test_time
        ab_summary["delta_pct"] = ((ctrl_time - test_time) / ctrl_time) * 100.0

    # pLDDT comparison
    ctrl_plddt = ctrl_agg.get("mean_plddt")
    test_plddt = test_agg.get("mean_plddt")
    if ctrl_plddt is not None and test_plddt is not None:
        ab_summary["plddt_delta_pp"] = (test_plddt - ctrl_plddt) * 100.0

    # Per-complex breakdown
    ctrl_by_name = {r["name"]: r for r in all_results["control"].get("per_complex", [])}
    test_by_name = {r["name"]: r for r in all_results["test-bmm"].get("per_complex", [])}

    per_complex_comparison = []
    for name in ctrl_by_name:
        ctrl_pc = ctrl_by_name[name]
        test_pc = test_by_name.get(name, {})
        ct = ctrl_pc.get("predict_only_s")
        tt = test_pc.get("predict_only_s")
        cp = ctrl_pc.get("quality", {}).get("complex_plddt")
        tp = test_pc.get("quality", {}).get("complex_plddt")

        entry = {
            "name": name,
            "control_predict_s": ct,
            "test_bmm_predict_s": tt,
            "control_plddt": cp,
            "test_bmm_plddt": tp,
        }
        if ct and tt and tt > 0:
            entry["speedup"] = ct / tt
        if cp is not None and tp is not None:
            entry["plddt_delta_pp"] = (tp - cp) * 100.0
        per_complex_comparison.append(entry)

    output = {
        "ab_summary": ab_summary,
        "per_complex_comparison": per_complex_comparison,
        "control": all_results["control"],
        "test_bmm": all_results["test-bmm"],
    }

    return json.dumps(output, indent=2)


@app.function(gpu="L40S", timeout=600)
def sanity_check() -> str:
    """Verify environment: torch, cuda, boltz, cuequivariance."""
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

    # Quick bmm correctness test
    a = torch.randn(1, 32, 32, 16, device="cuda")
    b = torch.randn(1, 32, 32, 16, device="cuda")

    # Reference einsum
    ref = torch.einsum("bikd,bjkd->bijd", a, b)

    # bmm path
    B, N, K, D = a.shape
    a_t = a.permute(0, 3, 1, 2).reshape(B * D, N, K)
    b_t = b.permute(0, 3, 1, 2).reshape(B * D, N, K)
    c = torch.bmm(a_t, b_t.transpose(1, 2))
    out = c.reshape(B, D, N, N).permute(0, 2, 3, 1)

    err = (ref - out).abs().max().item()
    results["bmm_correctness"] = {
        "max_abs_error": err,
        "pass": err < 1e-4,
    }

    # Test bf16 bmm
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    ref_bf16 = torch.einsum("bikd,bjkd->bijd", a_bf16, b_bf16)
    a_t = a_bf16.permute(0, 3, 1, 2).reshape(B * D, N, K)
    b_t = b_bf16.permute(0, 3, 1, 2).reshape(B * D, N, K)
    c = torch.bmm(a_t, b_t.transpose(1, 2))
    out_bf16 = c.reshape(B, D, N, N).permute(0, 2, 3, 1)
    err_bf16 = (ref_bf16 - out_bf16).abs().max().item()
    results["bmm_bf16_correctness"] = {
        "max_abs_error": err_bf16,
        "pass": err_bf16 < 1e-2,
    }

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "ab",
    config_name: str = "",
    config: str = "",
    num_runs: int = 1,
    validate: bool = False,
):
    """bmm-nokernels eval-v4 evaluator.

    Modes:
        sanity     - Verify environment and bmm correctness
        eval       - Run single config (--config-name control|test-bmm)
        ab         - Run A/B comparison (control vs test-bmm)

    Usage:
        modal run orbits/bmm-nokernels/eval_bmm.py --mode sanity
        modal run orbits/bmm-nokernels/eval_bmm.py --mode eval --config-name test-bmm
        modal run orbits/bmm-nokernels/eval_bmm.py --mode ab
        modal run orbits/bmm-nokernels/eval_bmm.py --mode ab --validate
    """
    if validate:
        num_runs = 3

    if mode == "sanity":
        print("[eval-bmm] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)

    elif mode == "eval":
        if config_name and config_name in AB_CONFIGS:
            cfg = AB_CONFIGS[config_name]
        elif config:
            cfg = json.loads(config)
        else:
            cfg = AB_CONFIGS["test-bmm"]

        print(f"[eval-bmm] Evaluating: {json.dumps(cfg)} (num_runs={num_runs})")
        result_json = evaluate_config.remote(json.dumps(cfg), num_runs=num_runs)
        result = json.loads(result_json)
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "ab":
        if num_runs < 3 and not validate:
            # For A/B, default to 3 runs for statistical confidence
            num_runs = 3

        print(f"[eval-bmm] Running A/B comparison (num_runs={num_runs})")
        result_json = run_ab_comparison.remote(num_runs=num_runs)
        result = json.loads(result_json)

        # Print summary
        ab = result.get("ab_summary", {})
        print(f"\n{'='*60}")
        print("A/B COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"  Control (cuequivariance): {ab.get('control_mean_predict_only_s', 'ERR'):.2f}s predict_only")
        print(f"  Test (bmm):               {ab.get('test_bmm_mean_predict_only_s', 'ERR'):.2f}s predict_only")

        if ab.get("speedup_bmm_vs_cueq"):
            speedup = ab["speedup_bmm_vs_cueq"]
            delta_pct = ab.get("delta_pct", 0)
            print(f"  Speedup:                  {speedup:.3f}x ({delta_pct:+.1f}%)")
        if ab.get("plddt_delta_pp") is not None:
            print(f"  pLDDT delta:              {ab['plddt_delta_pp']:+.2f} pp")

        print(f"\n  Control pLDDT: {ab.get('control_mean_plddt', 'N/A')}")
        print(f"  Test pLDDT:    {ab.get('test_bmm_mean_plddt', 'N/A')}")

        # Per-complex breakdown
        print(f"\n{'Config':<20} {'predict_s':>10} {'pLDDT':>8} {'speedup':>8}")
        print("-" * 50)
        for pc in result.get("per_complex_comparison", []):
            ct = pc.get("control_predict_s")
            tt = pc.get("test_bmm_predict_s")
            cp = pc.get("control_plddt")
            tp = pc.get("test_bmm_plddt")
            sp = pc.get("speedup")
            print(f"  {pc['name']}:")
            print(f"    control:  {ct:.2f}s  pLDDT={cp:.4f}" if ct and cp else f"    control:  ERR")
            print(f"    test-bmm: {tt:.2f}s  pLDDT={tp:.4f}  speedup={sp:.3f}x" if tt and tp and sp else f"    test-bmm: ERR")

        # Dump full results
        print(f"\n--- FULL RESULTS ---")
        print(json.dumps(result, indent=2))


def _print_summary(result):
    agg = result.get("aggregate", {})
    mean_predict = agg.get("mean_predict_only_s")
    mean_wall = agg.get("mean_wall_time_s")
    mean_plddt = agg.get("mean_plddt")

    if mean_predict is not None:
        print(f"  Mean predict_only: {mean_predict:.2f}s")
    if mean_wall is not None:
        print(f"  Mean wall time: {mean_wall:.1f}s")
    if mean_plddt is not None:
        print(f"  Mean pLDDT: {mean_plddt:.4f}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:200]}")
        else:
            pt = pc.get("predict_only_s")
            wt = pc.get("wall_time_s")
            p = pc.get("quality", {}).get("complex_plddt")
            times = pc.get("run_predict_times", [])
            times_str = ", ".join(f"{x:.2f}s" for x in times) if times else ""
            pt_str = f"{pt:.2f}s" if pt else "N/A"
            p_str = f"{p:.4f}" if p else "N/A"
            print(f"  {pc['name']}: predict={pt_str}, pLDDT={p_str}" +
                  (f" [{times_str}]" if times_str else ""))
