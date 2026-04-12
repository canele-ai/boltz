"""Evaluation for sparse triangular multiplication optimization.

Tests sparse window TriangleMult combined with ODE + TF32 + bf16.
Uses eval-v4 infrastructure: pre-cached MSAs, predict-only timing.

Baseline: 25.04s mean predict-only (eval-v4)
Current best: 13.08s = 1.91x (ODE-12/0r + TF32 + bf16)

Usage:
    modal run orbits/sparse-tri-ops/eval_sparse.py --mode sanity
    modal run orbits/sparse-tri-ops/eval_sparse.py --mode eval --window 128
    modal run orbits/sparse-tri-ops/eval_sparse.py --mode sweep
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
    .pip_install("boltz==2.2.1")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_sparse.py"),
        remote_path="/eval/boltz_wrapper_sparse.py",
    )
)

app = modal.App("boltz-eval-sparse-tri", image=boltz_image)
msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)

# eval-v2-winner best config as baseline
BASE_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "diffusion_samples": 1,
    "seed": 42,
    "sparse_window": 0,
}

SWEEP_CONFIGS = {
    "baseline-cueq": {**BASE_CONFIG, "sparse_window": 0},
    "sparse-W64": {**BASE_CONFIG, "sparse_window": 64},
    "sparse-W128": {**BASE_CONFIG, "sparse_window": 128},
    "sparse-W256": {**BASE_CONFIG, "sparse_window": 256},
}


def _load_config_yaml() -> dict:
    import yaml
    with Path("/eval/config.yaml").open() as f:
        return yaml.safe_load(f)


def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path) -> Optional[Path]:
    """Inject pre-cached MSA paths into input YAML to skip MSA server."""
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

    print(f"[eval] MSA cache: injected {injected} MSA(s) for {target_name}")
    return cached_yaml


def _run_boltz_prediction(input_yaml: Path, out_dir: Path, config: dict) -> dict:
    wrapper = "/eval/boltz_wrapper_sparse.py"
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
    ]

    # Diffusion params
    if config.get("gamma_0") is not None:
        cmd.extend(["--gamma_0", str(config["gamma_0"])])
    if config.get("noise_scale") is not None:
        cmd.extend(["--noise_scale", str(config["noise_scale"])])

    # bf16 trunk
    if config.get("bf16_trunk"):
        cmd.append("--bf16_trunk")

    # Sparse window
    sparse_w = config.get("sparse_window", 0)
    if sparse_w > 0:
        cmd.extend(["--sparse_window", str(sparse_w)])

    # MSA: if _msa_cached flag set, YAML has msa: fields; otherwise use server
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        # Parse phase timestamps
        predict_start = predict_end = None
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
                f"exited {proc.returncode}.\n"
                f"STDOUT: {proc.stdout[-2000:]}\n"
                f"STDERR: {proc.stderr[-2000:]}"
            )
            return result

        result["quality"] = _parse_confidence(out_dir, input_yaml)
    except subprocess.TimeoutExpired:
        result["error"] = "Timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Error: {exc}"
    return result


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict:
    target_name = input_yaml.stem
    # Handle _cached suffix in filename
    if target_name.endswith("_cached"):
        target_name = target_name[:-7]

    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if not pred_base.exists():
            # Try with _cached suffix
            pred_base2 = out_dir / f"boltz_results_{target_name}_cached" / "predictions"
            if pred_base2.exists():
                pred_base = pred_base2
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Not found: {results_dir}"}

    conf_files = sorted(results_dir.glob("confidence_*.json"))
    if not conf_files:
        return {"error": "No confidence JSON"}

    with conf_files[0].open() as f:
        conf = json.load(f)

    quality = {}
    for key in ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]:
        if key in conf:
            quality[key] = conf[key]
    return quality


def _compute_aggregates(results: dict, eval_config: dict) -> dict:
    successful = [
        r for r in results["per_complex"]
        if r["error"] is None and r.get("predict_only_s") is not None
    ]
    test_cases = eval_config.get("test_cases", [])

    if not successful:
        return {"error": "No successful test cases"}

    # Use predict_only_s as primary timing (eval-v4)
    predict_times = [r["predict_only_s"] for r in successful]
    mean_predict = sum(predict_times) / len(predict_times)

    plddts = [
        r["quality"]["complex_plddt"] for r in successful
        if "complex_plddt" in r["quality"]
        and r["quality"]["complex_plddt"] is not None
        and isinstance(r["quality"]["complex_plddt"], (int, float))
        and 0 <= r["quality"]["complex_plddt"] <= 1
    ]

    agg = {
        "num_successful": len(successful),
        "num_total": len(test_cases),
        "mean_predict_only_s": mean_predict,
        "total_predict_only_s": sum(predict_times),
        "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
    }

    # eval-v4 baseline: 25.04s mean predict-only
    baseline_predict_time = 25.04
    baseline_plddt = eval_config.get("baseline", {}).get("mean_plddt", 0.7170)

    if mean_predict > 0:
        agg["speedup_vs_baseline"] = baseline_predict_time / mean_predict

    if plddts:
        mean_plddt = sum(plddts) / len(plddts)
        agg["plddt_delta_pp"] = (mean_plddt - baseline_plddt) * 100.0
        regression = (baseline_plddt - mean_plddt) * 100.0
        agg["passes_quality_gate"] = regression <= 2.0

    return agg


@app.function(gpu="L40S", timeout=7200, volumes={"/msa_cache": msa_cache})
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    import statistics
    import torch

    config = json.loads(config_json)
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": config,
        "num_runs": num_runs,
        "per_complex": [],
    }
    results["env"] = {
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    msa_cache_root = Path("/msa_cache")

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name, "error": f"Not found: {tc_yaml}",
                "wall_time_s": None, "predict_only_s": None, "quality": {},
            })
            continue

        run_predict_times = []
        run_wall_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Inject cached MSAs
            run_config = dict(config)
            effective_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
            if effective_yaml:
                run_config["_msa_cached"] = True
            else:
                effective_yaml = tc_yaml
                print(f"[eval] WARNING: No cached MSAs for {tc_name}, using server")

            sparse_w = config.get("sparse_window", 0)
            print(f"[eval] {tc_name} run {run_idx+1}/{num_runs} W={sparse_w}")

            pred_result = _run_boltz_prediction(effective_yaml, work_dir, run_config)

            if pred_result["error"]:
                last_error = pred_result["error"]
                print(f"[eval] ERROR: {last_error[:500]}")
                break

            if pred_result["predict_only_s"] is not None:
                run_predict_times.append(pred_result["predict_only_s"])
            if pred_result["wall_time_s"] is not None:
                run_wall_times.append(pred_result["wall_time_s"])
            run_qualities.append(pred_result["quality"])

            plddt = pred_result["quality"].get("complex_plddt", "N/A")
            pt = pred_result.get("predict_only_s")
            pt_str = f"{pt:.1f}s" if pt else "N/A"
            wt = pred_result.get("wall_time_s")
            wt_str = f"{wt:.1f}s" if wt else "N/A"
            print(f"[eval] {tc_name}: predict={pt_str} wall={wt_str} pLDDT={plddt}")

        if last_error:
            entry = {"name": tc_name, "wall_time_s": None, "predict_only_s": None,
                     "quality": {}, "error": last_error}
        else:
            median_predict = statistics.median(run_predict_times) if run_predict_times else None
            median_wall = statistics.median(run_wall_times) if run_wall_times else None
            all_plddts = [q.get("complex_plddt") for q in run_qualities
                         if q.get("complex_plddt") is not None]
            mean_plddt = sum(all_plddts) / len(all_plddts) if all_plddts else None

            merged_quality = dict(run_qualities[-1]) if run_qualities else {}
            if mean_plddt is not None:
                merged_quality["complex_plddt"] = mean_plddt

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


@app.function(gpu="L40S", timeout=600, volumes={"/msa_cache": msa_cache})
def sanity_check() -> str:
    import torch
    results = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import cuequivariance_torch
        results["cuequivariance_torch"] = cuequivariance_torch.__version__
    except ImportError:
        results["cuequivariance_torch"] = None

    msa_path = Path("/msa_cache")
    results["msa_cache_exists"] = msa_path.exists()
    if msa_path.exists():
        import os
        for root, dirs, files in os.walk(msa_path):
            for f in files:
                results.setdefault("msa_files", []).append(
                    os.path.join(root, f).replace(str(msa_path), "")
                )
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    mode: str = "eval",
    window: int = 128,
    config_name: str = "",
    num_runs: int = 1,
    validate: bool = False,
):
    if validate:
        num_runs = 3

    if mode == "sanity":
        print(sanity_check.remote())

    elif mode == "eval":
        if config_name and config_name in SWEEP_CONFIGS:
            cfg = SWEEP_CONFIGS[config_name]
        else:
            cfg = {**BASE_CONFIG, "sparse_window": window}

        print(f"[eval] Config: sparse_window={cfg.get('sparse_window')}, runs={num_runs}")
        result = json.loads(evaluate_config.remote(json.dumps(cfg), num_runs=num_runs))
        print(json.dumps(result, indent=2))
        _print_summary(result)

    elif mode == "sweep":
        print(f"[eval] Sweeping {len(SWEEP_CONFIGS)} configs, runs={num_runs}")

        names = list(SWEEP_CONFIGS.keys())
        jsons = [json.dumps(SWEEP_CONFIGS[n]) for n in names]
        runs_list = [num_runs] * len(names)

        all_results = {}
        for name, result_json in zip(names, evaluate_config.map(jsons, runs_list)):
            result = json.loads(result_json)
            all_results[name] = result
            print(f"\n{'='*60}\nCONFIG: {name}\n{'='*60}")
            _print_summary(result)

        # Comparison table
        print(f"\n{'='*70}")
        print(f"{'Config':<20} {'Predict(s)':>10} {'pLDDT':>8} {'Delta':>8} {'Speedup':>8} {'Gate':>6}")
        print("-" * 70)
        for name in names:
            agg = all_results[name].get("aggregate", {})
            t = agg.get("mean_predict_only_s")
            p = agg.get("mean_plddt")
            d = agg.get("plddt_delta_pp")
            s = agg.get("speedup_vs_baseline")
            g = agg.get("passes_quality_gate")
            t_s = f"{t:.2f}" if t is not None else "ERR"
            p_s = f"{p:.4f}" if p is not None else "ERR"
            d_s = f"{d:+.2f}pp" if d is not None else "N/A"
            s_s = f"{s:.2f}x" if s is not None else "ERR"
            g_s = "PASS" if g else "FAIL" if g is not None else "N/A"
            print(f"{name:<20} {t_s:>10} {p_s:>8} {d_s:>8} {s_s:>8} {g_s:>6}")

        print("\n--- FULL RESULTS ---")
        print(json.dumps(all_results, indent=2))


def _print_summary(result: dict):
    agg = result.get("aggregate", {})
    if agg.get("mean_predict_only_s") is not None:
        print(f"  Mean predict time: {agg['mean_predict_only_s']:.2f}s")
    if agg.get("mean_plddt") is not None:
        print(f"  Mean pLDDT: {agg['mean_plddt']:.4f}")
    if agg.get("speedup_vs_baseline") is not None:
        print(f"  Speedup vs baseline: {agg['speedup_vs_baseline']:.2f}x")
    if agg.get("plddt_delta_pp") is not None:
        print(f"  pLDDT delta: {agg['plddt_delta_pp']:+.2f}pp")
    if agg.get("passes_quality_gate") is not None:
        print(f"  Quality gate: {'PASS' if agg['passes_quality_gate'] else 'FAIL'}")

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"  {pc['name']}: ERROR - {pc['error'][:200]}")
        else:
            pt = pc.get("predict_only_s")
            p = pc.get("quality", {}).get("complex_plddt")
            pts = pc.get("run_predict_times", [])
            pt_s = f"{pt:.2f}s" if pt is not None else "N/A"
            p_s = f"{p:.4f}" if p is not None else "N/A"
            pts_s = ", ".join(f"{x:.1f}s" for x in pts) if pts else ""
            print(f"  {pc['name']}: predict={pt_s}, pLDDT={p_s}" +
                  (f" [{pts_s}]" if pts_s else ""))
