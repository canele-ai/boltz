"""Eval script for INT8 post-training quantization (PTQ) on Boltz-2.

Uses torchao for GPU-compatible INT8 quantization of nn.Linear layers.
Stacks with TF32 + ODE-12/0r (the eval-v2-winner config).

Modes:
  w8     - INT8 weight-only (torchao): weights quantized, activations float
  w8a8   - INT8 dynamic activation + weight (torchao): both quantized at runtime
  w4     - INT4 weight-only (torchao): more aggressive, may hurt quality
  none   - No quantization (baseline comparison with same ODE-12/0r config)

Usage:
    modal run orbits/int8-ptq/eval_int8_ptq.py
    modal run orbits/int8-ptq/eval_int8_ptq.py --mode w8
    modal run orbits/int8-ptq/eval_int8_ptq.py --mode w8a8
    modal run orbits/int8-ptq/eval_int8_ptq.py --mode none
"""
from __future__ import annotations

import json
import math
import os
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"

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
    .pip_install(
        # torchao for GPU-compatible INT8 quantization
        # torch 2.6.0 requires torchao 0.8.x (0.9+ needs torch 2.7+)
        "torchao==0.8.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(str(SCRIPT_DIR / "int8_wrapper.py"), remote_path="/eval/int8_wrapper.py")
)

app = modal.App("boltz-int8-ptq-eval", image=boltz_image)

# Persistent volume for pre-cached MSAs (eval-v3)
msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)


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

    Mirrors the logic in research/eval/evaluator.py.
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

    print(f"[int8-ptq] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    """Parse Boltz confidence summary JSON from the prediction output."""
    target_name = input_yaml.stem
    # Handle both original and cached YAML names
    if target_name.endswith("_cached"):
        target_name = target_name[:-7]  # strip _cached suffix
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    if not results_dir.exists():
        # Try _cached variant
        cached_name = target_name + "_cached"
        results_dir2 = out_dir / f"boltz_results_{cached_name}" / "predictions" / cached_name
        if results_dir2.exists():
            results_dir = results_dir2
        else:
            # Search for any prediction directory
            for pattern in [f"boltz_results_{target_name}*", "boltz_results_*"]:
                parent = out_dir
                matches = list(parent.glob(pattern))
                for m in matches:
                    pred_base = m / "predictions"
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

    quality: dict[str, Any] = {}
    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]
    return quality


def _run_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
    quantize_mode: str,
) -> dict[str, Any]:
    """Run Boltz-2 prediction with INT8 quantization applied."""
    wrapper = "/eval/int8_wrapper.py"
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 12)),
        "--recycling_steps", str(config.get("recycling_steps", 0)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--matmul_precision", config.get("matmul_precision", "high"),
        "--quantize_mode", quantize_mode,
    ]

    # MSA: if cached, the YAML already has msa: fields — don't use server.
    # Otherwise fall back to server.
    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    result: dict[str, Any] = {"wall_time_s": None, "quality": {}, "error": None}

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"boltz exited with code {proc.returncode}.\n"
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


@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_volume},
)
def evaluate(quantize_mode: str = "w8", num_runs: int = 1) -> str:
    """Run INT8 PTQ evaluation on all test cases."""
    config = {
        "sampling_steps": 12,
        "recycling_steps": 0,
        "matmul_precision": "high",
        "diffusion_samples": 1,
        "seed": 42,
    }

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    # Check MSA cache
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())
    if use_msa_cache:
        print(f"[int8-ptq] MSA cache found at {msa_cache_root}")
        print(f"[int8-ptq] Contents: {list(msa_cache_root.iterdir())}")
    else:
        print("[int8-ptq] WARNING: No MSA cache found, will use MSA server (adds latency)")

    results: dict[str, Any] = {
        "config": config,
        "quantize_mode": quantize_mode,
        "num_runs": num_runs,
        "msa_cached": use_msa_cache,
        "per_complex": [],
        "aggregate": {},
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name, "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None, "quality": {},
            })
            continue

        run_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Inject cached MSAs if available
            run_config = dict(config)
            effective_yaml = tc_yaml
            if use_msa_cache:
                cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached_yaml is not None:
                    effective_yaml = cached_yaml
                    run_config["_msa_cached"] = True

            print(
                f"[int8-ptq] {tc_name} run {run_idx+1}/{num_runs} "
                f"mode={quantize_mode}"
                f"{' (MSA cached)' if run_config.get('_msa_cached') else ''}"
            )

            pred = _run_prediction(effective_yaml, work_dir, run_config, quantize_mode)

            if pred["error"] is not None:
                last_error = pred["error"]
                print(f"[int8-ptq] ERROR: {last_error[:500]}")
                break

            if pred["wall_time_s"] is not None:
                run_times.append(pred["wall_time_s"])
                print(f"[int8-ptq]   wall_time={pred['wall_time_s']:.2f}s")
            if pred["quality"]:
                plddt = pred["quality"].get("complex_plddt")
                print(f"[int8-ptq]   plddt={plddt}")
            run_qualities.append(pred["quality"])

        if last_error is not None:
            entry = {"name": tc_name, "wall_time_s": None, "quality": {}, "error": last_error}
        else:
            median_time = statistics.median(run_times) if run_times else None
            plddts = [q["complex_plddt"] for q in run_qualities if "complex_plddt" in q]
            mean_plddt = (sum(plddts) / len(plddts)) if plddts else None
            merged_quality = dict(run_qualities[-1]) if run_qualities else {}
            if mean_plddt is not None:
                merged_quality["complex_plddt"] = mean_plddt
            entry = {
                "name": tc_name, "wall_time_s": median_time,
                "quality": merged_quality, "error": None, "run_times": run_times,
            }

        results["per_complex"].append(entry)

    # Aggregate
    successful = [r for r in results["per_complex"] if r["error"] is None and r["wall_time_s"] is not None]

    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [r["quality"]["complex_plddt"] for r in successful if "complex_plddt" in r["quality"]]
        iptms = [r["quality"]["iptm"] for r in successful if "iptm" in r["quality"]]

        results["aggregate"] = {
            "num_successful": len(successful),
            "num_total": len(test_cases),
            "total_wall_time_s": total_time,
            "mean_wall_time_s": mean_time,
            "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
            "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
        }

        # Compare to eval-v4 baseline (25.04s)
        results["aggregate"]["speedup_vs_evalv4"] = 25.04 / mean_time

        # Compare to eval-v3 baseline from config.yaml
        baseline = eval_config.get("baseline")
        if baseline:
            baseline_time = baseline.get("mean_wall_time_s")
            baseline_plddt = baseline.get("mean_plddt")
            if baseline_time and mean_time > 0:
                results["aggregate"]["speedup_vs_200step_baseline"] = baseline_time / mean_time
            if baseline_plddt and plddts:
                mean_plddt_val = sum(plddts) / len(plddts)
                results["aggregate"]["plddt_delta_pp"] = (mean_plddt_val - baseline_plddt) * 100.0
                regression = (baseline_plddt - mean_plddt_val) * 100.0
                results["aggregate"]["passes_quality_gate"] = regression <= 2.0

                # Per-complex quality floor
                if baseline.get("per_complex"):
                    baseline_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                    violations = {}
                    for r in successful:
                        bl = baseline_by_name.get(r["name"])
                        if bl and bl.get("complex_plddt") is not None:
                            case_plddt = r["quality"].get("complex_plddt")
                            if case_plddt is not None:
                                case_reg = (bl["complex_plddt"] - case_plddt) * 100.0
                                if case_reg > 5.0:
                                    results["aggregate"]["passes_quality_gate"] = False
                                    violations[r["name"]] = f"-{case_reg:.1f}pp"
                    if violations:
                        results["aggregate"]["per_complex_regression"] = violations

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    mode: str = "w8",
    num_runs: int = 1,
    validate: bool = False,
):
    """INT8 PTQ evaluation.

    Usage:
        modal run orbits/int8-ptq/eval_int8_ptq.py
        modal run orbits/int8-ptq/eval_int8_ptq.py --mode w8 --validate
        modal run orbits/int8-ptq/eval_int8_ptq.py --mode none
    """
    if validate:
        num_runs = 3

    print(f"[int8-ptq] Mode: {mode}, num_runs: {num_runs}")
    result_json = evaluate.remote(quantize_mode=mode, num_runs=num_runs)
    result = json.loads(result_json)
    print(json.dumps(result, indent=2))

    agg = result.get("aggregate", {})
    speedup = agg.get("speedup_vs_evalv4")
    plddt_delta = agg.get("plddt_delta_pp")
    passes = agg.get("passes_quality_gate")

    if speedup is not None:
        print(f"\n[int8-ptq] Speedup vs eval-v4 baseline (25.04s): {speedup:.2f}x")
    if plddt_delta is not None:
        print(f"[int8-ptq] pLDDT delta vs 200-step baseline: {plddt_delta:+.2f} pp")
    if passes is not None:
        print(f"[int8-ptq] Quality gate: {'PASS' if passes else 'FAIL'}")
