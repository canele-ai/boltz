"""Evaluator for fast-model-load: persistent model loading + bypass-lightning.

Loads Boltz-2 model ONCE, predicts all test complexes in-process, reports
amortized speedup. Combines with bypass-lightning optimizations.

Usage:
    # Sanity check
    modal run orbits/fast-model-load/eval_persistent.py --mode sanity

    # Single seed eval
    modal run orbits/fast-model-load/eval_persistent.py --mode eval

    # Multi-seed validation (3 seeds in parallel)
    modal run orbits/fast-model-load/eval_persistent.py --mode eval --validate
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
        str(ORBIT_DIR / "persistent_predict.py"),
        remote_path="/eval/persistent_predict.py",
    )
)

app = modal.App("boltz-eval-fast-model-load", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)
boltz_cache = modal.Volume.from_name("boltz-model-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path):
    """Inject cached MSA paths into input YAML."""
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
    return cached_yaml


def _parse_persistent_results(stdout: str) -> dict[str, Any]:
    """Parse JSON output from persistent_predict.py."""
    # The script outputs JSON to stdout; find the JSON block
    lines = stdout.strip().split("\n")
    json_lines = []
    in_json = False
    for line in lines:
        if line.strip().startswith("{"):
            in_json = True
        if in_json:
            json_lines.append(line)

    if json_lines:
        try:
            return json.loads("\n".join(json_lines))
        except json.JSONDecodeError:
            pass
    return {"error": f"Could not parse JSON from stdout: {stdout[-500:]}"}


# ---------------------------------------------------------------------------
# Modal function: single-seed evaluation
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_cache, "/boltz_cache": boltz_cache},
)
def prep_cache() -> str:
    """Download model weights and create pickle on persistent volume (run once)."""
    import os
    import warnings
    from dataclasses import asdict, dataclass

    import torch

    cache_dir = Path("/boltz_cache")
    ckpt_path = cache_dir / "boltz2_conf.ckpt"
    pickle_path = cache_dir / "boltz2_full_model.pt"

    if pickle_path.exists():
        return json.dumps({"status": "pickle exists", "files": [str(p) for p in cache_dir.iterdir()]})

    # Download model if needed
    import boltz.main as boltz_main
    if not ckpt_path.exists():
        boltz_main.download_boltz2(cache_dir)

    # Create pickle for fast loading
    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    from boltz.main import (
        Boltz2, Boltz2DiffusionParams,
        PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams,
    )

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    # Patch diffusion params for ODE mode
    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = 0.0
        gamma_min: float = 1.0
        noise_scale: float = 1.003
        rho: float = 7
        step_scale: float = 1.5
        sigma_min: float = 0.0001
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True

    boltz_main.Boltz2DiffusionParams = PatchedBoltz2DiffusionParams

    diffusion_params = PatchedBoltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": 12,
        "diffusion_samples": 1,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    print("[prep_cache] Creating model pickle...")
    model = Boltz2.load_from_checkpoint(
        ckpt_path,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    torch.save(model, pickle_path)
    print(f"[prep_cache] Pickle saved: {os.path.getsize(pickle_path) / 1e9:.2f} GB")

    boltz_cache.commit()
    return json.dumps({"status": "pickle created", "files": [str(p) for p in cache_dir.iterdir()]})


@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_cache, "/boltz_cache": boltz_cache},
)
def evaluate_seed(seed: int) -> str:
    """Run persistent-model evaluation for a single seed."""
    import yaml

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())

    work_dir = Path(f"/tmp/boltz_persistent_{seed}_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Prepare input YAMLs (inject cached MSAs)
    input_yamls = []
    for tc in test_cases:
        tc_yaml = Path("/eval") / tc["yaml"]
        if not tc_yaml.exists():
            return json.dumps({"error": f"Test case YAML not found: {tc_yaml}", "seed": seed})

        effective_yaml = tc_yaml
        if use_msa_cache:
            tc_work = work_dir / tc["name"]
            tc_work.mkdir(parents=True, exist_ok=True)
            cached = _inject_cached_msas(tc_yaml, msa_cache_root, tc_work)
            if cached is not None:
                effective_yaml = cached

        input_yamls.append(str(effective_yaml))

    # Build command for persistent_predict.py
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "/eval/persistent_predict.py",
        "--inputs", *input_yamls,
        "--out_dir", str(out_dir),
        "--cache", "/boltz_cache",
        "--sampling_steps", "12",
        "--recycling_steps", "0",
        "--diffusion_samples", "1",
        "--gamma_0", "0.0",
        "--noise_scale", "1.003",
        "--matmul_precision", "high",
        "--bf16_trunk",
        "--use_pickle",
        "--override",
        "--seed", str(seed),
    ]

    # MSA: if we injected cached MSAs, don't use server
    # The persistent_predict.py handles MSA from what's in the YAML
    if not use_msa_cache:
        cmd.append("--use_msa_server")

    print(f"[eval-persistent] seed={seed}, cmd={' '.join(cmd[:10])}...")

    t_start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    t_end = time.perf_counter()
    total_wall_time = t_end - t_start

    result: dict[str, Any] = {
        "seed": seed,
        "total_wall_time_s": total_wall_time,
    }

    if proc.returncode != 0:
        result["error"] = (
            f"persistent_predict exited with code {proc.returncode}.\n"
            f"STDOUT: {proc.stdout[-3000:] if proc.stdout else '(empty)'}\n"
            f"STDERR: {proc.stderr[-5000:] if proc.stderr else '(empty)'}"
        )
        return json.dumps(result, indent=2)

    # Parse the JSON output
    parsed = _parse_persistent_results(proc.stdout)
    if "error" in parsed and not parsed.get("per_complex"):
        result["error"] = parsed["error"]
        result["stderr"] = proc.stderr[-2000:] if proc.stderr else "(empty)"
        return json.dumps(result, indent=2)

    result["persistent_results"] = parsed
    result["stderr_tail"] = proc.stderr[-1000:] if proc.stderr else "(empty)"

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Modal function: aggregate multi-seed results
# ---------------------------------------------------------------------------

@app.function(timeout=600)
def aggregate_results(seed_results_json: list[str]) -> str:
    """Aggregate results from multiple seeds."""
    eval_config_path = Path("/eval/config.yaml")
    import yaml
    with eval_config_path.open() as f:
        eval_config = yaml.safe_load(f)

    baseline = eval_config.get("baseline", {})
    baseline_mean_wall = baseline.get("mean_wall_time_s", 53.57)
    baseline_plddt = baseline.get("mean_plddt", 0.717)
    baseline_per_complex = {pc["name"]: pc for pc in baseline.get("per_complex", [])}

    seed_results = [json.loads(s) for s in seed_results_json]

    # Check for errors
    errors = [r for r in seed_results if "error" in r]
    if errors:
        return json.dumps({
            "error": f"{len(errors)} seed(s) failed",
            "errors": [{"seed": r["seed"], "error": r["error"][:3000]} for r in errors],
        }, indent=2)

    # Extract per-complex data across seeds
    all_onetime_costs = []
    all_model_loads = []
    all_per_complex = {}  # name -> list of dicts

    for sr in seed_results:
        pr = sr.get("persistent_results", {})
        all_model_loads.append(pr.get("model_load_time_s", 0))
        all_onetime_costs.append(pr.get("one_time_cost_s", pr.get("model_load_time_s", 0)))

        for pc in pr.get("per_complex", []):
            name = pc["name"]
            if name not in all_per_complex:
                all_per_complex[name] = []
            all_per_complex[name].append(pc)

    num_complexes = len(all_per_complex)
    num_seeds = len(seed_results)

    mean_model_load = sum(all_model_loads) / len(all_model_loads) if all_model_loads else 0
    std_model_load = (
        (sum((x - mean_model_load)**2 for x in all_model_loads) / len(all_model_loads))**0.5
        if len(all_model_loads) > 1 else 0
    )
    mean_onetime = sum(all_onetime_costs) / len(all_onetime_costs) if all_onetime_costs else 0

    # Per-complex aggregation
    per_complex_summary = []
    all_predict_times = []
    all_total_times = []
    all_plddts = []

    for name in sorted(all_per_complex.keys()):
        entries = all_per_complex[name]
        predict_times = [e["predict_time_s"] for e in entries]
        process_times = [e["process_time_s"] for e in entries]
        total_times = [e["total_per_complex_s"] for e in entries]
        plddts = [
            e["quality"]["complex_plddt"] for e in entries
            if "complex_plddt" in e.get("quality", {})
        ]

        mean_predict = sum(predict_times) / len(predict_times)
        mean_process = sum(process_times) / len(process_times)
        mean_total = sum(total_times) / len(total_times)
        mean_plddt = sum(plddts) / len(plddts) if plddts else None

        all_predict_times.extend(predict_times)
        all_total_times.extend(total_times)
        if plddts:
            all_plddts.extend(plddts)

        pc_summary = {
            "name": name,
            "mean_predict_s": mean_predict,
            "mean_process_s": mean_process,
            "mean_total_s": mean_total,
            "mean_plddt": mean_plddt,
            "predict_times": predict_times,
            "plddts": plddts,
        }

        # Quality gate per complex
        bl = baseline_per_complex.get(name, {})
        bl_plddt = bl.get("complex_plddt")
        if bl_plddt is not None and mean_plddt is not None:
            pc_summary["plddt_delta_pp"] = (mean_plddt - bl_plddt) * 100.0
            pc_summary["passes_5pp_gate"] = (bl_plddt - mean_plddt) * 100.0 <= 5.0

        per_complex_summary.append(pc_summary)

    # Compute amortized wall time per complex (one_time = model_load + warmup)
    mean_per_complex_total = sum(all_total_times) / len(all_total_times) if all_total_times else 0
    amortized_onetime = mean_onetime / num_complexes if num_complexes > 0 else 0
    amortized_wall_per_complex = amortized_onetime + mean_per_complex_total

    # Speedup
    speedup = baseline_mean_wall / amortized_wall_per_complex if amortized_wall_per_complex > 0 else 0

    # Quality gate
    mean_plddt = sum(all_plddts) / len(all_plddts) if all_plddts else None
    passes_quality_gate = True
    plddt_delta_pp = None
    if mean_plddt is not None and baseline_plddt is not None:
        plddt_delta_pp = (mean_plddt - baseline_plddt) * 100.0
        regression = (baseline_plddt - mean_plddt) * 100.0
        if regression > 2.0:
            passes_quality_gate = False

    # Per-complex gate
    for pc in per_complex_summary:
        if pc.get("passes_5pp_gate") is False:
            passes_quality_gate = False

    # Compute per-seed amortized metrics for std calculation
    per_seed_amortized = []
    for sr in seed_results:
        pr = sr.get("persistent_results", {})
        onetime = pr.get("one_time_cost_s", pr.get("model_load_time_s", 0))
        pcs = pr.get("per_complex", [])
        if pcs:
            mean_pc = sum(p["total_per_complex_s"] for p in pcs) / len(pcs)
            amort = onetime / len(pcs) + mean_pc
            per_seed_amortized.append(amort)

    mean_amortized = sum(per_seed_amortized) / len(per_seed_amortized) if per_seed_amortized else 0
    std_amortized = (
        (sum((x - mean_amortized)**2 for x in per_seed_amortized) / len(per_seed_amortized))**0.5
        if len(per_seed_amortized) > 1 else 0
    )

    per_seed_speedups = [baseline_mean_wall / a if a > 0 else 0 for a in per_seed_amortized]
    mean_speedup = sum(per_seed_speedups) / len(per_seed_speedups) if per_seed_speedups else 0
    std_speedup = (
        (sum((x - mean_speedup)**2 for x in per_seed_speedups) / len(per_seed_speedups))**0.5
        if len(per_seed_speedups) > 1 else 0
    )

    aggregate = {
        "num_seeds": num_seeds,
        "num_complexes": num_complexes,
        "mean_model_load_s": mean_model_load,
        "std_model_load_s": std_model_load,
        "mean_onetime_cost_s": mean_onetime,
        "amortized_onetime_per_complex_s": amortized_onetime,
        "mean_per_complex_total_s": mean_per_complex_total,
        "amortized_wall_per_complex_s": amortized_wall_per_complex,
        "std_amortized_wall_s": std_amortized,
        "per_seed_amortized_wall_s": per_seed_amortized,
        "speedup": speedup,
        "per_seed_speedups": per_seed_speedups,
        "mean_speedup": mean_speedup,
        "std_speedup": std_speedup,
        "mean_plddt": mean_plddt,
        "plddt_delta_pp": plddt_delta_pp,
        "passes_quality_gate": passes_quality_gate,
        "per_complex": per_complex_summary,
        "baseline_mean_wall_time_s": baseline_mean_wall,
        "baseline_mean_plddt": baseline_plddt,
    }

    return json.dumps(aggregate, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    seed: int = 42,
    validate: bool = False,
):
    """Fast-model-load evaluation harness.

    Modes:
        sanity  - Quick environment check
        eval    - Run persistent model evaluation

    Usage:
        modal run orbits/fast-model-load/eval_persistent.py --mode sanity
        modal run orbits/fast-model-load/eval_persistent.py --mode eval
        modal run orbits/fast-model-load/eval_persistent.py --mode eval --validate
    """
    if mode == "sanity":
        print("[eval-persistent] Running sanity check...")
        result_json = evaluate_seed.remote(42)
        result = json.loads(result_json)
        if "error" in result:
            print(f"FAILED: {result['error'][:500]}")
        else:
            print(json.dumps(result, indent=2))
        return

    if mode == "eval":
        if validate:
            seeds = [42, 123, 7]
        else:
            seeds = [seed]

        # Ensure model weights are cached on volume
        print("[eval-persistent] Ensuring model cache is populated...")
        cache_result = prep_cache.remote()
        print(f"[eval-persistent] Cache: {cache_result}")

        print(f"[eval-persistent] Running with seeds={seeds}")

        # Run seeds in parallel using Modal .map()
        seed_results = list(evaluate_seed.map(seeds))

        print("\n--- PER-SEED RESULTS ---")
        for sr_json in seed_results:
            sr = json.loads(sr_json)
            if "error" in sr:
                print(f"  Seed {sr.get('seed', '?')}: ERROR - {sr['error'][:2000]}")
            else:
                pr = sr.get("persistent_results", {})
                print(f"  Seed {sr['seed']}: load={pr.get('model_load_time_s', 0):.1f}s, "
                      f"amortized_wall={pr.get('amortized_wall_time_per_complex_s', 0):.1f}s")

        # Aggregate
        if len(seeds) >= 1:
            agg_json = aggregate_results.remote(seed_results)
            agg = json.loads(agg_json)
            print("\n--- AGGREGATE RESULTS ---")
            print(json.dumps(agg, indent=2))

            # Summary
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"  Seeds: {len(seeds)}")
            print(f"  Model load: {agg.get('mean_model_load_s', 0):.1f}s +/- {agg.get('std_model_load_s', 0):.1f}s")
            print(f"  Amortized load/complex: {agg.get('amortized_load_per_complex_s', 0):.1f}s")
            print(f"  Mean predict+process/complex: {agg.get('mean_per_complex_total_s', 0):.1f}s")
            print(f"  Amortized wall/complex: {agg.get('amortized_wall_per_complex_s', 0):.1f}s "
                  f"+/- {agg.get('std_amortized_wall_s', 0):.1f}s")
            print(f"  Speedup: {agg.get('mean_speedup', 0):.2f}x +/- {agg.get('std_speedup', 0):.2f}x")
            print(f"  Mean pLDDT: {agg.get('mean_plddt', 'N/A')}")
            if agg.get('plddt_delta_pp') is not None:
                print(f"  pLDDT delta: {agg['plddt_delta_pp']:+.2f} pp")
            print(f"  Quality gate: {'PASS' if agg.get('passes_quality_gate') else 'FAIL'}")

            print("\n  Per-complex breakdown:")
            for pc in agg.get("per_complex", []):
                plddt_str = f"{pc['mean_plddt']:.4f}" if pc.get('mean_plddt') is not None else "N/A"
                delta_str = f"{pc.get('plddt_delta_pp', 0):+.1f}pp" if pc.get('plddt_delta_pp') is not None else ""
                print(f"    {pc['name']}: process={pc['mean_process_s']:.1f}s, "
                      f"predict={pc['mean_predict_s']:.1f}s, "
                      f"pLDDT={plddt_str} {delta_str}")
