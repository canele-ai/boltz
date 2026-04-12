"""Recycling sweep evaluator for eval-v5: measure speed/quality across recycling_steps={0,1,2,3}.

All configs use the bypass-lightning wrapper with:
- ODE sampling (gamma_0=0.0), 12 diffusion steps
- TF32 matmul, bf16 trunk, cuequivariance kernels
- CUDA warmup pass

Runs 3 seeds (42, 123, 7) per recycling config in parallel using Modal .map().

Usage:
    modal run orbits/recycling-v5-study/eval_recycling.py --mode sanity
    modal run orbits/recycling-v5-study/eval_recycling.py --mode sweep
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
BYPASS_WRAPPER = REPO_ROOT / "orbits" / "bypass-lightning" / "boltz_bypass_wrapper.py"

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
        str(BYPASS_WRAPPER),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
)

app = modal.App("boltz-recycling-v5-study", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)

# ---------------------------------------------------------------------------
# Seeds and configs
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 7]

# Base config shared by all runs
BASE_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "matmul_precision": "high",
    "diffusion_samples": 1,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "enable_kernels": True,
    "bf16_trunk": True,
    "cuda_warmup": True,
}

RECYCLING_STEPS_TO_SWEEP = [0, 1, 2, 3]


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
    print(f"[recycling-sweep] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _run_boltz_bypass(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction using the bypass-lightning wrapper."""
    wrapper = str(Path("/eval/boltz_bypass_wrapper.py"))
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

    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    if config.get("cuda_warmup", False):
        cmd.append("--cuda_warmup")

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
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse phase timestamps for predict-only timing
        predict_start = None
        predict_end = None
        for line in proc.stderr.split("\n"):
            if "[PHASE] predict_start=" in line:
                predict_start = float(line.split("=")[1])
            elif "[PHASE] predict_end=" in line:
                predict_end = float(line.split("=")[1])

        if predict_start is not None and predict_end is not None:
            result["predict_only_s"] = predict_end - predict_start

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

    # Also look for CA RMSD from structural validation
    struct_files = sorted(results_dir.glob("structural_*.json"))
    if struct_files:
        with struct_files[0].open() as f:
            struct = json.load(f)
        if "ca_rmsd" in struct:
            quality["ca_rmsd"] = struct["ca_rmsd"]

    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]

    return quality


# ---------------------------------------------------------------------------
# Modal function: single (recycling_steps, seed, test_case) evaluation
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_cache},
)
def evaluate_single(recycling_steps: int, seed: int, tc_name: str, tc_yaml_rel: str) -> str:
    """Run a single evaluation: one recycling config, one seed, all 3 test cases (or just one).

    Returns JSON with per-complex results for this seed.
    """
    config = {**BASE_CONFIG, "recycling_steps": recycling_steps, "seed": seed}

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())

    tc_yaml = Path("/eval") / tc_yaml_rel

    if not tc_yaml.exists():
        return json.dumps({
            "recycling_steps": recycling_steps,
            "seed": seed,
            "tc_name": tc_name,
            "error": f"YAML not found: {tc_yaml}",
        })

    work_dir = Path(f"/tmp/boltz_eval/r{recycling_steps}_s{seed}_{tc_name}_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    run_config = dict(config)
    effective_yaml = tc_yaml
    if use_msa_cache:
        cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
        if cached_yaml is not None:
            effective_yaml = cached_yaml
            run_config["_msa_cached"] = True

    print(f"[recycling-sweep] recycling={recycling_steps}, seed={seed}, tc={tc_name}"
          f"{' (MSA cached)' if run_config.get('_msa_cached') else ''}")

    pred_result = _run_boltz_bypass(effective_yaml, work_dir, run_config)

    return json.dumps({
        "recycling_steps": recycling_steps,
        "seed": seed,
        "tc_name": tc_name,
        "wall_time_s": pred_result["wall_time_s"],
        "predict_only_s": pred_result.get("predict_only_s"),
        "quality": pred_result["quality"],
        "error": pred_result["error"],
    })


# ---------------------------------------------------------------------------
# Modal function: full sweep
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=600, volumes={"/msa_cache": msa_cache})
def sanity_check() -> str:
    """Quick environment check."""
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
        results["msa_cache_entries"] = len(list(msa_cache_path.iterdir()))
    else:
        results["msa_cache_entries"] = 0

    bypass_path = Path("/eval/boltz_bypass_wrapper.py")
    results["bypass_wrapper_found"] = bypass_path.exists()

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "sweep",
):
    """Recycling sweep for eval-v5.

    Modes:
        sanity - Verify environment
        sweep  - Run full recycling sweep (0,1,2,3) x 3 seeds x 3 test cases

    Usage:
        modal run orbits/recycling-v5-study/eval_recycling.py --mode sanity
        modal run orbits/recycling-v5-study/eval_recycling.py --mode sweep
    """
    # pyyaml may not be in Modal's local Python env; use a fallback
    try:
        import yaml
    except ImportError:
        # Minimal YAML loading via subprocess
        import ast
        _yaml_available = False
    else:
        _yaml_available = True

    if mode == "sanity":
        print("[recycling-sweep] Running sanity check...")
        result_json = sanity_check.remote()
        print(result_json)
        return

    if mode != "sweep":
        print(f"Unknown mode: {mode}")
        return

    # Load test cases
    config_path = EVAL_DIR / "config.yaml"
    if _yaml_available:
        with config_path.open() as f:
            eval_config = yaml.safe_load(f)
    else:
        # Fallback: parse just what we need from config.yaml
        # We need test_cases and baseline sections
        import subprocess as _sp
        _r = _sp.run(
            ["python3", "-c",
             f"import yaml, json; print(json.dumps(yaml.safe_load(open('{config_path}'))))"],
            capture_output=True, text=True)
        eval_config = json.loads(_r.stdout)
    test_cases = eval_config.get("test_cases", [])

    print(f"[recycling-sweep] Sweeping recycling_steps={RECYCLING_STEPS_TO_SWEEP}")
    print(f"[recycling-sweep] Seeds={SEEDS}")
    print(f"[recycling-sweep] Test cases: {[tc['name'] for tc in test_cases]}")
    print(f"[recycling-sweep] Total runs: {len(RECYCLING_STEPS_TO_SWEEP) * len(SEEDS) * len(test_cases)}")

    # Build all (recycling_steps, seed, test_case) tuples
    tasks = []
    for r in RECYCLING_STEPS_TO_SWEEP:
        for s in SEEDS:
            for tc in test_cases:
                tasks.append((r, s, tc["name"], tc["yaml"]))

    # Launch ALL tasks in parallel using .map()
    recycling_args = [t[0] for t in tasks]
    seed_args = [t[1] for t in tasks]
    tc_name_args = [t[2] for t in tasks]
    tc_yaml_args = [t[3] for t in tasks]

    print(f"[recycling-sweep] Launching {len(tasks)} parallel evaluations...")

    raw_results = []
    for result_json in evaluate_single.map(
        recycling_args, seed_args, tc_name_args, tc_yaml_args
    ):
        r = json.loads(result_json)
        raw_results.append(r)
        status = "OK" if r.get("error") is None else f"ERROR: {r['error'][:80]}"
        wall = r.get("wall_time_s")
        pred = r.get("predict_only_s")
        plddt = r.get("quality", {}).get("complex_plddt")
        print(f"  r={r['recycling_steps']}, seed={r['seed']}, tc={r['tc_name']}: "
              f"wall={wall:.1f}s" if wall else "N/A",
              f"pred={pred:.1f}s" if pred else "",
              f"pLDDT={plddt:.4f}" if plddt else "",
              f"[{status}]")

    # Save raw results
    output_path = ORBIT_DIR / "raw_results.json"
    with output_path.open("w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\n[recycling-sweep] Raw results saved to {output_path}")

    # Aggregate by recycling_steps
    baseline = eval_config.get("baseline", {})
    baseline_mean_time = baseline.get("mean_wall_time_s", 53.567)
    baseline_mean_plddt = baseline.get("mean_plddt", 0.7170)
    baseline_by_name = {pc["name"]: pc for pc in baseline.get("per_complex", [])}

    print(f"\n{'='*80}")
    print("RECYCLING SWEEP RESULTS (eval-v5)")
    print(f"Baseline: mean_time={baseline_mean_time:.1f}s, mean_pLDDT={baseline_mean_plddt:.4f}")
    print(f"{'='*80}\n")

    summary = {}
    for r_steps in RECYCLING_STEPS_TO_SWEEP:
        runs = [r for r in raw_results if r["recycling_steps"] == r_steps]
        errors = [r for r in runs if r.get("error") is not None]
        if errors:
            print(f"\nrecycling_steps={r_steps}: {len(errors)} ERRORS")
            for e in errors:
                print(f"  seed={e['seed']}, tc={e['tc_name']}: {e['error'][:200]}")
            continue

        # Group by test case, aggregate across seeds
        tc_results = {}
        for tc in test_cases:
            tc_name = tc["name"]
            tc_runs = [r for r in runs if r["tc_name"] == tc_name]
            wall_times = [r["wall_time_s"] for r in tc_runs if r["wall_time_s"] is not None]
            predict_times = [r["predict_only_s"] for r in tc_runs if r.get("predict_only_s") is not None]
            plddts = [r["quality"]["complex_plddt"] for r in tc_runs
                      if "complex_plddt" in r.get("quality", {})]
            iptms = [r["quality"]["iptm"] for r in tc_runs
                     if "iptm" in r.get("quality", {})]

            tc_results[tc_name] = {
                "wall_times": wall_times,
                "predict_times": predict_times,
                "mean_wall_time": sum(wall_times) / len(wall_times) if wall_times else None,
                "mean_predict_time": sum(predict_times) / len(predict_times) if predict_times else None,
                "plddts": plddts,
                "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
                "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
            }

        # Aggregate across test cases
        all_mean_times = [v["mean_wall_time"] for v in tc_results.values() if v["mean_wall_time"]]
        all_mean_predict = [v["mean_predict_time"] for v in tc_results.values() if v["mean_predict_time"]]
        all_plddts = [v["mean_plddt"] for v in tc_results.values() if v["mean_plddt"]]

        grand_mean_time = sum(all_mean_times) / len(all_mean_times) if all_mean_times else None
        grand_mean_predict = sum(all_mean_predict) / len(all_mean_predict) if all_mean_predict else None
        grand_mean_plddt = sum(all_plddts) / len(all_plddts) if all_plddts else None

        speedup = baseline_mean_time / grand_mean_time if grand_mean_time else None
        plddt_delta = (grand_mean_plddt - baseline_mean_plddt) * 100 if grand_mean_plddt else None

        # Per-complex quality gate
        passes_quality = True
        if grand_mean_plddt is not None:
            regression = (baseline_mean_plddt - grand_mean_plddt) * 100
            if regression > 2.0:
                passes_quality = False
        for tc_name, tc_data in tc_results.items():
            bl = baseline_by_name.get(tc_name)
            if bl and tc_data["mean_plddt"] is not None:
                case_regression = (bl["complex_plddt"] - tc_data["mean_plddt"]) * 100
                if case_regression > 5.0:
                    passes_quality = False

        summary[r_steps] = {
            "mean_wall_time": grand_mean_time,
            "mean_predict_time": grand_mean_predict,
            "mean_plddt": grand_mean_plddt,
            "speedup": speedup,
            "plddt_delta_pp": plddt_delta,
            "passes_quality_gate": passes_quality,
            "per_complex": tc_results,
        }

        # Print table
        print(f"\n--- recycling_steps={r_steps} ---")
        print(f"  Speedup: {speedup:.2f}x | mean_time={grand_mean_time:.1f}s | "
              f"mean_predict={grand_mean_predict:.1f}s" if grand_mean_predict else "")
        print(f"  pLDDT: {grand_mean_plddt:.4f} (delta={plddt_delta:+.2f}pp) | "
              f"Quality gate: {'PASS' if passes_quality else 'FAIL'}")
        print(f"  {'Complex':<20} {'Wall(s)':>10} {'Pred(s)':>10} {'pLDDT':>10} {'iPTM':>10}")
        print(f"  {'-'*60}")
        for tc_name, tc_data in tc_results.items():
            mt = tc_data["mean_wall_time"]
            mp = tc_data["mean_predict_time"]
            p = tc_data["mean_plddt"]
            i = tc_data["mean_iptm"]
            bl = baseline_by_name.get(tc_name, {})
            bl_p = bl.get("complex_plddt")
            delta = f" ({(p - bl_p)*100:+.1f}pp)" if bl_p and p else ""
            print(f"  {tc_name:<20} {mt:>10.1f} {mp:>10.1f}" if mp else f"  {tc_name:<20} {mt:>10.1f} {'N/A':>10}",
                  f"{p:>10.4f}{delta}" if p else f"{'N/A':>10}",
                  f"{i:>10.4f}" if i else f"{'N/A':>10}")

    # Save summary
    summary_path = ORBIT_DIR / "sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[recycling-sweep] Summary saved to {summary_path}")

    # Final comparison table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'Recycling':<12} {'Time(s)':>10} {'Pred(s)':>10} {'pLDDT':>10} {'Delta':>10} {'Speedup':>10} {'Gate':>8}")
    print("-" * 70)
    for r_steps in RECYCLING_STEPS_TO_SWEEP:
        s = summary.get(r_steps)
        if s is None:
            print(f"  {r_steps:<12} {'ERROR':>10}")
            continue
        t = s["mean_wall_time"]
        po = s["mean_predict_time"]
        p = s["mean_plddt"]
        d = s["plddt_delta_pp"]
        sp = s["speedup"]
        g = "PASS" if s["passes_quality_gate"] else "FAIL"
        print(f"  {r_steps:<12} {t:>10.1f} {po:>10.1f}" if po else f"  {r_steps:<12} {t:>10.1f} {'N/A':>10}",
              f"{p:>10.4f} {d:>+10.2f}pp {sp:>10.2f}x {g:>8}")
