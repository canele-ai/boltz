"""Multi-seed evaluation for profile-and-fuse orbit.

Runs the full-stack config (ODE + bf16 + TF32fix + SDPA + bf16 OPM)
across 3 independent Modal containers (seeds 42, 123, 7) and reports
mean +/- std. This matches the orbit protocol requirement of 3+ seeds.

Usage:
    modal run orbits/profile-and-fuse/eval_multi_seed.py
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
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_wrapper_sdpa.py"),
        remote_path="/eval/boltz_wrapper_sdpa.py",
    )
)

app = modal.App("boltz-eval-multiseed", image=boltz_image)

# Full-stack config
FULL_STACK_CONFIG = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "bf16_opm": True,
    "sdpa_attention": True,
    "compile_score": False,
    "enable_kernels": True,
    "diffusion_samples": 1,
}

SEEDS = [42, 123, 7]


def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _run_boltz_prediction(input_yaml, out_dir, config):
    wrapper = str(Path("/eval/boltz_wrapper_sdpa.py"))
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
    if config.get("bf16_opm", False):
        cmd.append("--bf16_opm")
    if config.get("sdpa_attention", False):
        cmd.append("--sdpa_attention")
    if config.get("compile_score", False):
        cmd.append("--compile_score")

    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    result = {"wall_time_s": None, "quality": {}, "error": None}

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"Exit code {proc.returncode}.\n"
                f"STDOUT: {proc.stdout[-2000:] if proc.stdout else '(empty)'}\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        target_name = input_yaml.stem
        results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name
        if not results_dir.exists():
            pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
            if pred_base.exists():
                subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
                if subdirs:
                    results_dir = subdirs[0]

        if results_dir.exists():
            confidence_files = sorted(results_dir.glob("confidence_*.json"))
            if confidence_files:
                with confidence_files[0].open() as f:
                    conf = json.load(f)
                for key in ["confidence_score", "ptm", "iptm", "ligand_iptm",
                            "protein_iptm", "complex_plddt", "complex_iplddt",
                            "complex_pde", "complex_ipde"]:
                    if key in conf:
                        result["quality"][key] = conf[key]

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


@app.function(gpu="L40S", timeout=7200)
def evaluate_seed(seed: int) -> str:
    """Run full-stack eval for a single seed (one independent container)."""
    config = {**FULL_STACK_CONFIG, "seed": seed}
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results = {
        "seed": seed,
        "config": config,
        "per_complex": [],
    }

    import torch
    results["env"] = {
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    for tc in test_cases:
        tc_yaml = Path("/eval") / tc["yaml"]
        work_dir = Path(f"/tmp/boltz_eval/{tc['name']}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[seed={seed}] Running {tc['name']}...")
        pred_result = _run_boltz_prediction(tc_yaml, work_dir, config)

        if pred_result["error"]:
            print(f"[seed={seed}] ERROR on {tc['name']}: {pred_result['error'][:200]}")
            results["per_complex"].append({
                "name": tc["name"],
                "wall_time_s": None,
                "quality": {},
                "error": pred_result["error"],
            })
        else:
            t = pred_result["wall_time_s"]
            p = pred_result["quality"].get("complex_plddt", "N/A")
            print(f"[seed={seed}] {tc['name']}: {t:.1f}s, pLDDT={p}")
            results["per_complex"].append({
                "name": tc["name"],
                "wall_time_s": t,
                "quality": pred_result["quality"],
                "error": None,
            })

    # Compute per-seed aggregate
    successful = [r for r in results["per_complex"]
                  if r["error"] is None and r["wall_time_s"] is not None]
    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [r["quality"]["complex_plddt"] for r in successful
                   if "complex_plddt" in r["quality"]]
        results["aggregate"] = {
            "mean_wall_time_s": mean_time,
            "total_wall_time_s": total_time,
            "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
        }
    else:
        results["aggregate"] = {"error": "No successful complexes"}

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    """Run 3 seeds in parallel and report aggregate."""
    import statistics

    print(f"[multi-seed] Running seeds {SEEDS} in parallel...")

    all_results = {}
    for seed, result_json in zip(SEEDS, evaluate_seed.map(SEEDS)):
        result = json.loads(result_json)
        all_results[seed] = result
        agg = result.get("aggregate", {})
        mean_time = agg.get("mean_wall_time_s")
        mean_plddt = agg.get("mean_plddt")
        print(f"\n[seed={seed}] Mean time: {mean_time:.1f}s, pLDDT: {mean_plddt:.4f}" if mean_time else f"\n[seed={seed}] FAILED")

        for pc in result.get("per_complex", []):
            if pc.get("error"):
                print(f"  {pc['name']}: ERROR")
            else:
                t = pc.get("wall_time_s", 0)
                p = pc.get("quality", {}).get("complex_plddt", "N/A")
                print(f"  {pc['name']}: {t:.1f}s, pLDDT={p}")

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("MULTI-SEED AGGREGATE")
    print(f"{'='*60}")

    eval_config_yaml = None
    try:
        import yaml
        config_path = Path(__file__).resolve().parent.parent.parent / "research" / "eval" / "config.yaml"
        with config_path.open() as f:
            eval_config_yaml = yaml.safe_load(f)
    except Exception:
        pass

    baseline_time = 53.56664509866667  # from config.yaml
    baseline_plddt = 0.7169803380966187

    seed_times = []
    seed_plddts = []
    seed_speedups = []

    for seed in SEEDS:
        result = all_results[seed]
        agg = result.get("aggregate", {})
        t = agg.get("mean_wall_time_s")
        p = agg.get("mean_plddt")
        if t and p:
            seed_times.append(t)
            seed_plddts.append(p)
            seed_speedups.append(baseline_time / t)

    if seed_times:
        mean_time = statistics.mean(seed_times)
        std_time = statistics.stdev(seed_times) if len(seed_times) > 1 else 0
        mean_plddt = statistics.mean(seed_plddts)
        std_plddt = statistics.stdev(seed_plddts) if len(seed_plddts) > 1 else 0
        mean_speedup = statistics.mean(seed_speedups)
        std_speedup = statistics.stdev(seed_speedups) if len(seed_speedups) > 1 else 0
        plddt_delta = (mean_plddt - baseline_plddt) * 100

        print(f"\nBaseline: {baseline_time:.1f}s, pLDDT={baseline_plddt:.4f}")
        print(f"\n| Seed | Mean Time | pLDDT | Speedup |")
        print(f"|------|-----------|-------|---------|")
        for i, seed in enumerate(SEEDS):
            if i < len(seed_times):
                print(f"| {seed} | {seed_times[i]:.1f}s | {seed_plddts[i]:.4f} | {seed_speedups[i]:.2f}x |")
        print(f"| **Mean** | **{mean_time:.1f} +/- {std_time:.1f}s** | **{mean_plddt:.4f} +/- {std_plddt:.4f}** | **{mean_speedup:.2f}x +/- {std_speedup:.2f}** |")
        print(f"\npLDDT delta: {plddt_delta:+.2f}pp")
        print(f"Quality gate: {'PASS' if plddt_delta >= -2.0 else 'FAIL'}")

    print("\n--- FULL RESULTS ---")
    print(json.dumps(all_results, indent=2))
