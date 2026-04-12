"""Head pruning experiment for Boltz-2 Pairformer sequence attention.

Hypothesis: Many of the 1024 attention heads (64 blocks x 16 heads) in the
Pairformer's sequence attention are redundant. Pruning the least important
ones by zeroing their weights can reduce effective compute without
meaningful quality loss.

Architecture:
- 64 PairformerLayer blocks, each with AttentionPairBias (16 heads, head_dim=24)
- Head importance = L2_norm(proj_o cols) * L2_norm(proj_v rows) * L2_norm(proj_g rows)
- Pruning = zero out Q/K/V/G/O weights for targeted heads

Note on speedup: Weight-zeroing does NOT reduce matmul sizes — the tensor shapes
are unchanged, so GPU time is nearly identical. A real speedup requires either:
(a) actually removing heads (reshape tensors), or (b) sparse matmul support.
This experiment measures QUALITY IMPACT ONLY. If quality holds, a follow-up
can implement structural head removal.

Usage:
    modal run orbits/head-prune/eval_head_prune.py --mode sanity
    modal run orbits/head-prune/eval_head_prune.py --mode sweep
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
        str(ORBIT_DIR / "boltz_wrapper_headprune.py"),
        remote_path="/eval/boltz_wrapper_headprune.py",
    )
)

app = modal.App("boltz-head-prune", image=boltz_image)
msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_FLAGS = {
    "sampling_steps": 20,
    "recycling_steps": 0,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "seed": 42,
    "diffusion_samples": 1,
}

PRUNE_FRACTIONS = [0.0, 0.25, 0.50, 0.75]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    with Path("/eval/config.yaml").open() as f:
        return yaml.safe_load(f)


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

    return cached_yaml


def _run_prediction(
    input_yaml: Path,
    out_dir: Path,
    prune_fraction: float,
    msa_directory: str | None = None,
) -> dict[str, Any]:
    """Run a single prediction via subprocess with head pruning."""
    wrapper = "/eval/boltz_wrapper_headprune.py"
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(BASE_FLAGS["sampling_steps"]),
        "--recycling_steps", str(BASE_FLAGS["recycling_steps"]),
        "--diffusion_samples", str(BASE_FLAGS["diffusion_samples"]),
        "--override",
        "--gamma_0", str(BASE_FLAGS["gamma_0"]),
        "--noise_scale", str(BASE_FLAGS["noise_scale"]),
        "--matmul_precision", BASE_FLAGS["matmul_precision"],
        "--seed", str(BASE_FLAGS["seed"]),
    ]

    if BASE_FLAGS["bf16_trunk"]:
        cmd.append("--bf16_trunk")
    if BASE_FLAGS["enable_kernels"]:
        cmd.append("--enable_kernels")

    cmd.extend(["--prune_fraction", str(prune_fraction)])

    # MSA handling
    if msa_directory:
        cmd.extend(["--msa_directory", msa_directory])
    else:
        cmd.append("--use_msa_server")

    result: dict[str, Any] = {
        "wall_time_s": None,
        "predict_only_s": None,
        "quality": {},
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
        )
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

        if predict_start and predict_end:
            result["predict_only_s"] = predict_end - predict_start

        if proc.returncode != 0:
            result["error"] = (
                f"Exit code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse quality
        quality = _parse_confidence(out_dir, input_yaml)
        result["quality"] = quality

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout 1800s"
    except Exception as exc:
        result["error"] = f"Error: {exc}"

    return result


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    target_name = input_yaml.stem
    # Handle _cached suffix
    if target_name.endswith("_cached"):
        target_name = target_name[:-7]

    results_dir = out_dir / f"boltz_results_{input_yaml.stem}" / "predictions" / target_name
    quality: dict[str, Any] = {}

    if not results_dir.exists():
        # Try with the original stem
        results_dir = out_dir / f"boltz_results_{input_yaml.stem}" / "predictions"
        if results_dir.exists():
            subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]
            else:
                return {"error": f"No prediction subdirs in {results_dir}"}
        else:
            # Try original name without _cached
            orig_name = input_yaml.stem
            if orig_name.endswith("_cached"):
                orig_name = orig_name[:-7]
            alt_dir = out_dir / f"boltz_results_{orig_name}" / "predictions" / orig_name
            if alt_dir.exists():
                results_dir = alt_dir
            else:
                return {"error": f"Prediction dir not found"}

    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON found"}

    with confidence_files[0].open() as f:
        conf = json.load(f)

    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]

    return quality


def _compute_aggregates(per_complex, eval_config):
    """Compute aggregate metrics."""
    baseline = eval_config.get("baseline", {})
    successful = [r for r in per_complex if r["error"] is None and r["wall_time_s"] is not None]

    if not successful:
        return {"error": "No successful test cases"}

    total_time = sum(r["wall_time_s"] for r in successful)
    mean_time = total_time / len(successful)

    plddts = [
        r["quality"]["complex_plddt"] for r in successful
        if "complex_plddt" in r["quality"]
        and r["quality"]["complex_plddt"] is not None
        and isinstance(r["quality"]["complex_plddt"], (int, float))
    ]
    mean_plddt = sum(plddts) / len(plddts) if plddts else None

    predict_times = [r["predict_only_s"] for r in successful if r.get("predict_only_s")]
    mean_predict = sum(predict_times) / len(predict_times) if predict_times else None

    agg: dict[str, Any] = {
        "num_successful": len(successful),
        "mean_wall_time_s": mean_time,
        "mean_predict_only_s": mean_predict,
        "mean_plddt": mean_plddt,
    }

    if baseline:
        bl_time = baseline.get("mean_wall_time_s")
        bl_plddt = baseline.get("mean_plddt")
        if bl_time and mean_time > 0:
            agg["speedup_vs_baseline"] = bl_time / mean_time
        if bl_plddt and mean_plddt:
            agg["plddt_delta_pp"] = (mean_plddt - bl_plddt) * 100.0
            regression = (bl_plddt - mean_plddt) * 100.0
            agg["passes_quality_gate"] = regression <= 2.0

            # Per-complex check
            if baseline.get("per_complex"):
                bl_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                violations = {}
                for r in successful:
                    bl_case = bl_by_name.get(r["name"])
                    if bl_case and bl_case.get("complex_plddt") is not None:
                        case_plddt = r["quality"].get("complex_plddt")
                        if case_plddt is not None:
                            case_reg = (bl_case["complex_plddt"] - case_plddt) * 100.0
                            if case_reg > 5.0:
                                agg["passes_quality_gate"] = False
                                violations[r["name"]] = f"-{case_reg:.1f}pp"
                if violations:
                    agg["per_complex_violations"] = violations

    return agg


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=7200, volumes={"/msa_cache": msa_volume})
def run_sweep() -> str:
    """Sweep head pruning fractions: 0%, 25%, 50%, 75%."""
    import torch

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    msa_cache_root = Path("/msa_cache")

    results: dict[str, Any] = {
        "experiment": "head-prune",
        "base_flags": BASE_FLAGS,
        "prune_fractions": PRUNE_FRACTIONS,
        "env": {
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "configs": {},
    }

    for frac in PRUNE_FRACTIONS:
        label = f"prune_{int(frac * 100)}pct"
        n_heads = int(1024 * frac)
        print(f"\n{'='*60}")
        print(f"[head-prune] {label}: zeroing {n_heads}/1024 heads")
        print(f"{'='*60}")

        per_complex = []
        for tc in test_cases:
            tc_name = tc["name"]
            tc_yaml = Path("/eval") / tc["yaml"]

            if not tc_yaml.exists():
                per_complex.append({
                    "name": tc_name, "error": f"YAML not found",
                    "wall_time_s": None, "quality": {},
                })
                continue

            work_dir = Path(f"/tmp/boltz_hp/{label}_{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Inject cached MSAs
            effective_yaml = tc_yaml
            if msa_cache_root.exists():
                cached = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
                if cached:
                    effective_yaml = cached

            print(f"[head-prune] Running {tc_name}...")
            pred = _run_prediction(effective_yaml, work_dir, frac)

            entry = {
                "name": tc_name,
                "wall_time_s": pred["wall_time_s"],
                "predict_only_s": pred.get("predict_only_s"),
                "quality": pred["quality"],
                "error": pred["error"],
            }
            per_complex.append(entry)

            if pred["error"]:
                print(f"[head-prune] ERROR {tc_name}: {pred['error'][:300]}")
            else:
                plddt = pred["quality"].get("complex_plddt", "N/A")
                t = pred["wall_time_s"]
                pt = pred.get("predict_only_s")
                print(f"[head-prune] {tc_name}: wall={t:.1f}s, "
                      f"predict={pt:.1f}s, " if pt else f"[head-prune] {tc_name}: wall={t:.1f}s, " +
                      f"pLDDT={plddt}")

        agg = _compute_aggregates(per_complex, eval_config)
        results["configs"][label] = {
            "prune_fraction": frac,
            "n_heads_pruned": n_heads,
            "aggregate": agg,
            "per_complex": per_complex,
        }

        print(f"[head-prune] {label} => mean_time={agg.get('mean_wall_time_s', 'ERR')}, "
              f"mean_plddt={agg.get('mean_plddt', 'ERR')}, "
              f"speedup={agg.get('speedup_vs_baseline', 'N/A')}")

    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=600, volumes={"/msa_cache": msa_volume})
def run_sanity() -> str:
    """Quick sanity: run 0% prune on small_complex only."""
    import torch

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    msa_cache_root = Path("/msa_cache")

    if not test_cases:
        return json.dumps({"error": "No test cases"})

    tc = test_cases[0]
    tc_yaml = Path("/eval") / tc["yaml"]
    work_dir = Path(f"/tmp/boltz_hp_sanity_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    effective_yaml = tc_yaml
    if msa_cache_root.exists():
        cached = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
        if cached:
            effective_yaml = cached

    print(f"[sanity] Running {tc['name']} with 0% pruning...")
    pred = _run_prediction(effective_yaml, work_dir, 0.0)

    result = {
        "env": {
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "test": tc["name"],
        "wall_time_s": pred["wall_time_s"],
        "predict_only_s": pred.get("predict_only_s"),
        "quality": pred["quality"],
        "error": pred["error"],
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(mode: str = "sweep"):
    """Head pruning experiment.

    Modes:
        sanity - Quick check with 0% prune on small_complex
        sweep  - Full sweep: 0/25/50/75% pruning on all complexes
    """
    if mode == "sanity":
        print("[head-prune] Sanity check...")
        result = run_sanity.remote()
        print(result)

    elif mode == "sweep":
        print("[head-prune] Running full sweep...")
        result_json = run_sweep.remote()
        parsed = json.loads(result_json)
        print(json.dumps(parsed, indent=2))

        # Summary table
        print(f"\n{'='*75}")
        print("HEAD PRUNING SWEEP RESULTS")
        print(f"{'='*75}")
        print(f"{'Config':<18} {'Pruned':>10} {'Wall(s)':>8} {'Pred(s)':>8} "
              f"{'pLDDT':>8} {'Delta':>8} {'Spdup':>7} {'Gate':>6}")
        print("-" * 75)
        for label, data in parsed.get("configs", {}).items():
            agg = data.get("aggregate", {})
            n = data.get("n_heads_pruned", 0)
            wt = agg.get("mean_wall_time_s")
            pt = agg.get("mean_predict_only_s")
            p = agg.get("mean_plddt")
            d = agg.get("plddt_delta_pp")
            s = agg.get("speedup_vs_baseline")
            g = agg.get("passes_quality_gate")
            print(f"{label:<18} {n:>5}/1024 "
                  f"{wt:>8.1f} " if wt else f"{'ERR':>8} " +
                  f"{pt:>8.1f} " if pt else f"{'N/A':>8} " +
                  f"{p:.4f} " if p else f"{'ERR':>8} " +
                  f"{d:+.2f}pp " if d is not None else f"{'N/A':>8} " +
                  f"{s:.2f}x " if s else f"{'ERR':>7} " +
                  f"{'PASS' if g else 'FAIL':>6}" if g is not None else f"{'N/A':>6}")

    else:
        print(f"Unknown mode: {mode}. Use: sanity, sweep")
