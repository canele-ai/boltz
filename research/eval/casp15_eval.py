"""CASP15 validation evaluation for Boltz-2 inference speedup.

Runs baseline and optimized configs on all CASP15 protein targets,
measuring pLDDT and CA RMSD against experimental ground truth.

Configs:
  Baseline:  SDE-200, recycling=3, fp32 (standard boltz_wrapper.py)
  Optimized: ODE-12, recycling=3, TF32, bf16, bypass Lightning (boltz_bypass_wrapper.py)

Usage:
    # Sanity check (1 target, 1 run)
    modal run research/eval/casp15_eval.py --mode sanity

    # Run baseline on all targets
    modal run research/eval/casp15_eval.py --mode baseline

    # Run optimized on all targets
    modal run research/eval/casp15_eval.py --mode optimized

    # Run both configs
    modal run research/eval/casp15_eval.py --mode both

    # Limit to N targets (sorted by size, smallest first)
    modal run research/eval/casp15_eval.py --mode both --max-targets 10
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

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
BYPASS_DIR = REPO_ROOT / "research" / "solutions" / "bypass-lightning"
STACKED_WRAPPER = REPO_ROOT / "research" / "solutions" / "eval-v2-winner" / "boltz_wrapper_stacked.py"

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install("torch==2.6.0", "numpy>=1.26,<2.0", "pyyaml==6.0.2")
    .pip_install("boltz==2.2.1")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .pip_install("biopython")
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_dir(str(EVAL_DIR / "casp15"), remote_path="/casp15")
    .add_local_file(
        str(BYPASS_DIR / "boltz_bypass_wrapper.py"),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
    .add_local_file(
        str(STACKED_WRAPPER),
        remote_path="/eval/boltz_wrapper_stacked.py",
    )
)

app = modal.App("boltz-casp15-eval", image=boltz_image)
msa_volume = modal.Volume.from_name("boltz-msa-cache-casp15", create_if_missing=True)
gt_volume = modal.Volume.from_name("boltz-gt-casp15", create_if_missing=True)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

BASELINE_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "seed": 42,
}

OPTIMIZED_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "recycling_steps": 3,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "cuda_warmup": True,
    "diffusion_samples": 1,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_casp15_targets() -> list[dict]:
    """Load CASP15 target config."""
    import yaml
    config_path = Path("/casp15/targets.yaml")
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return config["targets"]


def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path):
    """Inject cached MSA paths into input YAML. Returns modified YAML path or None."""
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


def _run_baseline(input_yaml: Path, out_dir: Path, config: dict) -> dict[str, Any]:
    """Run prediction using the standard Boltz wrapper (Lightning Trainer)."""
    wrapper = str(Path("/eval/boltz_wrapper.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--matmul_precision", str(config.get("matmul_precision", "highest")),
    ]

    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    return _run_subprocess(cmd)


def _run_optimized(input_yaml: Path, out_dir: Path, config: dict) -> dict[str, Any]:
    """Run prediction using the bypass-lightning wrapper."""
    wrapper = str(Path("/eval/boltz_bypass_wrapper.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 12)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--gamma_0", str(config.get("gamma_0", 0.0)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
        "--matmul_precision", str(config.get("matmul_precision", "high")),
    ]

    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    if config.get("bf16_trunk", True):
        cmd.append("--bf16_trunk")
    if config.get("cuda_warmup", True):
        cmd.append("--cuda_warmup")

    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    return _run_subprocess(cmd)


def _run_subprocess(cmd: list[str]) -> dict[str, Any]:
    """Execute a Boltz prediction subprocess and collect timing + phase timestamps."""
    result: dict[str, Any] = {
        "wall_time_s": None,
        "predict_only_s": None,
        "quality": {},
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"Exit code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse phase timestamps
        if proc.stderr:
            predict_start = predict_end = None
            for line in proc.stderr.split("\n"):
                if "[PHASE] predict_start=" in line:
                    predict_start = float(line.split("=")[1])
                elif "[PHASE] predict_end=" in line:
                    predict_end = float(line.split("=")[1])
            if predict_start and predict_end:
                result["predict_only_s"] = predict_end - predict_start

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 3600s"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    """Parse Boltz confidence metrics from prediction output."""
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Prediction dir not found: {results_dir}"}

    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON files found"}

    with confidence_files[0].open() as f:
        conf = json.load(f)

    quality = {}
    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]
    return quality


def _compare_structures(
    pred_dir: Path,
    input_yaml: Path,
    gt_path: Path,
    chain_mapping: dict[str, str],
) -> dict[str, Any]:
    """Compare predicted structure against PDB ground truth via CA RMSD."""
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
        from Bio.SVDSuperimposer import SVDSuperimposer
        import numpy as np

        parser = MMCIFParser(QUIET=True)

        # Find predicted mmCIF
        target_name = input_yaml.stem
        pred_base = pred_dir / f"boltz_results_{target_name}" / "predictions"
        pred_cif = None
        if pred_base.exists():
            for d in sorted(pred_base.iterdir()):
                if d.is_dir():
                    cifs = sorted(d.glob("*_model_0.cif"))
                    if cifs:
                        pred_cif = cifs[0]
                        break

        if pred_cif is None:
            return {"error": f"No predicted mmCIF found in {pred_base}"}

        gt_structure = parser.get_structure("gt", str(gt_path))
        pred_structure = parser.get_structure("pred", str(pred_cif))

        # Extract CA atoms from ground truth
        gt_ca = {}
        for chain in gt_structure[0]:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    gt_ca[(chain.id, residue.id[1])] = residue["CA"].get_vector().get_array()

        # Extract CA atoms from prediction with chain mapping
        pred_ca = {}
        for chain in pred_structure[0]:
            gt_chain_id = chain_mapping.get(chain.id, chain.id)
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    pred_ca[(gt_chain_id, residue.id[1])] = residue["CA"].get_vector().get_array()

        # Match CA pairs
        matched_gt = []
        matched_pred = []
        for key in sorted(gt_ca.keys()):
            if key in pred_ca:
                matched_gt.append(gt_ca[key])
                matched_pred.append(pred_ca[key])

        if len(matched_gt) < 10:
            return {"error": f"Too few matched CA atoms: {len(matched_gt)}",
                    "matched_residues": len(matched_gt)}

        gt_coords = np.array(matched_gt)
        pred_coords = np.array(matched_pred)

        # Superimpose and compute RMSD
        sup = SVDSuperimposer()
        sup.set(gt_coords, pred_coords)
        sup.run()
        ca_rmsd = sup.get_rms()

        # Per-residue deviations
        rotated = sup.get_rotran()
        pred_aligned = np.dot(pred_coords, rotated[0]) + rotated[1]
        per_residue_dev = np.sqrt(np.sum((gt_coords - pred_aligned) ** 2, axis=1))

        return {
            "ca_rmsd": round(float(ca_rmsd), 3),
            "matched_residues": len(matched_gt),
            "mean_per_residue_dev": round(float(np.mean(per_residue_dev)), 3),
            "max_per_residue_dev": round(float(np.max(per_residue_dev)), 3),
            "pct_within_2A": round(float(np.mean(per_residue_dev < 2.0) * 100), 1),
        }

    except Exception as e:
        return {"error": f"Structural comparison failed: {e}"}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=21600,  # 6 hours
    volumes={"/msa_cache": msa_volume, "/ground_truth": gt_volume},
)
def run_evaluation(
    config_name: str,
    config_json: str,
    use_bypass: bool = False,
    max_targets: int = 0,
    sanity: bool = False,
) -> str:
    """Run a single config on all CASP15 targets.

    Returns JSON with per-target results including pLDDT + CA RMSD.
    """
    import yaml

    config = json.loads(config_json)
    targets = _load_casp15_targets()

    if max_targets > 0:
        targets = targets[:max_targets]
    if sanity:
        targets = targets[:1]
        config["sampling_steps"] = min(config.get("sampling_steps", 200), 10)

    # Upload ground truth to volume if not already there
    gt_local = Path("/casp15/ground_truth")
    gt_vol = Path("/ground_truth")
    if gt_local.exists():
        for cif in gt_local.glob("*.cif"):
            dest = gt_vol / cif.name
            if not dest.exists():
                shutil.copy2(cif, dest)
        gt_volume.commit()

    # Check MSA cache
    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())
    print(f"[casp15] Config: {config_name}")
    print(f"[casp15] Targets: {len(targets)}, MSA cache: {use_msa_cache}, bypass: {use_bypass}")

    run_fn = _run_optimized if use_bypass else _run_baseline

    results: dict[str, Any] = {
        "config_name": config_name,
        "config": config,
        "use_bypass": use_bypass,
        "msa_cached": use_msa_cache,
        "num_targets": len(targets),
        "per_target": [],
    }

    for i, target in enumerate(targets):
        tc_name = target["name"]
        tc_yaml = Path(f"/casp15/{target['yaml']}")
        gt_path = gt_vol / Path(target["ground_truth"]).name
        chain_mapping = target.get("chain_mapping", {})

        print(f"\n[{i+1}/{len(targets)}] {tc_name} (PDB {target['pdb_id']}, "
              f"{target['total_residues']} res)")

        if not tc_yaml.exists():
            results["per_target"].append({
                "name": tc_name, "pdb_id": target["pdb_id"],
                "error": f"YAML not found: {tc_yaml}",
            })
            continue

        work_dir = Path(f"/tmp/casp15_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Inject cached MSAs
        run_config = dict(config)
        effective_yaml = tc_yaml
        if use_msa_cache:
            cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
            if cached_yaml is not None:
                effective_yaml = cached_yaml
                run_config["_msa_cached"] = True

        # Run prediction
        pred_result = run_fn(effective_yaml, work_dir, run_config)

        if pred_result["error"] is not None:
            print(f"  ERROR: {pred_result['error'][:200]}")
            results["per_target"].append({
                "name": tc_name,
                "pdb_id": target["pdb_id"],
                "total_residues": target["total_residues"],
                "wall_time_s": pred_result["wall_time_s"],
                "predict_only_s": pred_result.get("predict_only_s"),
                "error": pred_result["error"][:500],
            })
            continue

        # Parse quality metrics
        quality = _parse_confidence(work_dir, effective_yaml)

        # Compute structural comparison
        structural = {}
        if gt_path.exists():
            structural = _compare_structures(work_dir, effective_yaml, gt_path, chain_mapping)
        else:
            structural = {"error": f"Ground truth not found: {gt_path}"}

        plddt = quality.get("complex_plddt", "N/A")
        rmsd = structural.get("ca_rmsd", "N/A")
        wall_t = pred_result["wall_time_s"]
        pred_t = pred_result.get("predict_only_s")
        time_str = f"{pred_t:.1f}s (predict)" if pred_t else f"{wall_t:.1f}s (wall)"

        print(f"  {time_str} | pLDDT={plddt} | CA RMSD={rmsd}Å"
              f"{' | ' + structural.get('error', '') if structural.get('error') else ''}")

        entry = {
            "name": tc_name,
            "pdb_id": target["pdb_id"],
            "total_residues": target["total_residues"],
            "num_chains": target["num_chains"],
            "wall_time_s": wall_t,
            "predict_only_s": pred_t,
            "quality": quality,
            "structural": structural,
            "error": None,
        }
        results["per_target"].append(entry)

    # Compute aggregates
    successful = [r for r in results["per_target"] if r.get("error") is None]
    results["aggregate"] = _compute_aggregates(successful)

    return json.dumps(results, indent=2)


def _compute_aggregates(successful: list[dict]) -> dict[str, Any]:
    """Compute aggregate metrics from per-target results."""
    if not successful:
        return {"error": "No successful targets"}

    wall_times = [r["wall_time_s"] for r in successful if r.get("wall_time_s")]
    predict_times = [r["predict_only_s"] for r in successful if r.get("predict_only_s")]
    plddts = [
        r["quality"]["complex_plddt"] for r in successful
        if r.get("quality", {}).get("complex_plddt") is not None
        and isinstance(r["quality"]["complex_plddt"], (int, float))
        and not math.isnan(r["quality"]["complex_plddt"])
    ]
    rmsds = [
        r["structural"]["ca_rmsd"] for r in successful
        if r.get("structural", {}).get("ca_rmsd") is not None
    ]

    import statistics

    agg: dict[str, Any] = {
        "num_successful": len(successful),
        "num_with_plddt": len(plddts),
        "num_with_rmsd": len(rmsds),
    }

    if wall_times:
        agg["mean_wall_time_s"] = round(sum(wall_times) / len(wall_times), 2)
        agg["median_wall_time_s"] = round(statistics.median(wall_times), 2)
        agg["total_wall_time_s"] = round(sum(wall_times), 2)

    if predict_times:
        agg["mean_predict_only_s"] = round(sum(predict_times) / len(predict_times), 2)
        agg["median_predict_only_s"] = round(statistics.median(predict_times), 2)

    if plddts:
        agg["mean_plddt"] = round(sum(plddts) / len(plddts), 4)
        agg["median_plddt"] = round(statistics.median(plddts), 4)
        agg["min_plddt"] = round(min(plddts), 4)
        agg["max_plddt"] = round(max(plddts), 4)

    if rmsds:
        agg["mean_ca_rmsd"] = round(sum(rmsds) / len(rmsds), 3)
        agg["median_ca_rmsd"] = round(statistics.median(rmsds), 3)
        agg["min_ca_rmsd"] = round(min(rmsds), 3)
        agg["max_ca_rmsd"] = round(max(rmsds), 3)

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "both",
    max_targets: int = 0,
):
    """CASP15 validation evaluation.

    Modes:
        sanity     - 1 target, 1 run per config (verify everything works)
        baseline   - Run baseline (SDE-200, recycle=3, fp32) on all targets
        optimized  - Run optimized (ODE-12, recycle=3, TF32+bf16+bypass) on all targets
        both       - Run baseline then optimized

    Usage:
        modal run research/eval/casp15_eval.py --mode sanity
        modal run research/eval/casp15_eval.py --mode both
        modal run research/eval/casp15_eval.py --mode both --max-targets 10
    """
    sanity = mode == "sanity"
    results_dir = EVAL_DIR / "casp15" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if mode in ("baseline", "both", "sanity"):
        print(f"[casp15] Running BASELINE (SDE-200, recycle=3, fp32)...")
        baseline_json = run_evaluation.remote(
            config_name="baseline",
            config_json=json.dumps(BASELINE_CONFIG),
            use_bypass=False,
            max_targets=max_targets,
            sanity=sanity,
        )
        baseline = json.loads(baseline_json)

        out_path = results_dir / "baseline.json"
        with out_path.open("w") as f:
            json.dump(baseline, f, indent=2)
        print(f"\n[casp15] Baseline results saved to {out_path}")
        _print_summary("Baseline", baseline)

    if mode in ("optimized", "both", "sanity"):
        print(f"\n[casp15] Running OPTIMIZED (ODE-12, recycle=3, TF32+bf16+bypass)...")
        optimized_json = run_evaluation.remote(
            config_name="optimized",
            config_json=json.dumps(OPTIMIZED_CONFIG),
            use_bypass=True,
            max_targets=max_targets,
            sanity=sanity,
        )
        optimized = json.loads(optimized_json)

        out_path = results_dir / "optimized.json"
        with out_path.open("w") as f:
            json.dump(optimized, f, indent=2)
        print(f"\n[casp15] Optimized results saved to {out_path}")
        _print_summary("Optimized", optimized)

    if mode in ("both", "sanity"):
        _print_comparison(baseline, optimized)


def _print_summary(label: str, results: dict):
    """Print aggregate summary for one config."""
    agg = results.get("aggregate", {})
    n = agg.get("num_successful", 0)
    total = results.get("num_targets", 0)

    print(f"\n{'='*60}")
    print(f"  {label} Summary ({n}/{total} targets successful)")
    print(f"{'='*60}")

    if agg.get("mean_wall_time_s"):
        print(f"  Wall time:   mean={agg['mean_wall_time_s']:.1f}s, "
              f"median={agg.get('median_wall_time_s', 'N/A')}s, "
              f"total={agg.get('total_wall_time_s', 'N/A')}s")
    if agg.get("mean_predict_only_s"):
        print(f"  Predict-only: mean={agg['mean_predict_only_s']:.1f}s, "
              f"median={agg.get('median_predict_only_s', 'N/A')}s")
    if agg.get("mean_plddt"):
        print(f"  pLDDT:       mean={agg['mean_plddt']:.4f}, "
              f"median={agg.get('median_plddt', 'N/A')}, "
              f"range=[{agg.get('min_plddt', 'N/A')}, {agg.get('max_plddt', 'N/A')}]")
    if agg.get("mean_ca_rmsd"):
        print(f"  CA RMSD:     mean={agg['mean_ca_rmsd']:.3f}Å, "
              f"median={agg.get('median_ca_rmsd', 'N/A')}Å, "
              f"range=[{agg.get('min_ca_rmsd', 'N/A')}, {agg.get('max_ca_rmsd', 'N/A')}]Å")

    # Show errors
    errors = [r for r in results.get("per_target", []) if r.get("error")]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"    {e['name']}: {e['error'][:100]}")


def _print_comparison(baseline: dict, optimized: dict):
    """Print side-by-side comparison of baseline vs optimized."""
    b_agg = baseline.get("aggregate", {})
    o_agg = optimized.get("aggregate", {})

    print(f"\n{'='*60}")
    print(f"  CASP15 Validation: Baseline vs Optimized")
    print(f"{'='*60}")

    # Speedup
    b_time = b_agg.get("mean_wall_time_s")
    o_time = o_agg.get("mean_wall_time_s")
    o_pred = o_agg.get("mean_predict_only_s")
    if b_time and o_time:
        print(f"  Wall time speedup:    {b_time / o_time:.2f}x "
              f"({b_time:.1f}s → {o_time:.1f}s)")
    if b_time and o_pred:
        print(f"  Predict-only speedup: {b_time / o_pred:.2f}x "
              f"({b_time:.1f}s → {o_pred:.1f}s)")

    # Quality comparison
    b_plddt = b_agg.get("mean_plddt")
    o_plddt = o_agg.get("mean_plddt")
    if b_plddt and o_plddt:
        delta = (o_plddt - b_plddt) * 100
        print(f"  pLDDT delta:          {delta:+.2f}pp "
              f"({b_plddt:.4f} → {o_plddt:.4f})")

    b_rmsd = b_agg.get("mean_ca_rmsd")
    o_rmsd = o_agg.get("mean_ca_rmsd")
    if b_rmsd and o_rmsd:
        delta = o_rmsd - b_rmsd
        print(f"  CA RMSD delta:        {delta:+.3f}Å "
              f"({b_rmsd:.3f}Å → {o_rmsd:.3f}Å)")

    # Per-target comparison table
    b_targets = {r["name"]: r for r in baseline.get("per_target", []) if not r.get("error")}
    o_targets = {r["name"]: r for r in optimized.get("per_target", []) if not r.get("error")}

    common = sorted(set(b_targets.keys()) & set(o_targets.keys()))
    if common:
        print(f"\n  Per-target comparison ({len(common)} targets):")
        print(f"  {'Target':<10} {'Res':>5} {'B wall':>7} {'O wall':>7} "
              f"{'B pLDDT':>8} {'O pLDDT':>8} {'B RMSD':>7} {'O RMSD':>7}")
        print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")

        for name in common:
            b = b_targets[name]
            o = o_targets[name]
            b_t = f"{b['wall_time_s']:.1f}" if b.get("wall_time_s") else "N/A"
            o_t = f"{o['wall_time_s']:.1f}" if o.get("wall_time_s") else "N/A"
            b_p = f"{b['quality'].get('complex_plddt', 0):.4f}" if b.get("quality", {}).get("complex_plddt") else "N/A"
            o_p = f"{o['quality'].get('complex_plddt', 0):.4f}" if o.get("quality", {}).get("complex_plddt") else "N/A"
            b_r = f"{b.get('structural', {}).get('ca_rmsd', 'N/A')}"
            o_r = f"{o.get('structural', {}).get('ca_rmsd', 'N/A')}"
            res = b.get("total_residues", "?")
            print(f"  {name:<10} {res:>5} {b_t:>7} {o_t:>7} {b_p:>8} {o_p:>8} {b_r:>7} {o_r:>7}")
