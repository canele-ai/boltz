"""Structural validation: compare baseline vs optimized predicted structures.

Runs both baseline (200-step, fp32) and optimized (ODE-12, TF32+bf16, bypass+warmup)
predictions on the test set, then computes all-atom RMSD after optimal superposition
to verify that optimized predictions are structurally similar to baseline.

This catches the case where pLDDT is preserved but the actual structure diverges.

Usage:
    modal run orbits/structural-validation/validate_structures.py
    modal run orbits/structural-validation/validate_structures.py --sanity-check
"""

from __future__ import annotations

import json
import math
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
BYPASS_WRAPPER = REPO_ROOT / "research" / "solutions" / "bypass-lightning" / "boltz_bypass_wrapper.py"
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
    .add_local_file(str(BYPASS_WRAPPER), remote_path="/eval/boltz_bypass_wrapper.py")
    .add_local_file(str(STACKED_WRAPPER), remote_path="/eval/boltz_wrapper_stacked.py")
)

app = modal.App("boltz-structural-validation", image=boltz_image)

msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

BASELINE_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "seed": 42,
}

OPTIMIZED_CONFIG: dict[str, Any] = {
    "sampling_steps": 12,
    "recycling_steps": 3,
    "matmul_precision": "high",
    "gamma_0": 0.0,
    "bf16_trunk": True,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# MSA injection (from evaluator.py)
# ---------------------------------------------------------------------------

def _inject_cached_msas(
    input_yaml: Path,
    msa_cache_root: Path,
    work_dir: Path,
) -> Optional[Path]:
    """Create a modified input YAML with msa: fields pointing to cached MSA files."""
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
    print(f"[struct-val] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


# ---------------------------------------------------------------------------
# Prediction runners
# ---------------------------------------------------------------------------

def _run_baseline(input_yaml: Path, out_dir: Path, config: dict) -> dict:
    """Run baseline prediction using the standard boltz_wrapper.py (Lightning Trainer)."""
    wrapper = str(Path("/eval/boltz_wrapper.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", "1",
        "--override",
        "--matmul_precision", config.get("matmul_precision", "highest"),
    ]
    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")
    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    result: dict[str, Any] = {"error": None}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            result["error"] = (
                f"Baseline exited {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
    except subprocess.TimeoutExpired:
        result["error"] = "Baseline timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected: {exc}"
    return result


def _run_optimized(input_yaml: Path, out_dir: Path, config: dict) -> dict:
    """Run optimized prediction using the bypass-lightning wrapper."""
    wrapper = str(Path("/eval/boltz_bypass_wrapper.py"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 12)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", "1",
        "--override",
        "--matmul_precision", config.get("matmul_precision", "high"),
        "--gamma_0", str(config.get("gamma_0", 0.0)),
        "--noise_scale", str(config.get("noise_scale", 1.003)),
        "--enable_kernels",
        "--cuda_warmup",
    ]
    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")
    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")
    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    result: dict[str, Any] = {"error": None}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            result["error"] = (
                f"Optimized exited {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
    except subprocess.TimeoutExpired:
        result["error"] = "Optimized timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected: {exc}"
    return result


# ---------------------------------------------------------------------------
# Structure comparison
# ---------------------------------------------------------------------------

def _find_mmcif(out_dir: Path, input_yaml: Path) -> Optional[Path]:
    """Locate the predicted mmCIF file in boltz output."""
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return None

    cif_files = sorted(results_dir.glob("*.cif"))
    if cif_files:
        return cif_files[0]

    # Also try mmcif extension
    cif_files = sorted(results_dir.glob("*.mmcif"))
    if cif_files:
        return cif_files[0]

    return None


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    """Parse Boltz confidence summary JSON."""
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Not found: {results_dir}"}

    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON"}

    with confidence_files[0].open() as f:
        conf = json.load(f)

    quality: dict[str, Any] = {}
    for key in ["confidence_score", "ptm", "iptm", "complex_plddt", "complex_iplddt"]:
        if key in conf:
            quality[key] = conf[key]
    return quality


def _compare_structures(cif_baseline: Path, cif_optimized: Path) -> dict[str, Any]:
    """Compare two mmCIF structures: CA RMSD, all-atom RMSD, per-residue deviation.

    Uses BioPython Superimposer for optimal alignment.
    """
    import numpy as np
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB import Superimposer

    parser = MMCIFParser(QUIET=True)

    try:
        struct_bl = parser.get_structure("baseline", str(cif_baseline))
        struct_opt = parser.get_structure("optimized", str(cif_optimized))
    except Exception as exc:
        return {"error": f"Failed to parse mmCIF: {exc}"}

    # Extract atoms from first model of each structure
    model_bl = list(struct_bl.get_models())[0]
    model_opt = list(struct_opt.get_models())[0]

    # --- CA (alpha-carbon) atoms for backbone RMSD ---
    ca_bl = []
    ca_opt = []
    ca_bl_by_res = {}
    ca_opt_by_res = {}

    for chain in model_bl.get_chains():
        for residue in chain.get_residues():
            res_id = (chain.id, residue.id[1])
            for atom in residue.get_atoms():
                if atom.get_name() == "CA":
                    ca_bl.append(atom)
                    ca_bl_by_res[res_id] = atom

    for chain in model_opt.get_chains():
        for residue in chain.get_residues():
            res_id = (chain.id, residue.id[1])
            for atom in residue.get_atoms():
                if atom.get_name() == "CA":
                    ca_opt.append(atom)
                    ca_opt_by_res[res_id] = atom

    # Match CA atoms by residue identity
    matched_bl_ca = []
    matched_opt_ca = []
    for res_id in ca_bl_by_res:
        if res_id in ca_opt_by_res:
            matched_bl_ca.append(ca_bl_by_res[res_id])
            matched_opt_ca.append(ca_opt_by_res[res_id])

    result: dict[str, Any] = {
        "num_ca_baseline": len(ca_bl),
        "num_ca_optimized": len(ca_opt),
        "num_ca_matched": len(matched_bl_ca),
    }

    if len(matched_bl_ca) < 3:
        result["error"] = f"Too few matched CA atoms: {len(matched_bl_ca)}"
        return result

    # CA RMSD after superposition
    sup = Superimposer()
    sup.set_atoms(matched_bl_ca, matched_opt_ca)
    ca_rmsd = sup.rms
    result["ca_rmsd_A"] = float(ca_rmsd)

    # --- All-atom RMSD ---
    all_bl_by_id = {}
    all_opt_by_id = {}

    for chain in model_bl.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
                atom_id = (chain.id, residue.id[1], atom.get_name())
                all_bl_by_id[atom_id] = atom

    for chain in model_opt.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
                atom_id = (chain.id, residue.id[1], atom.get_name())
                all_opt_by_id[atom_id] = atom

    matched_bl_all = []
    matched_opt_all = []
    for atom_id in all_bl_by_id:
        if atom_id in all_opt_by_id:
            matched_bl_all.append(all_bl_by_id[atom_id])
            matched_opt_all.append(all_opt_by_id[atom_id])

    result["num_atoms_baseline"] = len(all_bl_by_id)
    result["num_atoms_optimized"] = len(all_opt_by_id)
    result["num_atoms_matched"] = len(matched_bl_all)

    if len(matched_bl_all) >= 3:
        sup_all = Superimposer()
        sup_all.set_atoms(matched_bl_all, matched_opt_all)
        result["all_atom_rmsd_A"] = float(sup_all.rms)
    else:
        result["all_atom_rmsd_A"] = None
        result["warning"] = f"Too few matched atoms for all-atom RMSD: {len(matched_bl_all)}"

    # --- Per-residue deviation (after CA superposition) ---
    # Apply the CA rotation/translation to the optimized structure, then
    # measure per-residue CA distance (no further alignment).
    sup.apply(matched_opt_ca)
    per_residue = []
    matched_res_ids = [res_id for res_id in ca_bl_by_res if res_id in ca_opt_by_res]
    matched_res_ids.sort()

    for res_id in matched_res_ids:
        bl_coord = ca_bl_by_res[res_id].get_vector().get_array()
        opt_coord = ca_opt_by_res[res_id].get_vector().get_array()
        dist = float(np.linalg.norm(bl_coord - opt_coord))
        per_residue.append({
            "chain": res_id[0],
            "resid": res_id[1],
            "ca_deviation_A": round(dist, 3),
        })

    result["per_residue_deviations"] = per_residue
    if per_residue:
        deviations = [r["ca_deviation_A"] for r in per_residue]
        result["max_ca_deviation_A"] = max(deviations)
        result["mean_ca_deviation_A"] = round(sum(deviations) / len(deviations), 3)
        result["median_ca_deviation_A"] = round(
            sorted(deviations)[len(deviations) // 2], 3
        )
        result["pct_under_2A"] = round(
            100.0 * sum(1 for d in deviations if d < 2.0) / len(deviations), 1
        )

    return result


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=10800,  # 3 hours for running both configs
    volumes={"/msa_cache": msa_volume},
)
def validate(sanity_check: bool = False) -> str:
    """Run baseline + optimized predictions and compare structures.

    Returns JSON with per-complex RMSD results.
    """
    import yaml

    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        eval_config = yaml.safe_load(f)

    test_cases = eval_config.get("test_cases", [])
    if sanity_check:
        test_cases = [test_cases[0]] if test_cases else []

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())
    print(f"[struct-val] MSA cache: {'available' if use_msa_cache else 'NOT available'}")

    results: dict[str, Any] = {
        "baseline_config": BASELINE_CONFIG,
        "optimized_config": OPTIMIZED_CONFIG,
        "sanity_check": sanity_check,
        "per_complex": [],
        "summary": {},
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"YAML not found: {tc_yaml}",
            })
            continue

        print(f"\n{'='*60}")
        print(f"[struct-val] Processing: {tc_name}")
        print(f"{'='*60}")

        # --- Baseline prediction ---
        bl_work = Path(f"/tmp/struct_val/baseline/{tc_name}_{uuid.uuid4().hex[:8]}")
        bl_work.mkdir(parents=True, exist_ok=True)

        bl_config = dict(BASELINE_CONFIG)
        bl_yaml = tc_yaml
        if use_msa_cache:
            cached = _inject_cached_msas(tc_yaml, msa_cache_root, bl_work)
            if cached is not None:
                bl_yaml = cached
                bl_config["_msa_cached"] = True

        if sanity_check:
            bl_config["sampling_steps"] = 10

        print(f"[struct-val] Running baseline (steps={bl_config['sampling_steps']})...")
        t0 = time.perf_counter()
        bl_result = _run_baseline(bl_yaml, bl_work, bl_config)
        bl_time = time.perf_counter() - t0
        print(f"[struct-val] Baseline done in {bl_time:.1f}s")

        if bl_result["error"]:
            results["per_complex"].append({
                "name": tc_name,
                "error": f"Baseline failed: {bl_result['error'][:500]}",
            })
            continue

        # --- Optimized prediction ---
        opt_work = Path(f"/tmp/struct_val/optimized/{tc_name}_{uuid.uuid4().hex[:8]}")
        opt_work.mkdir(parents=True, exist_ok=True)

        opt_config = dict(OPTIMIZED_CONFIG)
        opt_yaml = tc_yaml
        if use_msa_cache:
            cached = _inject_cached_msas(tc_yaml, msa_cache_root, opt_work)
            if cached is not None:
                opt_yaml = cached
                opt_config["_msa_cached"] = True

        if sanity_check:
            opt_config["sampling_steps"] = 5

        print(f"[struct-val] Running optimized (steps={opt_config['sampling_steps']}, "
              f"gamma_0={opt_config.get('gamma_0')})...")
        t0 = time.perf_counter()
        opt_result = _run_optimized(opt_yaml, opt_work, opt_config)
        opt_time = time.perf_counter() - t0
        print(f"[struct-val] Optimized done in {opt_time:.1f}s")

        if opt_result["error"]:
            results["per_complex"].append({
                "name": tc_name,
                "error": f"Optimized failed: {opt_result['error'][:500]}",
            })
            continue

        # --- Find mmCIF files ---
        cif_bl = _find_mmcif(bl_work, bl_yaml)
        cif_opt = _find_mmcif(opt_work, opt_yaml)

        if cif_bl is None:
            results["per_complex"].append({
                "name": tc_name,
                "error": "Baseline mmCIF not found",
            })
            continue
        if cif_opt is None:
            results["per_complex"].append({
                "name": tc_name,
                "error": "Optimized mmCIF not found",
            })
            continue

        print(f"[struct-val] Baseline mmCIF: {cif_bl}")
        print(f"[struct-val] Optimized mmCIF: {cif_opt}")

        # --- Parse confidence metrics ---
        bl_quality = _parse_confidence(bl_work, bl_yaml)
        opt_quality = _parse_confidence(opt_work, opt_yaml)

        # --- Structure comparison ---
        print("[struct-val] Computing structural comparison...")
        comparison = _compare_structures(cif_bl, cif_opt)

        entry: dict[str, Any] = {
            "name": tc_name,
            "baseline_time_s": round(bl_time, 1),
            "optimized_time_s": round(opt_time, 1),
            "baseline_plddt": bl_quality.get("complex_plddt"),
            "optimized_plddt": opt_quality.get("complex_plddt"),
            "baseline_iptm": bl_quality.get("iptm"),
            "optimized_iptm": opt_quality.get("iptm"),
            "ca_rmsd_A": comparison.get("ca_rmsd_A"),
            "all_atom_rmsd_A": comparison.get("all_atom_rmsd_A"),
            "num_ca_matched": comparison.get("num_ca_matched"),
            "num_atoms_matched": comparison.get("num_atoms_matched"),
            "max_ca_deviation_A": comparison.get("max_ca_deviation_A"),
            "mean_ca_deviation_A": comparison.get("mean_ca_deviation_A"),
            "median_ca_deviation_A": comparison.get("median_ca_deviation_A"),
            "pct_under_2A": comparison.get("pct_under_2A"),
            "error": comparison.get("error"),
        }

        # Truncate per-residue deviations for JSON output (keep summary stats above)
        if "per_residue_deviations" in comparison:
            devs = comparison["per_residue_deviations"]
            # Only include residues with deviation > 2A as outliers
            outliers = [r for r in devs if r["ca_deviation_A"] > 2.0]
            entry["outlier_residues_gt_2A"] = outliers

        results["per_complex"].append(entry)

        # Print summary for this complex
        ca_rmsd = comparison.get("ca_rmsd_A")
        aa_rmsd = comparison.get("all_atom_rmsd_A")
        print(f"[struct-val] {tc_name}: CA RMSD = {ca_rmsd:.3f} A, "
              f"All-atom RMSD = {aa_rmsd:.3f} A" if ca_rmsd and aa_rmsd
              else f"[struct-val] {tc_name}: comparison incomplete")

    # --- Aggregate summary ---
    successful = [r for r in results["per_complex"] if r.get("error") is None]
    if successful:
        ca_rmsds = [r["ca_rmsd_A"] for r in successful if r.get("ca_rmsd_A") is not None]
        aa_rmsds = [r["all_atom_rmsd_A"] for r in successful if r.get("all_atom_rmsd_A") is not None]

        summary: dict[str, Any] = {
            "num_successful": len(successful),
            "num_total": len(test_cases),
        }

        if ca_rmsds:
            summary["mean_ca_rmsd_A"] = round(sum(ca_rmsds) / len(ca_rmsds), 3)
            summary["max_ca_rmsd_A"] = round(max(ca_rmsds), 3)
            summary["all_ca_rmsd_under_0.5A"] = all(r < 0.5 for r in ca_rmsds)
            summary["all_ca_rmsd_under_1.0A"] = all(r < 1.0 for r in ca_rmsds)

        if aa_rmsds:
            summary["mean_all_atom_rmsd_A"] = round(sum(aa_rmsds) / len(aa_rmsds), 3)
            summary["max_all_atom_rmsd_A"] = round(max(aa_rmsds), 3)
            summary["all_aa_rmsd_under_1.0A"] = all(r < 1.0 for r in aa_rmsds)
            summary["all_aa_rmsd_under_2.0A"] = all(r < 2.0 for r in aa_rmsds)

        # Verdict
        if ca_rmsds and aa_rmsds:
            if summary["all_ca_rmsd_under_0.5A"] and summary["all_aa_rmsd_under_1.0A"]:
                summary["verdict"] = "PASS: structures are nearly identical (precision changes only)"
            elif summary["all_ca_rmsd_under_1.0A"] and summary["all_aa_rmsd_under_2.0A"]:
                summary["verdict"] = "MARGINAL: small structural differences detected"
            else:
                summary["verdict"] = "DIVERGENT: significant structural differences — ODE trajectory differs"

        results["summary"] = summary

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(sanity_check: bool = False):
    """Structural validation: compare baseline vs optimized predictions.

    Usage:
        modal run orbits/structural-validation/validate_structures.py
        modal run orbits/structural-validation/validate_structures.py --sanity-check
    """
    print("[struct-val] Starting structural validation...")
    result_json = validate.remote(sanity_check=sanity_check)
    result = json.loads(result_json)

    # Print results table
    print(f"\n{'='*80}")
    print("STRUCTURAL VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"\nBaseline: {json.dumps(BASELINE_CONFIG)}")
    print(f"Optimized: {json.dumps(OPTIMIZED_CONFIG)}")
    print()

    print(f"{'Complex':<20} {'CA RMSD':>10} {'AA RMSD':>10} {'Max Dev':>10} "
          f"{'%<2A':>8} {'BL pLDDT':>10} {'Opt pLDDT':>10}")
    print("-" * 80)

    for pc in result.get("per_complex", []):
        if pc.get("error"):
            print(f"{pc['name']:<20} ERROR: {pc['error'][:50]}")
            continue

        ca = pc.get("ca_rmsd_A")
        aa = pc.get("all_atom_rmsd_A")
        maxd = pc.get("max_ca_deviation_A")
        pct = pc.get("pct_under_2A")
        bp = pc.get("baseline_plddt")
        op = pc.get("optimized_plddt")

        ca_s = f"{ca:.3f} A" if ca is not None else "N/A"
        aa_s = f"{aa:.3f} A" if aa is not None else "N/A"
        maxd_s = f"{maxd:.3f} A" if maxd is not None else "N/A"
        pct_s = f"{pct:.1f}%" if pct is not None else "N/A"
        bp_s = f"{bp:.4f}" if bp is not None else "N/A"
        op_s = f"{op:.4f}" if op is not None else "N/A"

        print(f"{pc['name']:<20} {ca_s:>10} {aa_s:>10} {maxd_s:>10} "
              f"{pct_s:>8} {bp_s:>10} {op_s:>10}")

    summary = result.get("summary", {})
    if summary:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  Mean CA RMSD:      {summary.get('mean_ca_rmsd_A', 'N/A')} A")
        print(f"  Max CA RMSD:       {summary.get('max_ca_rmsd_A', 'N/A')} A")
        print(f"  Mean AA RMSD:      {summary.get('mean_all_atom_rmsd_A', 'N/A')} A")
        print(f"  Max AA RMSD:       {summary.get('max_all_atom_rmsd_A', 'N/A')} A")
        print(f"  All CA < 0.5A:     {summary.get('all_ca_rmsd_under_0.5A', 'N/A')}")
        print(f"  All AA < 1.0A:     {summary.get('all_aa_rmsd_under_1.0A', 'N/A')}")
        verdict = summary.get("verdict", "UNKNOWN")
        print(f"\n  VERDICT: {verdict}")

    # Save full results
    out_path = SCRIPT_DIR / "validation_results.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[struct-val] Full results saved to {out_path}")

    print("\n--- FULL JSON ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    print("Use 'modal run orbits/structural-validation/validate_structures.py' to run on GPU.")
