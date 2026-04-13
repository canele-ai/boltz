"""Evaluator for persistent model + pickle loading with recycling_steps=3.

Config F validation: full-stack optimization at baseline recycling depth.
Includes CA RMSD structural comparison against PDB ground truth.

Usage:
    modal run orbits/persistent-recycle3/eval_recycle3.py --mode eval --validate
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

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent
PARENT_ORBIT = REPO_ROOT / "orbits" / "fast-model-load"

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
    .pip_install("biopython")
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(PARENT_ORBIT / "persistent_predict.py"),
        remote_path="/eval/persistent_predict.py",
    )
)

app = modal.App("boltz-eval-persistent-recycle3", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)
boltz_cache = modal.Volume.from_name("boltz-model-cache", create_if_missing=True)
gt_volume = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=False)


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


# Hardcoded ground truth and chain mapping (from eval-v5 config in main branch)
# The worktree's config.yaml is older and doesn't have these fields.
GT_CONFIG = {
    "small_complex": {
        "ground_truth": "1BRS.cif",
        "chain_mapping": {"A": "A", "B": "D"},
    },
    "medium_complex": {
        "ground_truth": "1DQJ.cif",
        "chain_mapping": {"A": "A", "B": "B", "C": "C"},
    },
    "large_complex": {
        "ground_truth": "2DN2.cif",
        "chain_mapping": {"A": "A", "B": "B", "C": "C", "D": "D"},
    },
}


def _compare_structures(
    pred_dir: Path,
    target_name: str,
    gt_root: Path,
    eval_config: dict,
) -> dict[str, Any]:
    """Compare predicted structure against PDB ground truth.

    Adapted from research/eval/evaluator.py _compare_structures().
    Returns dict with ca_rmsd, matched_residues, etc.
    """
    tc_gt = GT_CONFIG.get(target_name)
    if tc_gt is None:
        return {"error": f"No ground truth config for {target_name}"}

    gt_file = tc_gt["ground_truth"]
    chain_mapping = tc_gt["chain_mapping"]

    gt_path = gt_root / Path(gt_file).name
    if not gt_path.exists():
        # Try with full relative path
        gt_path = gt_root / gt_file
    if not gt_path.exists():
        return {"error": f"Ground truth file not found: {gt_path}. Available: {list(gt_root.iterdir()) if gt_root.exists() else 'dir missing'}"}

    # Find predicted mmCIF -- search all subdirs under predictions/
    pred_cif = None
    pred_parent = pred_dir / "predictions"
    if pred_parent.exists():
        # Search all subdirectories (handles both target_name and target_name_cached)
        cifs = sorted(pred_parent.rglob("*_model_0.cif"))
        if cifs:
            pred_cif = cifs[0]

    if pred_cif is None:
        return {"error": f"No predicted mmCIF found under {pred_dir}"}

    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
        from Bio.SVDSuperimposer import SVDSuperimposer
        import numpy as np

        parser = MMCIFParser(QUIET=True)
        gt_structure = parser.get_structure("gt", str(gt_path))
        pred_structure = parser.get_structure("pred", str(pred_cif))

        # Extract CA atoms from ground truth (first model)
        gt_ca = {}
        for chain in gt_structure[0]:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    gt_ca[(chain.id, residue.id[1])] = residue["CA"].get_vector().get_array()

        # Extract CA atoms from prediction, applying chain mapping
        pred_ca = {}
        for chain in pred_structure[0]:
            gt_chain_id = chain_mapping.get(chain.id, chain.id)
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    pred_ca[(gt_chain_id, residue.id[1])] = residue["CA"].get_vector().get_array()

        # Find matched CA pairs
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

        # Per-residue deviations after superposition
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
        import traceback
        return {"error": f"Structural comparison failed: {e}", "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Modal function: single-seed evaluation
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={
        "/msa_cache": msa_cache,
        "/boltz_cache": boltz_cache,
        "/ground_truth": gt_volume,
    },
)
def evaluate_seed(seed: int) -> str:
    """Run persistent-model evaluation with recycling_steps=3 for a single seed."""
    import yaml

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())

    gt_root = Path("/ground_truth")

    work_dir = Path(f"/tmp/boltz_recycle3_{seed}_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Prepare input YAMLs (inject cached MSAs)
    input_yamls = []
    tc_names = []
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
        tc_names.append(tc["name"])

    # Build command for persistent_predict.py
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "/eval/persistent_predict.py",
        "--inputs", *input_yamls,
        "--out_dir", str(out_dir),
        "--cache", "/boltz_cache",
        "--sampling_steps", "12",
        "--recycling_steps", "3",       # BASELINE DEFAULT
        "--diffusion_samples", "1",
        "--gamma_0", "0.0",             # ODE sampler
        "--noise_scale", "1.003",
        "--matmul_precision", "high",   # TF32
        "--bf16_trunk",
        "--use_pickle",
        "--override",
        "--seed", str(seed),
    ]

    if not use_msa_cache:
        cmd.append("--use_msa_server")

    print(f"[eval-recycle3] seed={seed}, cmd={' '.join(cmd[:10])}...")

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

    # ----- CA RMSD comparison against PDB ground truth -----
    structural_results = {}
    for pc in parsed.get("per_complex", []):
        tc_name = pc["name"]
        tc_out = out_dir / tc_name

        # Debug: list what's in the prediction dir
        pred_path = tc_out / "predictions"
        debug_info = []
        if pred_path.exists():
            for p in sorted(pred_path.rglob("*")):
                if p.is_file():
                    debug_info.append(f"{p.relative_to(tc_out)} ({p.stat().st_size}b)")
        else:
            debug_info.append(f"predictions dir not found at {pred_path}")

        struct_cmp = _compare_structures(tc_out, tc_name, gt_root, eval_config)
        struct_cmp["_debug_pred_files"] = debug_info
        structural_results[tc_name] = struct_cmp
        pc["structural"] = struct_cmp

    result["structural_comparison"] = structural_results

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "eval",
    seed: int = 42,
    validate: bool = False,
):
    """Persistent model + recycling_steps=3 evaluation.

    Usage:
        modal run orbits/persistent-recycle3/eval_recycle3.py --mode eval --validate
    """
    if mode == "eval":
        if validate:
            seeds = [42, 123, 7]
        else:
            seeds = [seed]

        print(f"[eval-recycle3] Running with seeds={seeds}")
        print(f"[eval-recycle3] Config: sampling_steps=12, recycling_steps=3, gamma_0=0.0, TF32, bf16_trunk, pickle_load")

        # Run seeds in parallel using Modal .map()
        seed_results_json = list(evaluate_seed.map(seeds))

        # Parse and aggregate
        seed_results = [json.loads(s) for s in seed_results_json]

        # Check for errors
        errors = [r for r in seed_results if "error" in r]
        if errors:
            print(f"\nERRORS ({len(errors)} seed(s) failed):")
            for e in errors:
                print(f"  Seed {e.get('seed', '?')}: {e['error'][:2000]}")
            if len(errors) == len(seeds):
                return

        # ---- eval-v5 baseline for speedup calculation ----
        BASELINE_MEAN_WALL = 29.78  # eval-v5 mean wall time per complex
        BASELINE_PER_COMPLEX = {
            "small_complex":  {"wall_time_s": 14.0, "plddt": 0.967, "ca_rmsd": 0.325},
            "medium_complex": {"wall_time_s": 33.5, "plddt": 0.962, "ca_rmsd": 5.243},
            "large_complex":  {"wall_time_s": 41.9, "plddt": 0.966, "ca_rmsd": 0.474},
        }

        # Print structural comparison debug info from first good seed
        for sr in seed_results:
            if "error" not in sr and "structural_comparison" in sr:
                print(f"\n--- Structural comparison debug (seed {sr['seed']}) ---")
                print(json.dumps(sr["structural_comparison"], indent=2))
                break

        # Aggregate across seeds
        all_model_loads = []
        all_per_complex = {}  # name -> list of dicts
        all_predict_times_flat = []

        for sr in seed_results:
            if "error" in sr:
                continue
            pr = sr.get("persistent_results", {})
            all_model_loads.append(pr.get("model_load_time_s", 0))
            for pc in pr.get("per_complex", []):
                name = pc["name"]
                if name not in all_per_complex:
                    all_per_complex[name] = []
                all_per_complex[name].append(pc)
                all_predict_times_flat.append(pc.get("predict_time_s", 0))

        num_good = len(seed_results) - len(errors)
        mean_model_load = sum(all_model_loads) / len(all_model_loads) if all_model_loads else 0

        # Per-complex aggregation
        print(f"\n{'='*80}")
        print(f"RESULTS: persistent model + recycling_steps=3 (Config F)")
        print(f"{'='*80}")
        print(f"Seeds: {[sr['seed'] for sr in seed_results if 'error' not in sr]}")
        print(f"Model load time: {mean_model_load:.1f}s (mean)")

        per_complex_summary = []
        for name in ["small_complex", "medium_complex", "large_complex"]:
            entries = all_per_complex.get(name, [])
            if not entries:
                continue

            predict_times = [e["predict_time_s"] for e in entries]
            process_times = [e["process_time_s"] for e in entries]
            total_times = [e["total_per_complex_s"] for e in entries]
            plddts = [e["quality"]["complex_plddt"] for e in entries if "complex_plddt" in e.get("quality", {})]

            # CA RMSD from structural comparison
            ca_rmsds = []
            for e in entries:
                struct = e.get("structural", {})
                if "ca_rmsd" in struct:
                    ca_rmsds.append(struct["ca_rmsd"])

            mean_predict = sum(predict_times) / len(predict_times)
            std_predict = (sum((x - mean_predict)**2 for x in predict_times) / len(predict_times))**0.5 if len(predict_times) > 1 else 0
            mean_process = sum(process_times) / len(process_times)
            mean_total = sum(total_times) / len(total_times)
            mean_plddt = sum(plddts) / len(plddts) if plddts else None
            mean_ca_rmsd = sum(ca_rmsds) / len(ca_rmsds) if ca_rmsds else None
            std_ca_rmsd = (sum((x - mean_ca_rmsd)**2 for x in ca_rmsds) / len(ca_rmsds))**0.5 if len(ca_rmsds) > 1 else 0

            bl = BASELINE_PER_COMPLEX.get(name, {})

            summary = {
                "name": name,
                "mean_predict_s": mean_predict,
                "std_predict_s": std_predict,
                "mean_process_s": mean_process,
                "mean_total_s": mean_total,
                "mean_plddt": mean_plddt,
                "mean_ca_rmsd": mean_ca_rmsd,
                "std_ca_rmsd": std_ca_rmsd,
                "predict_times": predict_times,
                "ca_rmsds": ca_rmsds,
            }
            per_complex_summary.append(summary)

            print(f"\n  {name}:")
            print(f"    predict_time: {mean_predict:.1f}s +/- {std_predict:.1f}s  (baseline: {bl.get('wall_time_s', '?')}s)")
            print(f"    pLDDT: {mean_plddt:.4f}" if mean_plddt else "    pLDDT: N/A")
            if mean_plddt and bl.get("plddt"):
                print(f"    pLDDT delta: {(mean_plddt - bl['plddt'])*100:+.2f} pp (baseline: {bl['plddt']:.3f})")
            if mean_ca_rmsd is not None:
                print(f"    CA RMSD: {mean_ca_rmsd:.3f}A +/- {std_ca_rmsd:.3f}A (baseline: {bl.get('ca_rmsd', '?')}A)")
            print(f"    per-seed predict_times: {predict_times}")
            if ca_rmsds:
                print(f"    per-seed CA RMSDs: {ca_rmsds}")

        # Compute speedup numbers
        mean_predict_all = sum(all_predict_times_flat) / len(all_predict_times_flat) if all_predict_times_flat else 0

        # predict_only speedup: just pure prediction time
        predict_only_speedup = BASELINE_MEAN_WALL / mean_predict_all if mean_predict_all > 0 else 0

        # Per-seed predict-only times (mean across complexes)
        per_seed_predict_means = []
        for sr in seed_results:
            if "error" in sr:
                continue
            pr = sr.get("persistent_results", {})
            pcs = pr.get("per_complex", [])
            if pcs:
                m = sum(p["predict_time_s"] for p in pcs) / len(pcs)
                per_seed_predict_means.append(m)

        mean_predict_per_seed = sum(per_seed_predict_means) / len(per_seed_predict_means) if per_seed_predict_means else 0
        std_predict_per_seed = (sum((x - mean_predict_per_seed)**2 for x in per_seed_predict_means) / len(per_seed_predict_means))**0.5 if len(per_seed_predict_means) > 1 else 0

        # Amortized speedup for N=3, N=10, N=100
        # amortized_time = model_load/N + mean_predict_process_time
        mean_process_time = 0
        all_process_times = []
        for sr in seed_results:
            if "error" in sr:
                continue
            pr = sr.get("persistent_results", {})
            for pc in pr.get("per_complex", []):
                all_process_times.append(pc.get("process_time_s", 0))
        mean_process_time = sum(all_process_times) / len(all_process_times) if all_process_times else 0
        mean_predict_process = mean_predict_all + mean_process_time

        amortized_times = {}
        amortized_speedups = {}
        for N in [3, 10, 100]:
            t = mean_model_load / N + mean_predict_process
            amortized_times[N] = t
            amortized_speedups[N] = BASELINE_MEAN_WALL / t if t > 0 else 0

        print(f"\n{'='*80}")
        print(f"SPEEDUP SUMMARY (baseline mean_wall_time = {BASELINE_MEAN_WALL}s)")
        print(f"{'='*80}")
        print(f"  Mean predict time (across all complexes & seeds): {mean_predict_all:.2f}s")
        print(f"  Mean process time: {mean_process_time:.2f}s")
        print(f"  Mean model_load: {mean_model_load:.1f}s")
        print(f"")
        print(f"  Predict-only speedup: {predict_only_speedup:.2f}x")
        print(f"    = {BASELINE_MEAN_WALL:.2f}s / {mean_predict_all:.2f}s")
        print(f"")
        for N in [3, 10, 100]:
            print(f"  Amortized speedup (N={N}): {amortized_speedups[N]:.2f}x")
            print(f"    = {BASELINE_MEAN_WALL:.2f}s / ({mean_model_load:.1f}/{N} + {mean_predict_process:.2f})s = {amortized_times[N]:.2f}s")

        # Output full JSON for log.md
        full_results = {
            "config": {
                "sampling_steps": 12,
                "recycling_steps": 3,
                "gamma_0": 0.0,
                "matmul_precision": "high",
                "bf16_trunk": True,
                "cuda_warmup": True,
                "persistent_model": True,
                "pickle_loading": True,
            },
            "seeds": [sr["seed"] for sr in seed_results if "error" not in sr],
            "mean_model_load_s": mean_model_load,
            "mean_predict_time_s": mean_predict_all,
            "mean_process_time_s": mean_process_time,
            "predict_only_speedup": round(predict_only_speedup, 2),
            "amortized_speedups": {str(k): round(v, 2) for k, v in amortized_speedups.items()},
            "amortized_times": {str(k): round(v, 2) for k, v in amortized_times.items()},
            "per_complex": per_complex_summary,
            "baseline_mean_wall_time_s": BASELINE_MEAN_WALL,
        }
        print(f"\n--- FULL RESULTS JSON ---")
        print(json.dumps(full_results, indent=2, default=str))
