"""Evaluator for bypass-clean: ODE-12 + bypass Trainer WITHOUT TF32 or bf16.

Isolates the effect of ODE sampling + bypass Trainer alone by using:
- matmul_precision=highest (no TF32)
- NO bf16_trunk flag
- gamma_0=0.0 (ODE sampler)
- sampling_steps=12
- recycling_steps=3
- cuda_warmup=true

Includes CA RMSD structural comparison against PDB ground truth.
Runs 3 seeds in parallel via Modal .map().

Usage:
    modal run orbits/bypass-clean/eval_bypass_clean.py
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
    .pip_install("biopython")
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(BYPASS_WRAPPER),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
)

app = modal.App("boltz-eval-bypass-clean", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v5", create_if_missing=False)
gt_volume = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=False)

# ---------------------------------------------------------------------------
# Configuration — the key difference: highest precision, NO bf16
# ---------------------------------------------------------------------------

EVAL_CONFIG = {
    "sampling_steps": 12,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "enable_kernels": True,
    "bf16_trunk": False,   # KEY: no bf16 trunk
    "cuda_warmup": True,
}

SEEDS = [42, 123, 7]

# Hardcoded ground truth info (from eval-v5 config, not in this branch's config.yaml)
GT_INFO = {
    "small_complex": {
        "pdb_id": "1BRS",
        "ground_truth": "1BRS.cif",
        "chain_mapping": {"A": "A", "B": "D"},
    },
    "medium_complex": {
        "pdb_id": "1DQJ",
        "ground_truth": "1DQJ.cif",
        "chain_mapping": {"A": "A", "B": "B", "C": "C"},
    },
    "large_complex": {
        "pdb_id": "2DN2",
        "ground_truth": "2DN2.cif",
        "chain_mapping": {"A": "A", "B": "B", "C": "C", "D": "D"},
    },
}


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
    print(f"[eval-clean] MSA cache: injected {injected} cached MSA(s) for {target_name}")
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

    # Kernel control
    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    # bf16 trunk — NOT passed for this experiment
    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    # CUDA warmup
    if config.get("cuda_warmup", False):
        cmd.append("--cuda_warmup")

    # MSA handling
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

        # Parse phase timestamps from stderr
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
    gt_root: Path,
    tc_name_override: str = None,
) -> dict[str, Any]:
    """Compare predicted structure against PDB ground truth.

    Returns dict with ca_rmsd, matched_residues, etc.
    """
    tc_name = tc_name_override or input_yaml.stem
    if tc_name.endswith("_cached"):
        tc_name = tc_name.rsplit("_cached", 1)[0]

    gt_info = GT_INFO.get(tc_name)
    if gt_info is None:
        return {"error": f"No GT info for {tc_name}"}

    gt_file = gt_info["ground_truth"]
    chain_mapping = gt_info["chain_mapping"]

    gt_path = gt_root / Path(gt_file).name
    if not gt_path.exists():
        return {"error": f"Ground truth file not found: {gt_path}"}

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

    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
        from Bio.SVDSuperimposer import SVDSuperimposer
        import numpy as np

        parser = MMCIFParser(QUIET=True)
        gt_structure = parser.get_structure("gt", str(gt_path))
        pred_structure = parser.get_structure("pred", str(pred_cif))

        gt_ca = {}
        for chain in gt_structure[0]:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    gt_ca[(chain.id, residue.id[1])] = residue["CA"].get_vector().get_array()

        pred_ca = {}
        for chain in pred_structure[0]:
            gt_chain_id = chain_mapping.get(chain.id, chain.id)
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    pred_ca[(gt_chain_id, residue.id[1])] = residue["CA"].get_vector().get_array()

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

        sup = SVDSuperimposer()
        sup.set(gt_coords, pred_coords)
        sup.run()
        ca_rmsd = sup.get_rms()

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
# Modal function: evaluate one (seed, test_case) pair
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={
        "/msa_cache": msa_cache,
        "/ground_truth": gt_volume,
    },
)
def evaluate_single(seed: int, tc_name: str) -> str:
    """Evaluate a single test case with a single seed."""
    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    tc_config = None
    for tc in test_cases:
        if tc["name"] == tc_name:
            tc_config = tc
            break

    if tc_config is None:
        return json.dumps({"error": f"Test case {tc_name} not found"})

    tc_yaml = Path("/eval") / tc_config["yaml"]
    if not tc_yaml.exists():
        return json.dumps({"error": f"YAML not found: {tc_yaml}"})

    config = dict(EVAL_CONFIG)
    config["seed"] = seed

    # MSA cache
    msa_cache_root = Path("/msa_cache")
    gt_root = Path("/ground_truth")

    work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{seed}_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    effective_yaml = tc_yaml
    if msa_cache_root.exists() and any(msa_cache_root.iterdir()):
        cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
        if cached_yaml is not None:
            effective_yaml = cached_yaml
            config["_msa_cached"] = True

    print(f"[eval-clean] Running {tc_name} seed={seed}, "
          f"steps={config['sampling_steps']}, recycle={config['recycling_steps']}, "
          f"gamma_0={config['gamma_0']}, matmul={config['matmul_precision']}, "
          f"bf16_trunk={config['bf16_trunk']}, warmup={config['cuda_warmup']}")

    pred_result = _run_boltz_bypass(effective_yaml, work_dir, config)

    # Structural comparison
    struct_result = {}
    if pred_result["error"] is None:
        struct_result = _compare_structures(
            work_dir, effective_yaml, gt_root, tc_name_override=tc_config["name"]
        )

    output = {
        "tc_name": tc_name,
        "seed": seed,
        "wall_time_s": pred_result["wall_time_s"],
        "predict_only_s": pred_result.get("predict_only_s"),
        "quality": pred_result["quality"],
        "structural": struct_result,
        "error": pred_result["error"],
        "config": config,
    }

    # Remove non-serializable items
    output["config"] = {k: v for k, v in config.items() if k != "_msa_cached"}

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# CLI entrypoint: run all seeds x test_cases in parallel
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run bypass-clean evaluation: 3 seeds x 3 test cases = 9 jobs in parallel."""

    test_cases = ["small_complex", "medium_complex", "large_complex"]

    # Build all (seed, tc_name) pairs
    jobs = []
    for seed in SEEDS:
        for tc in test_cases:
            jobs.append((seed, tc))

    print(f"[eval-clean] Launching {len(jobs)} jobs: {len(SEEDS)} seeds x {len(test_cases)} test cases")
    print(f"[eval-clean] Config: matmul_precision=highest, bf16_trunk=False, "
          f"gamma_0=0.0, steps=12, recycle=3, warmup=True")

    # Run all in parallel via .map()
    seed_args = [j[0] for j in jobs]
    tc_args = [j[1] for j in jobs]

    results_json = list(evaluate_single.map(seed_args, tc_args))

    # Parse results
    all_results = []
    for rj in results_json:
        all_results.append(json.loads(rj))

    # Print per-job results
    print("\n" + "=" * 80)
    print("INDIVIDUAL RESULTS")
    print("=" * 80)
    for r in all_results:
        if r.get("error"):
            print(f"  {r['tc_name']} seed={r['seed']}: ERROR - {r['error'][:200]}")
        else:
            po = r.get("predict_only_s")
            po_str = f"{po:.1f}s" if po else "N/A"
            plddt = r.get("quality", {}).get("complex_plddt", "N/A")
            plddt_str = f"{plddt:.4f}" if isinstance(plddt, (int, float)) else str(plddt)
            ca = r.get("structural", {}).get("ca_rmsd", "N/A")
            ca_str = f"{ca:.3f}" if isinstance(ca, (int, float)) else str(ca)
            wt = r.get("wall_time_s")
            wt_str = f"{wt:.1f}s" if wt else "N/A"
            print(f"  {r['tc_name']} seed={r['seed']}: "
                  f"wall={wt_str}, predict={po_str}, pLDDT={plddt_str}, CA_RMSD={ca_str}")

    # Aggregate by test case
    print("\n" + "=" * 80)
    print("AGGREGATED BY TEST CASE (mean +/- std across seeds)")
    print("=" * 80)

    baseline_times = {"small_complex": 14.0, "medium_complex": 33.5, "large_complex": 41.9}
    baseline_plddts = {"small_complex": 0.967, "medium_complex": 0.962, "large_complex": 0.966}
    baseline_rmsds = {"small_complex": 0.325, "medium_complex": 5.243, "large_complex": 0.474}
    baseline_mean_time = 29.78

    import statistics

    grand_predict_times = []
    grand_wall_times = []
    grand_plddts = []

    for tc in test_cases:
        tc_results = [r for r in all_results if r["tc_name"] == tc and r.get("error") is None]

        if not tc_results:
            print(f"\n{tc}: ALL FAILED")
            continue

        wall_times = [r["wall_time_s"] for r in tc_results if r.get("wall_time_s")]
        predict_times = [r["predict_only_s"] for r in tc_results if r.get("predict_only_s")]
        plddts = [r["quality"]["complex_plddt"] for r in tc_results
                   if r.get("quality", {}).get("complex_plddt") is not None]
        ca_rmsds = [r["structural"]["ca_rmsd"] for r in tc_results
                    if r.get("structural", {}).get("ca_rmsd") is not None]

        mean_wall = statistics.mean(wall_times) if wall_times else 0
        std_wall = statistics.stdev(wall_times) if len(wall_times) > 1 else 0
        mean_pred = statistics.mean(predict_times) if predict_times else 0
        std_pred = statistics.stdev(predict_times) if len(predict_times) > 1 else 0
        mean_plddt = statistics.mean(plddts) if plddts else 0
        std_plddt = statistics.stdev(plddts) if len(plddts) > 1 else 0
        mean_rmsd = statistics.mean(ca_rmsds) if ca_rmsds else 0
        std_rmsd = statistics.stdev(ca_rmsds) if len(ca_rmsds) > 1 else 0

        grand_predict_times.extend(predict_times)
        grand_wall_times.extend(wall_times)
        grand_plddts.extend(plddts)

        bl_time = baseline_times.get(tc, 0)
        bl_plddt = baseline_plddts.get(tc, 0)
        bl_rmsd = baseline_rmsds.get(tc, 0)

        speedup = bl_time / mean_pred if mean_pred > 0 else 0

        print(f"\n{tc}:")
        print(f"  predict_only_s: {mean_pred:.1f} +/- {std_pred:.1f} (baseline: {bl_time:.1f}s, speedup: {speedup:.2f}x)")
        print(f"  wall_time_s:    {mean_wall:.1f} +/- {std_wall:.1f}")
        print(f"  pLDDT:          {mean_plddt:.4f} +/- {std_plddt:.4f} (baseline: {bl_plddt:.3f})")
        print(f"  CA RMSD:        {mean_rmsd:.3f} +/- {std_rmsd:.3f} A (baseline: {bl_rmsd:.3f} A)")
        print(f"  Seeds: {[r['seed'] for r in tc_results]}")
        for r in tc_results:
            po = r.get("predict_only_s", 0)
            p = r.get("quality", {}).get("complex_plddt", 0)
            ca = r.get("structural", {}).get("ca_rmsd", 0)
            print(f"    seed={r['seed']}: predict={po:.1f}s, pLDDT={p:.4f}, CA_RMSD={ca:.3f}")

    # Grand aggregate
    print("\n" + "=" * 80)
    print("GRAND AGGREGATE")
    print("=" * 80)

    if grand_predict_times:
        # Mean per-complex predict time (average of per-complex means)
        per_tc_means = []
        for tc in test_cases:
            tc_pred = [r["predict_only_s"] for r in all_results
                       if r["tc_name"] == tc and r.get("predict_only_s") and r.get("error") is None]
            if tc_pred:
                per_tc_means.append(statistics.mean(tc_pred))

        grand_mean_predict = statistics.mean(per_tc_means) if per_tc_means else 0
        grand_speedup = baseline_mean_time / grand_mean_predict if grand_mean_predict > 0 else 0
        grand_mean_plddt = statistics.mean(grand_plddts) if grand_plddts else 0

        print(f"  Mean predict time (across complexes): {grand_mean_predict:.1f}s")
        print(f"  Speedup vs baseline (29.78s): {grand_speedup:.2f}x")
        print(f"  Mean pLDDT: {grand_mean_plddt:.4f} (baseline: 0.9650)")
        plddt_delta = (grand_mean_plddt - 0.9650) * 100
        print(f"  pLDDT delta: {plddt_delta:+.2f} pp")
        passes = (0.9650 - grand_mean_plddt) * 100 <= 2.0
        print(f"  Quality gate: {'PASS' if passes else 'FAIL'}")

    # Dump full JSON
    print("\n--- FULL JSON RESULTS ---")
    print(json.dumps(all_results, indent=2))
