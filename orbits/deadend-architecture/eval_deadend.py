"""Evaluator for dead-end architecture approaches on eval-v5.

Tests 4 configurations, all stacked on the winning config
(ODE-12 + TF32 + bf16 + bypass Lightning + recycling_steps=3):

1. Control (bypass only) — the winning config baseline
2. + DiffusionTransformer 24->8 layers
3. + Pairformer 64->48 blocks
4. + Token Merging 10%

Includes CA RMSD structural comparison against PDB ground truth.
All configs run with 3 seeds in parallel via Modal .map().

Usage:
    modal run orbits/deadend-architecture/eval_deadend.py
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
# Use eval-v5 test cases with PDB ground truth (copied from the eval-v5 commit)
EVAL_V5_DIR = ORBIT_DIR / "eval_v5"

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
    .pip_install("biopython")
    # Mount eval-v5 test cases and config (with ground truth references)
    .add_local_dir(str(EVAL_V5_DIR), remote_path="/eval")
    .add_local_file(
        str(ORBIT_DIR / "boltz_bypass_deadend.py"),
        remote_path="/eval/boltz_bypass_deadend.py",
    )
)

app = modal.App("boltz-eval-deadend-v5", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v5", create_if_missing=False)
gt_volume = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=False)

# ---------------------------------------------------------------------------
# Configurations to test
# ---------------------------------------------------------------------------

CONFIGS = {
    "control": {
        "label": "Control (bypass only)",
        "sampling_steps": 12,
        "recycling_steps": 3,
        "gamma_0": 0.0,
        "noise_scale": 1.003,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "cuda_warmup": True,
        # No pruning or merging
    },
    "dt8": {
        "label": "DiffTransformer 24->8",
        "sampling_steps": 12,
        "recycling_steps": 3,
        "gamma_0": 0.0,
        "noise_scale": 1.003,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "cuda_warmup": True,
        "diff_transformer_k": 8,
    },
    "pf48": {
        "label": "Pairformer 64->48",
        "sampling_steps": 12,
        "recycling_steps": 3,
        "gamma_0": 0.0,
        "noise_scale": 1.003,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "cuda_warmup": True,
        "pairformer_k": 48,
    },
    "tome10": {
        "label": "Token Merging 10%",
        "sampling_steps": 12,
        "recycling_steps": 3,
        "gamma_0": 0.0,
        "noise_scale": 1.003,
        "matmul_precision": "high",
        "bf16_trunk": True,
        "enable_kernels": True,
        "cuda_warmup": True,
        "tome_ratio": 0.1,
        "tome_merge_after_layer": 0,
    },
}

SEEDS = [42, 123, 7]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    import yaml
    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path):
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
    return cached_yaml


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
    eval_config: dict,
) -> dict[str, Any]:
    """Compare predicted structure against PDB ground truth via CA RMSD."""
    tc_name = input_yaml.stem
    if tc_name.endswith("_cached"):
        tc_name = tc_name.rsplit("_cached", 1)[0]

    test_cases = eval_config.get("test_cases", [])
    tc_config = None
    for tc in test_cases:
        if tc["name"] == tc_name:
            tc_config = tc
            break

    if tc_config is None:
        return {"error": f"No test case config found for {tc_name}"}

    gt_file = tc_config.get("ground_truth")
    chain_mapping = tc_config.get("chain_mapping", {})
    if not gt_file:
        return {"error": f"No ground_truth path in config for {tc_name}"}

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


def _run_prediction(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction using the deadend bypass wrapper."""
    wrapper = str(Path("/eval/boltz_bypass_deadend.py"))
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
    ]

    if config.get("enable_kernels", True):
        cmd.append("--enable_kernels")
    else:
        cmd.append("--no_kernels_flag")

    if config.get("bf16_trunk", False):
        cmd.append("--bf16_trunk")

    if config.get("cuda_warmup", False):
        cmd.append("--cuda_warmup")

    # Layer pruning flags
    if config.get("diff_transformer_k") is not None:
        cmd.extend(["--diff_transformer_k", str(config["diff_transformer_k"])])
    if config.get("pairformer_k") is not None:
        cmd.extend(["--pairformer_k", str(config["pairformer_k"])])

    # Token merging flags
    if config.get("tome_ratio", 0.0) > 0:
        cmd.extend(["--tome_ratio", str(config["tome_ratio"])])
        cmd.extend(["--tome_merge_after_layer", str(config.get("tome_merge_after_layer", 0))])

    # MSA handling
    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "high")
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

        if proc.returncode != 0:
            result["error"] = (
                f"exited with code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        # Parse phase timestamps
        predict_start = predict_end = None
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


# ---------------------------------------------------------------------------
# Modal function: evaluate one (config, seed) combination on all test cases
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={"/msa_cache": msa_cache, "/ground_truth": gt_volume},
)
def evaluate_single(config_key: str, seed: int) -> str:
    """Evaluate a single config+seed on all test cases. Returns JSON results."""
    config = dict(CONFIGS[config_key])
    config["seed"] = seed

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    msa_cache_root = Path("/msa_cache")
    gt_root = Path("/ground_truth")

    results = {
        "config_key": config_key,
        "label": config.get("label", config_key),
        "seed": seed,
        "per_complex": [],
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"YAML not found: {tc_yaml}",
                "wall_time_s": None,
                "quality": {},
                "structural": {},
            })
            continue

        work_dir = Path(f"/tmp/boltz_eval/{config_key}_{seed}_{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Inject cached MSAs
        run_config = dict(config)
        effective_yaml = tc_yaml
        if msa_cache_root.exists() and any(msa_cache_root.iterdir()):
            cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
            if cached_yaml is not None:
                effective_yaml = cached_yaml
                run_config["_msa_cached"] = True

        print(f"[eval-deadend] {config_key} seed={seed} {tc_name}: starting...")
        pred_result = _run_prediction(effective_yaml, work_dir, run_config)

        structural = {}
        if pred_result["error"] is None:
            structural = _compare_structures(work_dir, effective_yaml, gt_root, eval_config)
            if structural.get("error"):
                print(f"[eval-deadend] Structural: {structural['error']}")
            else:
                print(f"[eval-deadend] {tc_name}: CA RMSD={structural['ca_rmsd']}A, "
                      f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")

        entry = {
            "name": tc_name,
            "wall_time_s": pred_result["wall_time_s"],
            "predict_only_s": pred_result.get("predict_only_s"),
            "quality": pred_result["quality"],
            "structural": structural,
            "error": pred_result["error"],
        }
        results["per_complex"].append(entry)

        if pred_result["error"]:
            print(f"[eval-deadend] ERROR on {tc_name}: {pred_result['error'][:300]}")

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run all 4 configs x 3 seeds in parallel via Modal .map()."""
    print("[eval-deadend] Starting evaluation of 4 dead-end configs x 3 seeds")
    print(f"[eval-deadend] Configs: {list(CONFIGS.keys())}")
    print(f"[eval-deadend] Seeds: {SEEDS}")

    # Build list of (config_key, seed) pairs — 12 total jobs
    jobs = []
    for config_key in CONFIGS:
        for seed in SEEDS:
            jobs.append((config_key, seed))

    print(f"[eval-deadend] Launching {len(jobs)} parallel evaluations...")

    # Use starmap for parallel execution
    config_keys = [j[0] for j in jobs]
    seeds = [j[1] for j in jobs]

    all_results = {}
    for result_json in evaluate_single.map(config_keys, seeds):
        result = json.loads(result_json)
        key = f"{result['config_key']}_seed{result['seed']}"
        all_results[key] = result
        label = result.get("label", result["config_key"])
        seed = result["seed"]
        # Quick summary
        plddts = [
            pc["quality"].get("complex_plddt", 0)
            for pc in result["per_complex"]
            if pc["error"] is None
        ]
        times = [
            pc["wall_time_s"]
            for pc in result["per_complex"]
            if pc["error"] is None and pc["wall_time_s"] is not None
        ]
        mean_plddt = sum(plddts) / len(plddts) if plddts else 0
        mean_time = sum(times) / len(times) if times else 0
        print(f"[eval-deadend] {label} seed={seed}: mean_time={mean_time:.1f}s, mean_pLDDT={mean_plddt:.4f}")

    # Aggregate results by config
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (mean +/- std across 3 seeds)")
    print("=" * 80)

    import statistics
    aggregated = {}

    for config_key in CONFIGS:
        seed_results = [
            all_results[f"{config_key}_seed{s}"]
            for s in SEEDS
            if f"{config_key}_seed{s}" in all_results
        ]

        if not seed_results:
            print(f"\n{CONFIGS[config_key]['label']}: NO RESULTS")
            continue

        agg = {"label": CONFIGS[config_key]["label"], "per_complex": {}, "overall": {}}

        # Per-complex aggregation
        tc_names = [pc["name"] for pc in seed_results[0]["per_complex"]]
        all_times = []
        all_plddts = []

        for tc_name in tc_names:
            tc_data = {
                "wall_times": [],
                "predict_times": [],
                "plddts": [],
                "iptms": [],
                "ca_rmsds": [],
                "matched_residues": [],
                "pct_within_2A": [],
            }
            for sr in seed_results:
                for pc in sr["per_complex"]:
                    if pc["name"] == tc_name and pc["error"] is None:
                        if pc["wall_time_s"] is not None:
                            tc_data["wall_times"].append(pc["wall_time_s"])
                        if pc.get("predict_only_s") is not None:
                            tc_data["predict_times"].append(pc["predict_only_s"])
                        plddt = pc["quality"].get("complex_plddt")
                        if plddt is not None:
                            tc_data["plddts"].append(plddt)
                        iptm = pc["quality"].get("iptm")
                        if iptm is not None:
                            tc_data["iptms"].append(iptm)
                        rmsd = pc.get("structural", {}).get("ca_rmsd")
                        if rmsd is not None:
                            tc_data["ca_rmsds"].append(rmsd)
                        mr = pc.get("structural", {}).get("matched_residues")
                        if mr is not None:
                            tc_data["matched_residues"].append(mr)
                        pw = pc.get("structural", {}).get("pct_within_2A")
                        if pw is not None:
                            tc_data["pct_within_2A"].append(pw)

            agg["per_complex"][tc_name] = {
                "mean_wall_time": statistics.mean(tc_data["wall_times"]) if tc_data["wall_times"] else None,
                "std_wall_time": statistics.stdev(tc_data["wall_times"]) if len(tc_data["wall_times"]) > 1 else 0,
                "mean_plddt": statistics.mean(tc_data["plddts"]) if tc_data["plddts"] else None,
                "std_plddt": statistics.stdev(tc_data["plddts"]) if len(tc_data["plddts"]) > 1 else 0,
                "mean_iptm": statistics.mean(tc_data["iptms"]) if tc_data["iptms"] else None,
                "mean_ca_rmsd": statistics.mean(tc_data["ca_rmsds"]) if tc_data["ca_rmsds"] else None,
                "std_ca_rmsd": statistics.stdev(tc_data["ca_rmsds"]) if len(tc_data["ca_rmsds"]) > 1 else 0,
                "mean_matched_residues": statistics.mean(tc_data["matched_residues"]) if tc_data["matched_residues"] else None,
                "mean_pct_within_2A": statistics.mean(tc_data["pct_within_2A"]) if tc_data["pct_within_2A"] else None,
            }

            all_times.extend(tc_data["wall_times"])
            all_plddts.extend(tc_data["plddts"])

        # Overall aggregation
        all_mean_times = [
            agg["per_complex"][tc]["mean_wall_time"]
            for tc in tc_names
            if agg["per_complex"][tc]["mean_wall_time"] is not None
        ]
        all_mean_plddts = [
            agg["per_complex"][tc]["mean_plddt"]
            for tc in tc_names
            if agg["per_complex"][tc]["mean_plddt"] is not None
        ]
        all_mean_rmsds = [
            agg["per_complex"][tc]["mean_ca_rmsd"]
            for tc in tc_names
            if agg["per_complex"][tc]["mean_ca_rmsd"] is not None
        ]

        agg["overall"] = {
            "mean_wall_time": statistics.mean(all_mean_times) if all_mean_times else None,
            "mean_plddt": statistics.mean(all_mean_plddts) if all_mean_plddts else None,
            "mean_ca_rmsd": statistics.mean(all_mean_rmsds) if all_mean_rmsds else None,
        }

        aggregated[config_key] = agg

        # Print summary table
        label = CONFIGS[config_key]["label"]
        print(f"\n--- {label} ---")
        print(f"  Overall: time={agg['overall']['mean_wall_time']:.1f}s, "
              f"pLDDT={agg['overall']['mean_plddt']:.4f}, "
              f"CA RMSD={agg['overall']['mean_ca_rmsd']:.3f}A"
              if agg['overall']['mean_ca_rmsd'] is not None
              else f"  Overall: time={agg['overall'].get('mean_wall_time', 'N/A')}s, "
                   f"pLDDT={agg['overall'].get('mean_plddt', 'N/A')}")
        for tc_name in tc_names:
            tc = agg["per_complex"][tc_name]
            print(f"  {tc_name}: time={tc['mean_wall_time']:.1f}s +/- {tc['std_wall_time']:.1f}, "
                  f"pLDDT={tc['mean_plddt']:.4f} +/- {tc['std_plddt']:.4f}, "
                  f"CA RMSD={tc['mean_ca_rmsd']:.3f}A +/- {tc['std_ca_rmsd']:.3f}"
                  if tc['mean_ca_rmsd'] is not None
                  else f"  {tc_name}: time={tc.get('mean_wall_time', 'N/A')}, "
                       f"pLDDT={tc.get('mean_plddt', 'N/A')}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    baseline_plddt = 0.9650  # eval-v5 baseline
    baseline_time = 29.78    # eval-v5 baseline

    header = f"{'Config':<30} {'Time(s)':>8} {'Speedup':>8} {'pLDDT':>8} {'dPLDDT':>8} {'CA RMSD':>10} {'Gate':>6}"
    print(header)
    print("-" * 100)

    for config_key in CONFIGS:
        if config_key not in aggregated:
            continue
        agg = aggregated[config_key]
        label = CONFIGS[config_key]["label"][:28]
        t = agg["overall"]["mean_wall_time"]
        p = agg["overall"]["mean_plddt"]
        r = agg["overall"]["mean_ca_rmsd"]

        t_str = f"{t:.1f}" if t else "ERR"
        s_str = f"{baseline_time / t:.2f}x" if t else "ERR"
        p_str = f"{p:.4f}" if p else "ERR"
        dp = (p - baseline_plddt) * 100 if p else None
        dp_str = f"{dp:+.2f}pp" if dp is not None else "ERR"
        r_str = f"{r:.3f}A" if r is not None else "N/A"
        gate = "PASS" if dp is not None and dp >= -2.0 else "FAIL"

        print(f"{label:<30} {t_str:>8} {s_str:>8} {p_str:>8} {dp_str:>8} {r_str:>10} {gate:>6}")

    # Dump full results
    output = {
        "raw_results": all_results,
        "aggregated": aggregated,
    }

    output_path = Path("deadend_results.json")
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[eval-deadend] Full results written to {output_path}")

    # Also print full JSON for log parsing
    print("\n--- FULL AGGREGATED JSON ---")
    print(json.dumps(aggregated, indent=2))
