"""Re-validation of SDPA attention and head pruning on eval-v5 with CA RMSD.

Tests 3 configs x 3 seeds = 9 runs, all parallelized via Modal .map():
1. Control (bypass only) - fair comparison baseline
2. + SDPA attention (replacing einsum with torch.nn.functional.scaled_dot_product_attention)
3. + 75% head pruning (zeroing least-important attention heads)

All configs use: ODE-12, recycle=3, gamma_0=0.0, TF32, bf16, CUDA warmup.

Usage:
    modal run orbits/deadend-attention/eval_deadend.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent
BYPASS_WRAPPER = REPO_ROOT / "orbits" / "bypass-lightning" / "boltz_bypass_wrapper.py"

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

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
    .add_local_file(
        str(ORBIT_DIR / "boltz_bypass_sdpa.py"),
        remote_path="/eval/boltz_bypass_sdpa.py",
    )
    .add_local_file(
        str(ORBIT_DIR / "boltz_bypass_headprune.py"),
        remote_path="/eval/boltz_bypass_headprune.py",
    )
)

app = modal.App("boltz-eval-deadend-attention", image=boltz_image)

msa_volume = modal.Volume.from_name("boltz-msa-cache-v5", create_if_missing=True)
gt_volume = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=True)

# ---------------------------------------------------------------------------
# eval-v5 test cases (inline to avoid modifying research/)
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "small_complex",
        "yaml_name": "small_complex",
        "pdb_id": "1BRS",
        "ground_truth": "1BRS.cif",
        "chain_mapping": {"A": "A", "B": "D"},
    },
    {
        "name": "medium_complex",
        "yaml_name": "medium_complex",
        "pdb_id": "1DQJ",
        "ground_truth": "1DQJ.cif",
        "chain_mapping": {"A": "A", "B": "B", "C": "C"},
    },
    {
        "name": "large_complex",
        "yaml_name": "large_complex",
        "pdb_id": "2DN2",
        "ground_truth": "2DN2.cif",
        "chain_mapping": {"A": "A", "B": "B", "C": "C", "D": "D"},
    },
]

# Wrapper script names
WRAPPER_MAP = {
    "control": "boltz_bypass_wrapper.py",
    "sdpa": "boltz_bypass_sdpa.py",
    "headprune": "boltz_bypass_headprune.py",
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
    return cached_yaml


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name

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
    gt_root: Path,
    tc_config: dict,
) -> dict[str, Any]:
    """Compare predicted structure against PDB ground truth via CA RMSD."""
    gt_file = tc_config.get("ground_truth")
    chain_mapping = tc_config.get("chain_mapping", {})
    if not gt_file:
        return {"error": "No ground_truth in config"}

    gt_path = gt_root / gt_file
    if not gt_path.exists():
        return {"error": f"Ground truth not found: {gt_path}"}

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

        matched_gt, matched_pred = [], []
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
    wrapper: str,
    input_yaml: Path,
    out_dir: Path,
    seed: int,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run a single prediction with the given wrapper."""
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", "12",
        "--recycling_steps", "3",
        "--diffusion_samples", "1",
        "--override",
        "--gamma_0", "0.0",
        "--noise_scale", "1.003",
        "--matmul_precision", "high",
        "--bf16_trunk",
        "--enable_kernels",
        "--cuda_warmup",
        "--seed", str(seed),
    ]
    if extra_args:
        cmd.extend(extra_args)

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
                f"Exit code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else '(empty)'}"
            )
            return result

        predict_start = None
        predict_end = None
        for line in proc.stderr.split("\n"):
            if "[PHASE] predict_start=" in line:
                predict_start = float(line.split("=")[1])
            elif "[PHASE] predict_end=" in line:
                predict_end = float(line.split("=")[1])

        if predict_start is not None and predict_end is not None:
            result["predict_only_s"] = predict_end - predict_start

        result["quality"] = _parse_confidence(out_dir, input_yaml)
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected: {exc}"
    return result


# ---------------------------------------------------------------------------
# Modal function: single (config, test_case, seed) evaluation
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_volume, "/ground_truth": gt_volume},
)
def evaluate_single(config_name: str, tc_name: str, seed: int) -> str:
    """Run one (config, test_case, seed) and return JSON results."""
    # Find test case config
    tc_config = None
    for tc in TEST_CASES:
        if tc["name"] == tc_name:
            tc_config = tc
            break
    if tc_config is None:
        return json.dumps({"error": f"Unknown test case: {tc_name}"})

    tc_yaml = Path("/eval") / "test_cases" / f"{tc_config['yaml_name']}.yaml"
    if not tc_yaml.exists():
        return json.dumps({"error": f"YAML not found: {tc_yaml}"})

    # Setup work directory
    work_dir = Path(f"/tmp/boltz_eval/{config_name}_{tc_name}_s{seed}_{uuid.uuid4().hex[:6]}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Inject cached MSAs
    msa_cache_root = Path("/msa_cache")
    effective_yaml = tc_yaml
    use_msa_server = True
    if msa_cache_root.exists() and any(msa_cache_root.iterdir()):
        cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
        if cached_yaml is not None:
            effective_yaml = cached_yaml
            use_msa_server = False

    wrapper = f"/eval/{WRAPPER_MAP[config_name]}"
    extra_args = []
    if use_msa_server:
        extra_args.append("--use_msa_server")
    if config_name == "headprune":
        extra_args.extend(["--prune_fraction", "0.75"])

    print(f"[eval] {config_name}/{tc_name}/seed={seed} wrapper={wrapper} msa_cached={not use_msa_server}")

    pred_result = _run_prediction(wrapper, effective_yaml, work_dir, seed, extra_args)

    # Structural comparison (CA RMSD)
    structural = {}
    if pred_result["error"] is None:
        gt_root = Path("/ground_truth")
        structural = _compare_structures(work_dir, effective_yaml, gt_root, tc_config)
        if structural.get("error"):
            print(f"[eval] CA RMSD error: {structural['error']}")
        else:
            print(f"[eval] CA RMSD = {structural['ca_rmsd']}A ({structural['matched_residues']} residues)")

    result = {
        "config": config_name,
        "test_case": tc_name,
        "seed": seed,
        "predict_only_s": pred_result.get("predict_only_s"),
        "wall_time_s": pred_result.get("wall_time_s"),
        "plddt": pred_result["quality"].get("complex_plddt"),
        "iptm": pred_result["quality"].get("iptm"),
        "ca_rmsd": structural.get("ca_rmsd"),
        "matched_residues": structural.get("matched_residues"),
        "pct_within_2A": structural.get("pct_within_2A"),
        "error": pred_result.get("error") or structural.get("error"),
    }

    print(f"[eval] Result: predict_only={result['predict_only_s']}s, "
          f"pLDDT={result['plddt']}, CA_RMSD={result['ca_rmsd']}")

    return json.dumps(result)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run all 3 configs x 3 test cases x 3 seeds = 27 runs in parallel."""
    configs = ["control", "sdpa", "headprune"]
    seeds = [42, 123, 7]

    # Build all (config, tc, seed) jobs
    jobs = []
    for config_name in configs:
        for tc in TEST_CASES:
            for seed in seeds:
                jobs.append((config_name, tc["name"], seed))

    print(f"[eval-deadend] Launching {len(jobs)} parallel evaluations...")
    print(f"  Configs: {configs}")
    print(f"  Test cases: {[tc['name'] for tc in TEST_CASES]}")
    print(f"  Seeds: {seeds}")

    # Parallel execution via Modal .map()
    config_args = [j[0] for j in jobs]
    tc_args = [j[1] for j in jobs]
    seed_args = [j[2] for j in jobs]

    results = []
    for result_json in evaluate_single.map(config_args, tc_args, seed_args):
        result = json.loads(result_json)
        results.append(result)
        status = "OK" if result.get("error") is None else f"ERR: {result['error'][:80]}"
        print(f"  {result['config']}/{result['test_case']}/s{result['seed']}: {status}")

    # Organize results by config
    by_config = {}
    for r in results:
        cfg = r["config"]
        if cfg not in by_config:
            by_config[cfg] = []
        by_config[cfg].append(r)

    # Print summary tables
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    for config_name in configs:
        config_results = by_config.get(config_name, [])
        print(f"\n--- {config_name.upper()} ---")
        print(f"{'Test Case':<18} {'Seed':>5} {'Predict(s)':>10} {'pLDDT':>8} {'CA RMSD':>8} {'%<2A':>6}")
        print("-" * 60)

        for tc in TEST_CASES:
            tc_runs = [r for r in config_results if r["test_case"] == tc["name"]]
            for r in sorted(tc_runs, key=lambda x: x["seed"]):
                pred_s = f"{r['predict_only_s']:.1f}" if r.get("predict_only_s") else "N/A"
                plddt = f"{r['plddt']:.4f}" if r.get("plddt") else "ERR"
                rmsd = f"{r['ca_rmsd']:.3f}" if r.get("ca_rmsd") else "ERR"
                pct = f"{r['pct_within_2A']:.0f}%" if r.get("pct_within_2A") else "N/A"
                print(f"{tc['name']:<18} {r['seed']:>5} {pred_s:>10} {plddt:>8} {rmsd:>8} {pct:>6}")

        # Compute means
        valid_runs = [r for r in config_results if r.get("error") is None]
        if valid_runs:
            mean_pred = sum(r["predict_only_s"] for r in valid_runs if r.get("predict_only_s")) / len(valid_runs)
            mean_plddt = sum(r["plddt"] for r in valid_runs if r.get("plddt")) / len([r for r in valid_runs if r.get("plddt")])
            rmsds = [r["ca_rmsd"] for r in valid_runs if r.get("ca_rmsd")]
            mean_rmsd = sum(rmsds) / len(rmsds) if rmsds else None

            print(f"{'MEAN':<18} {'':>5} {mean_pred:>10.1f} {mean_plddt:>8.4f}", end="")
            if mean_rmsd is not None:
                print(f" {mean_rmsd:>8.3f}")
            else:
                print()

    # Cross-config comparison
    print(f"\n{'=' * 70}")
    print("CROSS-CONFIG COMPARISON (means over 3 seeds x 3 complexes)")
    print(f"{'=' * 70}")
    print(f"{'Config':<15} {'Predict(s)':>10} {'pLDDT':>8} {'CA RMSD':>8} {'Speed vs ctrl':>14}")
    print("-" * 60)

    ctrl_mean_pred = None
    for config_name in configs:
        valid = [r for r in by_config.get(config_name, []) if r.get("error") is None]
        if not valid:
            print(f"{config_name:<15} ERROR")
            continue

        preds = [r["predict_only_s"] for r in valid if r.get("predict_only_s")]
        plddts = [r["plddt"] for r in valid if r.get("plddt")]
        rmsds = [r["ca_rmsd"] for r in valid if r.get("ca_rmsd")]

        mean_pred = sum(preds) / len(preds) if preds else 0
        mean_plddt = sum(plddts) / len(plddts) if plddts else 0
        mean_rmsd = sum(rmsds) / len(rmsds) if rmsds else 0

        if config_name == "control":
            ctrl_mean_pred = mean_pred

        speed_str = ""
        if ctrl_mean_pred and mean_pred > 0:
            ratio = ctrl_mean_pred / mean_pred
            speed_str = f"{ratio:.3f}x"

        print(f"{config_name:<15} {mean_pred:>10.1f} {mean_plddt:>8.4f} {mean_rmsd:>8.3f} {speed_str:>14}")

    # Per-complex breakdown
    print(f"\n{'=' * 70}")
    print("PER-COMPLEX COMPARISON (mean over 3 seeds)")
    print(f"{'=' * 70}")

    for tc in TEST_CASES:
        print(f"\n  {tc['name']} ({tc['pdb_id']}):")
        print(f"  {'Config':<15} {'Predict(s)':>10} {'pLDDT':>8} {'CA RMSD':>8}")
        print(f"  {'-' * 45}")
        for config_name in configs:
            tc_runs = [r for r in by_config.get(config_name, [])
                       if r["test_case"] == tc["name"] and r.get("error") is None]
            if not tc_runs:
                print(f"  {config_name:<15} ERROR")
                continue
            preds = [r["predict_only_s"] for r in tc_runs if r.get("predict_only_s")]
            plddts = [r["plddt"] for r in tc_runs if r.get("plddt")]
            rmsds = [r["ca_rmsd"] for r in tc_runs if r.get("ca_rmsd")]
            mp = sum(preds) / len(preds) if preds else 0
            ml = sum(plddts) / len(plddts) if plddts else 0
            mr = sum(rmsds) / len(rmsds) if rmsds else 0
            print(f"  {config_name:<15} {mp:>10.1f} {ml:>8.4f} {mr:>8.3f}")

    # Dump raw JSON
    print("\n\n--- RAW RESULTS ---")
    print(json.dumps(results, indent=2))
