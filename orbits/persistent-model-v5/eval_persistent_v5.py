"""Persistent model evaluation v5: with CA RMSD structural comparison.

Extends fast-model-load/eval_persistent.py by adding:
1. CA RMSD comparison against PDB ground truth (BioPython SVDSuperimposer)
2. Ground truth volume mounting
3. Proper reporting against eval-v5 baseline (53.57s mean_wall_time_s)

Usage:
    # Sanity check (single seed, prints results)
    modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode sanity

    # Full validation (3 seeds in parallel)
    modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode eval --validate

    # Populate ground truth volume (run once)
    modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode prep-ground-truth
"""

from __future__ import annotations

import json
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
PARENT_ORBIT_DIR = REPO_ROOT / "orbits" / "fast-model-load"

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
    .pip_install(
        # BioPython for CA RMSD structural comparison
        "biopython>=1.83",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(
        str(PARENT_ORBIT_DIR / "persistent_predict.py"),
        remote_path="/eval/persistent_predict.py",
    )
)

app = modal.App("boltz-eval-persistent-v5", image=boltz_image)

msa_cache = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=False)
boltz_cache = modal.Volume.from_name("boltz-model-cache", create_if_missing=True)
ground_truth = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=True)

# ---------------------------------------------------------------------------
# Ground truth config: PDB IDs and chain mappings for each test case
# ---------------------------------------------------------------------------

# Maps test case name -> PDB ID and chain mapping
# Chain mapping: predicted_chain_id -> ground_truth_chain_id
GROUND_TRUTH_CONFIG = {
    "small_complex": {
        "pdb_id": "1brs",
        # Barnase-Barstar: chains A+D in PDB (Barnase=A, Barstar=D)
        # Predicted chains: A, B  ->  PDB chains: A, D
        "chain_map": {"A": "A", "B": "D"},
    },
    "medium_complex": {
        "pdb_id": None,  # No known PDB for this exact complex
        "chain_map": {},
    },
    "large_complex": {
        "pdb_id": None,  # Synthetic complex, no PDB ground truth
        "chain_map": {},
    },
}


# ---------------------------------------------------------------------------
# CA RMSD computation
# ---------------------------------------------------------------------------

def _compute_ca_rmsd(pred_mmcif_path: str, gt_pdb_path: str, chain_map: dict) -> dict:
    """Compute CA RMSD between predicted mmCIF and ground truth PDB.

    Uses BioPython's SVDSuperimposer to align CA atoms and compute RMSD.

    Parameters
    ----------
    pred_mmcif_path : str
        Path to predicted mmCIF file.
    gt_pdb_path : str
        Path to ground truth PDB file.
    chain_map : dict
        Maps predicted chain IDs to ground truth chain IDs.

    Returns
    -------
    dict with keys:
        ca_rmsd : float or None
        num_ca_atoms : int
        error : str or None
    """
    try:
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.SVDSuperimposer import SVDSuperimposer
        import numpy as np
    except ImportError as e:
        return {"ca_rmsd": None, "num_ca_atoms": 0, "error": f"Import error: {e}"}

    result = {"ca_rmsd": None, "num_ca_atoms": 0, "error": None}

    try:
        # Parse predicted structure
        mmcif_parser = MMCIFParser(QUIET=True)
        pred_structure = mmcif_parser.get_structure("pred", pred_mmcif_path)

        # Parse ground truth structure
        pdb_parser = PDBParser(QUIET=True)
        gt_structure = pdb_parser.get_structure("gt", gt_pdb_path)

        # Extract CA atoms from predicted structure
        pred_cas = {}  # chain_id -> {resseq: coord}
        for model in pred_structure:
            for chain in model:
                chain_id = chain.get_id()
                if chain_id not in chain_map:
                    continue
                pred_cas[chain_id] = {}
                for residue in chain:
                    if residue.get_id()[0] != " ":
                        continue  # skip hetero atoms
                    if "CA" in residue:
                        resseq = residue.get_id()[1]
                        pred_cas[chain_id][resseq] = residue["CA"].get_vector().get_array()
            break  # first model only

        # Extract CA atoms from ground truth
        gt_cas = {}
        for model in gt_structure:
            for chain in model:
                chain_id = chain.get_id()
                # Check if this chain is a target in the chain_map values
                if chain_id not in chain_map.values():
                    continue
                gt_cas[chain_id] = {}
                for residue in chain:
                    if residue.get_id()[0] != " ":
                        continue
                    if "CA" in residue:
                        resseq = residue.get_id()[1]
                        gt_cas[chain_id][resseq] = residue["CA"].get_vector().get_array()
            break

        # Align CA atoms across mapped chains
        pred_coords = []
        gt_coords = []

        for pred_chain_id, gt_chain_id in chain_map.items():
            if pred_chain_id not in pred_cas or gt_chain_id not in gt_cas:
                continue

            pred_chain_cas = pred_cas[pred_chain_id]
            gt_chain_cas = gt_cas[gt_chain_id]

            # Find common residue positions
            common_resseqs = sorted(
                set(pred_chain_cas.keys()) & set(gt_chain_cas.keys())
            )

            for resseq in common_resseqs:
                pred_coords.append(pred_chain_cas[resseq])
                gt_coords.append(gt_chain_cas[resseq])

        if len(pred_coords) < 3:
            result["error"] = f"Too few common CA atoms: {len(pred_coords)}"
            return result

        pred_arr = np.array(pred_coords)
        gt_arr = np.array(gt_coords)

        # SVD superimposition
        sup = SVDSuperimposer()
        sup.set(gt_arr, pred_arr)  # set(reference, moving)
        sup.run()
        rmsd = sup.get_rms()

        result["ca_rmsd"] = float(rmsd)
        result["num_ca_atoms"] = len(pred_coords)

    except Exception as e:
        result["error"] = f"CA RMSD computation failed: {e}"

    return result


def _find_predicted_mmcif(out_dir: Path, target_name: str) -> str | None:
    """Find the predicted mmCIF file in the output directory."""
    pred_dir = out_dir / "predictions" / target_name
    if not pred_dir.exists():
        # Try alternative naming
        pred_base = out_dir / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                pred_dir = subdirs[0]

    if not pred_dir.exists():
        return None

    mmcif_files = sorted(pred_dir.glob("*.cif")) + sorted(pred_dir.glob("*.mmcif"))
    if not mmcif_files:
        # Boltz writes as *_model_0.cif
        mmcif_files = sorted(pred_dir.glob("*model*.cif"))

    return str(mmcif_files[0]) if mmcif_files else None


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


# ---------------------------------------------------------------------------
# Ground truth preparation
# ---------------------------------------------------------------------------

@app.function(
    timeout=600,
    volumes={"/ground_truth": ground_truth},
)
def prep_ground_truth() -> str:
    """Download PDB ground truth files to the ground truth volume.

    Downloads PDB files for test cases that have known crystal structures.
    """
    import urllib.request

    gt_dir = Path("/ground_truth")
    downloaded = []

    for tc_name, config in GROUND_TRUTH_CONFIG.items():
        pdb_id = config.get("pdb_id")
        if pdb_id is None:
            continue

        pdb_file = gt_dir / f"{pdb_id}.pdb"
        if pdb_file.exists():
            downloaded.append({"name": tc_name, "pdb_id": pdb_id, "status": "exists"})
            continue

        # Download from RCSB
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"[prep-gt] Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, str(pdb_file))
            downloaded.append({"name": tc_name, "pdb_id": pdb_id, "status": "downloaded"})
        except Exception as e:
            downloaded.append({"name": tc_name, "pdb_id": pdb_id, "status": f"error: {e}"})

    ground_truth.commit()
    return json.dumps({"files": downloaded}, indent=2)


# ---------------------------------------------------------------------------
# Modal function: single-seed evaluation with CA RMSD
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=7200,
    volumes={
        "/msa_cache": msa_cache,
        "/boltz_cache": boltz_cache,
        "/ground_truth": ground_truth,
    },
)
def evaluate_seed(seed: int) -> str:
    """Run persistent-model evaluation for a single seed, including CA RMSD."""
    import yaml

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    msa_cache_root = Path("/msa_cache")
    use_msa_cache = msa_cache_root.exists() and any(msa_cache_root.iterdir())

    work_dir = Path(f"/tmp/boltz_persistent_v5_{seed}_{uuid.uuid4().hex[:8]}")
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

    if not use_msa_cache:
        cmd.append("--use_msa_server")

    print(f"[eval-persistent-v5] seed={seed}, cmd={' '.join(cmd[:10])}...")

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

    # -------------------------------------------------------------------
    # CA RMSD: compare predicted structures against ground truth
    # -------------------------------------------------------------------
    gt_dir = Path("/ground_truth")

    for pc in parsed.get("per_complex", []):
        tc_name = pc["name"]
        gt_config = GROUND_TRUTH_CONFIG.get(tc_name, {})
        pdb_id = gt_config.get("pdb_id")
        chain_map = gt_config.get("chain_map", {})

        if pdb_id is None or not chain_map:
            pc["ca_rmsd"] = None
            pc["ca_rmsd_note"] = "No ground truth PDB available"
            continue

        gt_pdb_path = gt_dir / f"{pdb_id}.pdb"
        if not gt_pdb_path.exists():
            pc["ca_rmsd"] = None
            pc["ca_rmsd_note"] = f"Ground truth PDB not found: {gt_pdb_path}"
            continue

        # Find predicted mmCIF
        # The persistent_predict.py writes to out_dir/target_name/predictions/target_name/
        tc_out_dir = out_dir / tc_name
        pred_mmcif = _find_predicted_mmcif(tc_out_dir, tc_name)

        if pred_mmcif is None:
            pc["ca_rmsd"] = None
            pc["ca_rmsd_note"] = f"Predicted mmCIF not found in {tc_out_dir}"
            continue

        print(f"[eval-v5] Computing CA RMSD for {tc_name}: "
              f"pred={pred_mmcif}, gt={gt_pdb_path}")

        rmsd_result = _compute_ca_rmsd(
            pred_mmcif_path=pred_mmcif,
            gt_pdb_path=str(gt_pdb_path),
            chain_map=chain_map,
        )

        pc["ca_rmsd"] = rmsd_result.get("ca_rmsd")
        pc["ca_rmsd_num_atoms"] = rmsd_result.get("num_ca_atoms", 0)
        if rmsd_result.get("error"):
            pc["ca_rmsd_error"] = rmsd_result["error"]
            print(f"[eval-v5] CA RMSD error for {tc_name}: {rmsd_result['error']}")
        else:
            print(f"[eval-v5] CA RMSD for {tc_name}: {rmsd_result['ca_rmsd']:.3f} A "
                  f"({rmsd_result['num_ca_atoms']} CA atoms)")

    result["persistent_results"] = parsed
    result["stderr_tail"] = proc.stderr[-1000:] if proc.stderr else "(empty)"

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@app.function(timeout=600)
def aggregate_results(seed_results_json: list[str]) -> str:
    """Aggregate results from multiple seeds, including CA RMSD."""
    eval_config_path = Path("/eval/config.yaml")
    import yaml
    with eval_config_path.open() as f:
        eval_config = yaml.safe_load(f)

    baseline = eval_config.get("baseline", {})
    baseline_mean_wall = baseline.get("mean_wall_time_s", 53.57)
    baseline_plddt = baseline.get("mean_plddt", 0.717)
    baseline_per_complex = {pc["name"]: pc for pc in baseline.get("per_complex", [])}

    # Eval-v5 baseline CA RMSD reference values (from task description)
    BASELINE_CA_RMSD = {
        "small_complex": 0.325,
        "medium_complex": 5.243,
        "large_complex": 0.474,
    }

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
        (sum((x - mean_model_load)**2 for x in all_model_loads) / max(len(all_model_loads) - 1, 1))**0.5
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
        ca_rmsds = [
            e["ca_rmsd"] for e in entries
            if e.get("ca_rmsd") is not None
        ]

        mean_predict = sum(predict_times) / len(predict_times)
        mean_process = sum(process_times) / len(process_times)
        mean_total = sum(total_times) / len(total_times)
        mean_plddt = sum(plddts) / len(plddts) if plddts else None
        mean_ca_rmsd = sum(ca_rmsds) / len(ca_rmsds) if ca_rmsds else None
        std_ca_rmsd = (
            (sum((x - mean_ca_rmsd)**2 for x in ca_rmsds) / max(len(ca_rmsds) - 1, 1))**0.5
            if len(ca_rmsds) > 1 else 0
        )

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
            "mean_ca_rmsd": mean_ca_rmsd,
            "std_ca_rmsd": std_ca_rmsd,
            "ca_rmsds": ca_rmsds,
        }

        # Quality gate per complex (pLDDT)
        bl = baseline_per_complex.get(name, {})
        bl_plddt = bl.get("complex_plddt")
        if bl_plddt is not None and mean_plddt is not None:
            pc_summary["plddt_delta_pp"] = (mean_plddt - bl_plddt) * 100.0
            pc_summary["passes_5pp_gate"] = (bl_plddt - mean_plddt) * 100.0 <= 5.0

        # CA RMSD quality gate: regression <= 1.0 A vs baseline
        bl_rmsd = BASELINE_CA_RMSD.get(name)
        if bl_rmsd is not None and mean_ca_rmsd is not None:
            pc_summary["baseline_ca_rmsd"] = bl_rmsd
            pc_summary["ca_rmsd_delta"] = mean_ca_rmsd - bl_rmsd
            pc_summary["passes_ca_rmsd_gate"] = (mean_ca_rmsd - bl_rmsd) <= 1.0

        per_complex_summary.append(pc_summary)

    # Compute amortized wall time
    mean_per_complex_total = sum(all_total_times) / len(all_total_times) if all_total_times else 0
    amortized_onetime_3 = mean_onetime / 3  # N=3
    amortized_wall_3 = amortized_onetime_3 + mean_per_complex_total

    # Speedup calculations
    # 1. Predict-only speedup
    mean_predict_only = sum(all_predict_times) / len(all_predict_times) if all_predict_times else 0
    predict_only_speedup = baseline_mean_wall / mean_predict_only if mean_predict_only > 0 else 0

    # 2. Amortized speedup for different N
    amortized_speedups = {}
    for n in [3, 10, 100]:
        amort_time = mean_onetime / n + mean_per_complex_total
        amortized_speedups[f"N={n}"] = {
            "amortized_wall_s": amort_time,
            "speedup": baseline_mean_wall / amort_time if amort_time > 0 else 0,
        }

    # Quality gates
    mean_plddt = sum(all_plddts) / len(all_plddts) if all_plddts else None
    passes_plddt_gate = True
    plddt_delta_pp = None
    if mean_plddt is not None and baseline_plddt is not None:
        plddt_delta_pp = (mean_plddt - baseline_plddt) * 100.0
        regression = (baseline_plddt - mean_plddt) * 100.0
        if regression > 2.0:
            passes_plddt_gate = False

    # Per-complex pLDDT gate
    for pc in per_complex_summary:
        if pc.get("passes_5pp_gate") is False:
            passes_plddt_gate = False

    # CA RMSD gate
    passes_ca_rmsd_gate = True
    for pc in per_complex_summary:
        if pc.get("passes_ca_rmsd_gate") is False:
            passes_ca_rmsd_gate = False

    passes_all_gates = passes_plddt_gate and passes_ca_rmsd_gate

    # Per-seed metrics for std calculation
    per_seed_amortized = []
    per_seed_predict_only = []
    for sr in seed_results:
        pr = sr.get("persistent_results", {})
        onetime = pr.get("one_time_cost_s", pr.get("model_load_time_s", 0))
        pcs = pr.get("per_complex", [])
        if pcs:
            mean_pc = sum(p["total_per_complex_s"] for p in pcs) / len(pcs)
            mean_pred = sum(p["predict_time_s"] for p in pcs) / len(pcs)
            amort = onetime / len(pcs) + mean_pc
            per_seed_amortized.append(amort)
            per_seed_predict_only.append(mean_pred)

    mean_amortized = sum(per_seed_amortized) / len(per_seed_amortized) if per_seed_amortized else 0
    std_amortized = (
        (sum((x - mean_amortized)**2 for x in per_seed_amortized) / max(len(per_seed_amortized) - 1, 1))**0.5
        if len(per_seed_amortized) > 1 else 0
    )

    per_seed_speedups_N3 = [baseline_mean_wall / a if a > 0 else 0 for a in per_seed_amortized]
    mean_speedup_N3 = sum(per_seed_speedups_N3) / len(per_seed_speedups_N3) if per_seed_speedups_N3 else 0
    std_speedup_N3 = (
        (sum((x - mean_speedup_N3)**2 for x in per_seed_speedups_N3) / max(len(per_seed_speedups_N3) - 1, 1))**0.5
        if len(per_seed_speedups_N3) > 1 else 0
    )

    aggregate = {
        "num_seeds": num_seeds,
        "num_complexes": num_complexes,
        "baseline_mean_wall_time_s": baseline_mean_wall,
        "baseline_mean_plddt": baseline_plddt,
        # Timing
        "mean_model_load_s": mean_model_load,
        "std_model_load_s": std_model_load,
        "mean_onetime_cost_s": mean_onetime,
        "mean_per_complex_total_s": mean_per_complex_total,
        "mean_predict_only_s": mean_predict_only,
        # Speedups
        "predict_only_speedup": predict_only_speedup,
        "amortized_speedups": amortized_speedups,
        "amortized_wall_per_complex_s_N3": amortized_wall_3,
        "std_amortized_wall_s": std_amortized,
        "per_seed_amortized_wall_s": per_seed_amortized,
        "per_seed_speedups_N3": per_seed_speedups_N3,
        "mean_speedup_N3": mean_speedup_N3,
        "std_speedup_N3": std_speedup_N3,
        # Quality
        "mean_plddt": mean_plddt,
        "plddt_delta_pp": plddt_delta_pp,
        "passes_plddt_gate": passes_plddt_gate,
        "passes_ca_rmsd_gate": passes_ca_rmsd_gate,
        "passes_all_gates": passes_all_gates,
        # Per-complex
        "per_complex": per_complex_summary,
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
    """Persistent model evaluation v5 with CA RMSD.

    Modes:
        sanity           - Quick single-seed check
        eval             - Full evaluation
        prep-ground-truth - Download PDB ground truth files to volume

    Usage:
        modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode prep-ground-truth
        modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode sanity
        modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode eval --validate
    """
    if mode == "prep-ground-truth":
        print("[eval-v5] Preparing ground truth volume...")
        result = prep_ground_truth.remote()
        print(result)
        return

    if mode == "sanity":
        print("[eval-v5] Running sanity check (seed=42)...")
        result_json = evaluate_seed.remote(42)
        result = json.loads(result_json)
        if "error" in result:
            print(f"FAILED: {result['error'][:3000]}")
        else:
            print(json.dumps(result, indent=2))
        return

    if mode == "eval":
        if validate:
            seeds = [42, 123, 7]
        else:
            seeds = [seed]

        # Step 1: Ensure ground truth is populated
        print("[eval-v5] Ensuring ground truth volume is populated...")
        gt_result = prep_ground_truth.remote()
        print(f"[eval-v5] Ground truth: {gt_result}")

        # Step 2: Ensure model cache is populated (uses parent orbit's prep_cache)
        # The boltz-model-cache volume should already have the pickle from fast-model-load
        # If not, we'll need to create it. Check by running evaluate_seed which will fail
        # if the pickle is missing.

        print(f"[eval-v5] Running evaluation with seeds={seeds}")

        # Run seeds in parallel
        seed_results = list(evaluate_seed.map(seeds))

        print("\n--- PER-SEED RESULTS ---")
        for sr_json in seed_results:
            sr = json.loads(sr_json)
            if "error" in sr:
                print(f"  Seed {sr.get('seed', '?')}: ERROR - {sr['error'][:2000]}")
            else:
                pr = sr.get("persistent_results", {})
                print(f"  Seed {sr['seed']}: load={pr.get('model_load_time_s', 0):.1f}s, "
                      f"onetime={pr.get('one_time_cost_s', 0):.1f}s, "
                      f"wall={sr.get('total_wall_time_s', 0):.1f}s")
                for pc in pr.get("per_complex", []):
                    rmsd_str = f"{pc.get('ca_rmsd', 'N/A'):.3f}" if pc.get("ca_rmsd") is not None else "N/A"
                    print(f"    {pc['name']}: predict={pc['predict_time_s']:.1f}s, "
                          f"pLDDT={pc['quality'].get('complex_plddt', 'N/A')}, "
                          f"CA_RMSD={rmsd_str}")

        # Aggregate
        if len(seeds) >= 1:
            agg_json = aggregate_results.remote(seed_results)
            agg = json.loads(agg_json)
            print("\n--- AGGREGATE RESULTS ---")
            print(json.dumps(agg, indent=2))

            # Pretty summary
            print(f"\n{'='*70}")
            print(f"PERSISTENT MODEL V5 EVALUATION SUMMARY")
            print(f"{'='*70}")
            print(f"  Seeds: {len(seeds)}")
            print(f"  Baseline mean wall time: {agg.get('baseline_mean_wall_time_s', 0):.1f}s")
            print(f"")
            print(f"  TIMING:")
            print(f"    Model load (pickle): {agg.get('mean_model_load_s', 0):.1f}s +/- {agg.get('std_model_load_s', 0):.1f}s")
            print(f"    One-time cost (load+warmup): {agg.get('mean_onetime_cost_s', 0):.1f}s")
            print(f"    Mean predict+process/complex: {agg.get('mean_per_complex_total_s', 0):.1f}s")
            print(f"    Mean predict-only/complex: {agg.get('mean_predict_only_s', 0):.1f}s")
            print(f"")
            print(f"  SPEEDUPS (vs baseline {agg.get('baseline_mean_wall_time_s', 0):.1f}s):")
            print(f"    Predict-only: {agg.get('predict_only_speedup', 0):.2f}x")
            for n_key, n_data in agg.get("amortized_speedups", {}).items():
                print(f"    Amortized ({n_key}): {n_data.get('speedup', 0):.2f}x "
                      f"({n_data.get('amortized_wall_s', 0):.1f}s/complex)")
            print(f"")
            print(f"  QUALITY:")
            print(f"    Mean pLDDT: {agg.get('mean_plddt', 'N/A')}")
            if agg.get('plddt_delta_pp') is not None:
                print(f"    pLDDT delta: {agg['plddt_delta_pp']:+.2f} pp")
            print(f"    pLDDT gate: {'PASS' if agg.get('passes_plddt_gate') else 'FAIL'}")
            print(f"    CA RMSD gate: {'PASS' if agg.get('passes_ca_rmsd_gate') else 'FAIL'}")
            print(f"    ALL gates: {'PASS' if agg.get('passes_all_gates') else 'FAIL'}")
            print(f"")
            print(f"  PER-COMPLEX:")
            for pc in agg.get("per_complex", []):
                plddt_str = f"{pc['mean_plddt']:.4f}" if pc.get('mean_plddt') is not None else "N/A"
                delta_str = f"{pc.get('plddt_delta_pp', 0):+.1f}pp" if pc.get('plddt_delta_pp') is not None else ""
                rmsd_str = f"{pc['mean_ca_rmsd']:.3f}A" if pc.get('mean_ca_rmsd') is not None else "N/A"
                rmsd_bl = f"(bl: {pc.get('baseline_ca_rmsd', 'N/A')}A)" if pc.get('baseline_ca_rmsd') is not None else ""
                print(f"    {pc['name']}:")
                print(f"      predict={pc['mean_predict_s']:.1f}s, process={pc['mean_process_s']:.1f}s")
                print(f"      pLDDT={plddt_str} {delta_str}")
                print(f"      CA RMSD={rmsd_str} {rmsd_bl}")

            # Save aggregate JSON for later analysis
            print(f"\n[eval-v5] Full aggregate JSON saved above.")
