"""Download PDB ground truth structures and generate MSA cache for eval-v5.

Run once to set up the evaluation infrastructure:
    modal run research/eval/setup_ground_truth.py

This:
1. Downloads ground truth mmCIF files from RCSB PDB
2. Saves them locally to research/eval/ground_truth/
3. Generates MSA cache for the new test cases on Modal volume
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent

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
)

app = modal.App("boltz-eval-setup", image=boltz_image)
msa_volume = modal.Volume.from_name("boltz-msa-cache-v5", create_if_missing=True)
gt_volume = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=True)


PDB_IDS = ["1BRS", "1DQJ", "2DN2"]


@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_volume, "/ground_truth": gt_volume},
)
def setup_all() -> str:
    """Download ground truth and generate MSA cache."""
    import yaml
    import urllib.request

    results = {"ground_truth": {}, "msa_cache": {}}

    # --- Step 1: Download ground truth mmCIF from RCSB ---
    print("[setup] Downloading ground truth structures from RCSB PDB...")
    for pdb_id in PDB_IDS:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        local_path = Path(f"/ground_truth/{pdb_id}.cif")
        gt_local = Path(f"/eval/ground_truth/{pdb_id}.cif")

        if local_path.exists():
            print(f"[setup] {pdb_id}.cif already cached on volume")
            results["ground_truth"][pdb_id] = "cached"
        else:
            print(f"[setup] Downloading {url}...")
            urllib.request.urlretrieve(url, str(local_path))
            print(f"[setup] Saved {pdb_id}.cif ({local_path.stat().st_size / 1024:.0f} KB)")
            results["ground_truth"][pdb_id] = "downloaded"

    gt_volume.commit()

    # --- Step 2: Generate MSA cache for each test case ---
    print("\n[setup] Generating MSA cache for test cases...")

    # Download boltz model files (needed for process_inputs)
    cache = Path.home() / ".boltz"
    cache.mkdir(parents=True, exist_ok=True)

    import boltz.main as boltz_main
    boltz_main.download_boltz2(cache)

    test_cases_dir = Path("/eval/test_cases")
    for tc_yaml in sorted(test_cases_dir.glob("*.yaml")):
        tc_name = tc_yaml.stem
        cache_dir = Path(f"/msa_cache/{tc_name}")

        if cache_dir.exists() and any(cache_dir.glob("*.csv")):
            print(f"[setup] MSA cache for {tc_name} already exists")
            results["msa_cache"][tc_name] = "cached"
            continue

        print(f"[setup] Generating MSAs for {tc_name}...")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use boltz's process_inputs to generate MSAs via ColabFold server
        work_dir = Path(f"/tmp/msa_gen/{tc_name}")
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = boltz_main.check_inputs(tc_yaml)
            boltz_main.process_inputs(
                data=data,
                out_dir=work_dir,
                ccd_path=cache / "ccd.pkl",
                mol_dir=cache / "mols",
                use_msa_server=True,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy",
                boltz2=True,
                preprocessing_threads=1,
                max_msa_seqs=8192,
            )

            # Copy generated MSA files to the cache volume
            msa_dir = work_dir / "processed" / "msa"
            if msa_dir.exists():
                for msa_file in msa_dir.glob("*.csv"):
                    dest = cache_dir / f"{tc_name}_{msa_file.stem}.csv"
                    shutil.copy2(msa_file, dest)
                    print(f"[setup] Cached MSA: {dest.name}")
                results["msa_cache"][tc_name] = "generated"
            else:
                results["msa_cache"][tc_name] = "no MSA files found"

        except Exception as e:
            print(f"[setup] ERROR generating MSAs for {tc_name}: {e}")
            results["msa_cache"][tc_name] = f"error: {e}"

    msa_volume.commit()

    # --- Step 3: Validate ground truth structures ---
    print("\n[setup] Validating ground truth structures...")
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)

    for pdb_id in PDB_IDS:
        gt_path = Path(f"/ground_truth/{pdb_id}.cif")
        if gt_path.exists():
            structure = parser.get_structure(pdb_id, str(gt_path))
            chains = list(structure[0].get_chains())
            total_residues = sum(
                len([r for r in chain.get_residues() if r.id[0] == " "])
                for chain in chains
            )
            chain_ids = [c.id for c in chains]
            print(f"[setup] {pdb_id}: {len(chains)} chains ({', '.join(chain_ids)}), "
                  f"{total_residues} residues")
            results["ground_truth"][f"{pdb_id}_chains"] = chain_ids
            results["ground_truth"][f"{pdb_id}_residues"] = total_residues

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print("[setup] Setting up eval-v5 ground truth and MSA cache...")
    result_json = setup_all.remote()
    result = json.loads(result_json)
    print("\n" + "=" * 60)
    print("SETUP RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    # Also save ground truth files locally
    print("\n[setup] Note: Ground truth files are on Modal volume 'boltz-ground-truth-v1'.")
    print("[setup] To download locally: modal volume get boltz-ground-truth-v1 /")
