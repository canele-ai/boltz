"""Prepare CASP15 targets for Boltz evaluation.

Downloads ground truth structures from RCSB PDB, extracts protein sequences,
and creates Boltz input YAMLs for all protein targets (T* + H*).

Usage:
    python research/eval/prepare_casp15.py
    python research/eval/prepare_casp15.py --dry-run   # show targets without downloading

Output:
    research/eval/casp15/
      targets.yaml          - Target metadata (PDB IDs, chain info, residue counts)
      test_cases/           - Boltz input YAMLs
      ground_truth/         - Downloaded mmCIF files from RCSB
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# CASP15 target → PDB mapping (protein targets only, RNA excluded)
#
# Sources: predictioncenter.org/casp15/targetlist.cgi
#          predictioncenter.org/casp15/domains_summary.cgi
# ---------------------------------------------------------------------------

CASP15_TARGET_PDB: dict[str, str] = {
    # T* tertiary targets
    "T1104": "7roa",
    "T1106": "7qih",
    "T1112": "8ork",
    "T1113": "7uyx",
    "T1114": "7utd",
    "T1119": "7sq4",
    "T1120": "7qvb",
    "T1121": "7til",
    "T1122": "8bbt",
    "T1123": "7uzt",
    "T1124": "7ux8",
    "T1125": "8h2n",
    "T1129": "8a8c",
    "T1132": "8ecx",
    "T1133": "8dys",
    "T1134": "7ubz",
    "T1137": "8fef",
    "T1145": "7uww",
    "T1147": "8em5",
    "T1151": "8d5v",
    "T1152": "7r1l",
    "T1154": "7zcx",
    "T1155": "8pbv",
    "T1157": "8pko",
    "T1158": "8sxa",
    "T1159": "7pzt",
    "T1160": "8jvn",
    "T1161": "8jvp",
    "T1173": "8on4",
    "T1174": "8ond",
    "T1176": "8smq",
    "T1178": "8ufn",
    "T1179": "8tn8",
    "T1183": "8ifx",
    "T1185": "8ouy",
    "T1187": "8ad2",
    "T1188": "8c6z",
    "T1194": "8okh",
    # H* hetero-oligomer targets (share PDB with corresponding T* target)
    "H1106": "7qih",
    "H1134": "7ubz",
    "H1151": "8d5v",
    "H1157": "8pko",
    "H1185": "8ouy",
}

# Targets from casp15_ids.txt that are skipped:
# - R* targets (RNA): R1117, R1149, R1156, R1107, R1128, R1108, R1116, R1136, R1117v2
# - T1118v1: protein-ligand target with no deposited PDB structure
# - Domain segments (T1137s1-s9, etc.): folded into parent target
# - Version variants (T1158v1-v4): same PDB as parent T1158

# Three-letter to one-letter amino acid code mapping
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    # Common modified residues mapped to standard
    "MSE": "M",  # selenomethionine
    "SEC": "C",  # selenocysteine
    "CSE": "C",  # selenocysteine variant
    "HYP": "P",  # hydroxyproline
    "TPO": "T",  # phosphothreonine
    "SEP": "S",  # phosphoserine
    "PTR": "Y",  # phosphotyrosine
}


def extract_protein_chains(cif_path: Path) -> list[dict]:
    """Extract protein chain sequences from an mmCIF file using BioPython.

    Returns list of dicts: [{"chain_id": "A", "sequence": "MKVL...", "num_residues": 123}, ...]
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_path))
    model = structure[0]

    chains = []
    for chain in model:
        residues = []
        for residue in chain:
            # Skip heteroatoms (water, ligands) but keep standard + modified residues
            hetflag = residue.id[0]
            if hetflag == "W":
                continue  # water
            resname = residue.resname.strip()
            if hetflag == " " or resname in AA3_TO_1:
                aa = AA3_TO_1.get(resname)
                if aa:
                    residues.append(aa)

        if len(residues) >= 10:  # skip very short chains (likely artifacts)
            chains.append({
                "chain_id": chain.id,
                "sequence": "".join(residues),
                "num_residues": len(residues),
            })

    return chains


def create_boltz_yaml(chains: list[dict]) -> dict:
    """Create a Boltz input YAML dict from extracted chain info."""
    sequences = []
    for chain in chains:
        sequences.append({
            "protein": {
                "id": chain["chain_id"],
                "sequence": chain["sequence"],
            }
        })
    return {"version": 1, "sequences": sequences}


def create_chain_mapping(chains: list[dict]) -> dict[str, str]:
    """Create identity chain mapping for RMSD computation."""
    return {c["chain_id"]: c["chain_id"] for c in chains}


def download_cif(pdb_id: str, dest: Path) -> bool:
    """Download mmCIF from RCSB PDB. Returns True on success."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  ERROR downloading {pdb_id}: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CASP15 targets for Boltz evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Show targets without downloading")
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent
    casp15_dir = eval_dir / "casp15"
    test_cases_dir = casp15_dir / "test_cases"
    gt_dir = casp15_dir / "ground_truth"

    # Deduplicate: multiple CASP targets may share the same PDB
    # Group by PDB to avoid redundant downloads/predictions
    pdb_to_casp: dict[str, list[str]] = {}
    for casp_id, pdb_id in CASP15_TARGET_PDB.items():
        pdb_to_casp.setdefault(pdb_id, []).append(casp_id)

    unique_pdbs = sorted(pdb_to_casp.keys())
    print(f"CASP15 protein targets: {len(CASP15_TARGET_PDB)} CASP IDs → {len(unique_pdbs)} unique PDB structures")

    if args.dry_run:
        for pdb_id in unique_pdbs:
            casp_ids = pdb_to_casp[pdb_id]
            print(f"  {pdb_id.upper()} ← {', '.join(casp_ids)}")
        return

    # Create directories
    test_cases_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Process each unique PDB structure
    targets = []
    skipped = []

    for pdb_id in unique_pdbs:
        casp_ids = pdb_to_casp[pdb_id]
        pdb_upper = pdb_id.upper()
        # Use primary CASP target ID as the name (prefer T* over H*)
        primary_id = sorted(casp_ids, key=lambda x: (x.startswith("H"), x))[0]

        print(f"\n[{primary_id}] PDB {pdb_upper} ({', '.join(casp_ids)})")

        # Download ground truth
        gt_path = gt_dir / f"{pdb_upper}.cif"
        if gt_path.exists():
            print(f"  Ground truth already exists: {gt_path.name}")
        else:
            print(f"  Downloading {pdb_upper}.cif from RCSB...")
            if not download_cif(pdb_upper, gt_path):
                skipped.append({"pdb_id": pdb_upper, "casp_ids": casp_ids, "reason": "download failed"})
                continue

        # Extract protein chains
        try:
            chains = extract_protein_chains(gt_path)
        except Exception as e:
            print(f"  ERROR parsing {pdb_upper}: {e}")
            skipped.append({"pdb_id": pdb_upper, "casp_ids": casp_ids, "reason": f"parse error: {e}"})
            continue

        if not chains:
            print(f"  WARNING: No protein chains found in {pdb_upper}, skipping")
            skipped.append({"pdb_id": pdb_upper, "casp_ids": casp_ids, "reason": "no protein chains"})
            continue

        total_residues = sum(c["num_residues"] for c in chains)
        chain_ids = [c["chain_id"] for c in chains]
        print(f"  Chains: {', '.join(chain_ids)} ({total_residues} residues total)")

        # Create Boltz input YAML
        boltz_yaml = create_boltz_yaml(chains)
        yaml_path = test_cases_dir / f"{primary_id}.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(boltz_yaml, f, default_flow_style=False)
        print(f"  Created {yaml_path.name}")

        # Record target metadata
        target_entry = {
            "name": primary_id,
            "casp_ids": casp_ids,
            "pdb_id": pdb_upper,
            "yaml": f"test_cases/{primary_id}.yaml",
            "ground_truth": f"ground_truth/{pdb_upper}.cif",
            "chain_mapping": create_chain_mapping(chains),
            "num_chains": len(chains),
            "total_residues": total_residues,
            "chain_details": [
                {"id": c["chain_id"], "residues": c["num_residues"]}
                for c in chains
            ],
        }
        targets.append(target_entry)

    # Sort targets by residue count for reporting
    targets.sort(key=lambda t: t["total_residues"])

    # Write targets.yaml
    config = {
        "description": "CASP15 protein targets for Boltz-2 validation",
        "num_targets": len(targets),
        "num_skipped": len(skipped),
        "targets": targets,
    }
    if skipped:
        config["skipped"] = skipped

    config_path = casp15_dir / "targets.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"CASP15 Preparation Complete")
    print(f"{'='*60}")
    print(f"Targets prepared: {len(targets)}")
    print(f"Targets skipped:  {len(skipped)}")
    print(f"Total residues:   {sum(t['total_residues'] for t in targets)}")
    if targets:
        print(f"Size range:       {targets[0]['total_residues']}–{targets[-1]['total_residues']} residues")
    print(f"\nOutput:")
    print(f"  Config:       {config_path}")
    print(f"  Test cases:   {test_cases_dir}/")
    print(f"  Ground truth: {gt_dir}/")

    if skipped:
        print(f"\nSkipped targets:")
        for s in skipped:
            print(f"  {s['pdb_id']} ({', '.join(s['casp_ids'])}): {s['reason']}")


if __name__ == "__main__":
    main()
