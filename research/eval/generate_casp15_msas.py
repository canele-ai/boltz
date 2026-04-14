"""Generate and store pre-computed MSA files for CASP15 targets.

Runs Boltz's MSA pipeline (ColabFold) for each CASP15 target and stores
the results on a Modal persistent volume. This eliminates MSA server latency
from all subsequent eval runs.

Usage:
    modal run research/eval/generate_casp15_msas.py                # Generate all
    modal run research/eval/generate_casp15_msas.py --verify-only  # Check cache
    modal run research/eval/generate_casp15_msas.py --max-targets 5 # Generate first 5 only
"""
from __future__ import annotations

import json
import shutil
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
    # No cuequivariance needed — MSA generation is CPU + network only.
    .add_local_dir(str(EVAL_DIR / "casp15"), remote_path="/casp15")
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
)

app = modal.App("boltz-casp15-msa-gen", image=boltz_image)
msa_volume = modal.Volume.from_name("boltz-msa-cache-casp15", create_if_missing=True)


@app.function(
    # No GPU needed — MSA generation is CPU + network (ColabFold API calls).
    timeout=14400,  # 4 hours — ColabFold rate limiting makes this slow
    volumes={"/msa_cache": msa_volume},
)
def generate_msas(max_targets: int = 0) -> str:
    """Generate MSA cache for all CASP15 targets."""
    import yaml

    config_path = Path("/casp15/targets.yaml")
    with config_path.open() as f:
        config = yaml.safe_load(f)

    targets = config["targets"]
    if max_targets > 0:
        targets = targets[:max_targets]

    results = {"cached": [], "skipped": [], "errors": []}

    for i, target in enumerate(targets):
        tc_name = target["name"]
        tc_yaml = Path(f"/casp15/{target['yaml']}")
        cache_dir = Path(f"/msa_cache/{tc_name}")

        print(f"\n[{i+1}/{len(targets)}] {tc_name} (PDB {target['pdb_id']}, "
              f"{target['total_residues']} res, {target['num_chains']} chains)")

        # Check if already cached
        if cache_dir.exists() and any(cache_dir.glob("*.csv")):
            existing = list(cache_dir.glob("*.csv"))
            print(f"  Already cached: {len(existing)} MSA files")
            results["skipped"].append(tc_name)
            continue

        if not tc_yaml.exists():
            print(f"  ERROR: YAML not found: {tc_yaml}")
            results["errors"].append({"target": tc_name, "error": "YAML not found"})
            continue

        # Use boltz's process_inputs to generate MSAs
        cache_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(f"/tmp/msa_gen/{tc_name}")
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            import boltz.main as boltz_main

            # Download CCD data (needed for process_inputs, but NOT the model weights)
            model_cache = Path.home() / ".boltz"
            model_cache.mkdir(parents=True, exist_ok=True)
            ccd_path = model_cache / "ccd.pkl"
            mol_dir = model_cache / "mols"
            if not ccd_path.exists() or not mol_dir.exists():
                boltz_main.download_boltz2(model_cache)

            t_start = time.perf_counter()

            data = boltz_main.check_inputs(tc_yaml)
            boltz_main.process_inputs(
                data=data,
                out_dir=work_dir,
                ccd_path=ccd_path,
                mol_dir=mol_dir,
                use_msa_server=True,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy",
                boltz2=True,
                preprocessing_threads=1,
                max_msa_seqs=8192,
            )

            t_elapsed = time.perf_counter() - t_start

            # Find and copy MSA files
            msa_dir = work_dir / "processed" / "msa"
            if msa_dir.exists():
                msa_files = sorted(msa_dir.glob("*.csv"))
                for msa_file in msa_files:
                    dest = cache_dir / f"{tc_name}_{msa_file.stem}.csv"
                    shutil.copy2(msa_file, dest)
                    print(f"  Cached: {dest.name} ({msa_file.stat().st_size / 1024:.1f} KB)")

                results["cached"].append({
                    "target": tc_name,
                    "num_files": len(msa_files),
                    "time_s": round(t_elapsed, 1),
                })
                print(f"  Done in {t_elapsed:.1f}s")
            else:
                # Search more broadly
                csv_files = sorted(work_dir.rglob("*.csv"))
                csv_files = [f for f in csv_files if "confidence" not in f.name]
                if csv_files:
                    for csv_file in csv_files:
                        dest = cache_dir / f"{tc_name}_{csv_file.stem}.csv"
                        shutil.copy2(csv_file, dest)
                        print(f"  Cached: {dest.name}")
                    results["cached"].append({
                        "target": tc_name,
                        "num_files": len(csv_files),
                        "time_s": round(t_elapsed, 1),
                    })
                else:
                    print(f"  WARNING: No MSA files generated")
                    results["errors"].append({
                        "target": tc_name,
                        "error": "no MSA files found",
                    })

        except Exception as e:
            print(f"  ERROR: {e}")
            results["errors"].append({"target": tc_name, "error": str(e)[:500]})

        msa_volume.commit()

    print(f"\n{'='*60}")
    print(f"MSA Generation Complete")
    print(f"{'='*60}")
    print(f"Cached:  {len(results['cached'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Errors:  {len(results['errors'])}")

    return json.dumps(results, indent=2)


@app.function(
    timeout=120,
    volumes={"/msa_cache": msa_volume},
)
def verify_cache() -> str:
    """List all cached MSA files on the volume."""
    results = {}
    cache_root = Path("/msa_cache")

    if not cache_root.exists():
        return json.dumps({"error": "Volume empty"})

    for target_dir in sorted(cache_root.iterdir()):
        if target_dir.is_dir():
            files = []
            for f in sorted(target_dir.iterdir()):
                files.append({
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
            results[target_dir.name] = files

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(verify_only: bool = False, max_targets: int = 0):
    if verify_only:
        print("[casp15-msa] Verifying cache contents...")
        result = verify_cache.remote()
        data = json.loads(result)
        if "error" in data:
            print(f"  {data['error']}")
        else:
            for target, files in data.items():
                total_kb = sum(f["size_kb"] for f in files)
                print(f"  {target}: {len(files)} files ({total_kb:.0f} KB)")
            print(f"\nTotal: {len(data)} targets cached")
        return

    print(f"[casp15-msa] Generating MSA cache for CASP15 targets...")
    if max_targets > 0:
        print(f"[casp15-msa] Limited to first {max_targets} targets")

    result_json = generate_msas.remote(max_targets=max_targets)
    result = json.loads(result_json)
    print(json.dumps(result, indent=2))

    if result.get("errors"):
        print(f"\n[casp15-msa] {len(result['errors'])} errors encountered")
        sys.exit(1)
