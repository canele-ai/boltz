"""Generate and store pre-computed MSA files for the eval test cases.

Runs Boltz once per test case with --use_msa_server, then extracts the
generated MSA CSV files from the processing output and stores them on a
Modal persistent volume AND downloads them locally to research/eval/msa_cache/.

This eliminates MSA server latency (~5-30s per complex, highly variable)
from all subsequent eval runs, giving clean GPU-only timing measurements.

Usage:
    # Generate cache (runs on Modal GPU to trigger Boltz's MSA pipeline)
    modal run research/eval/generate_msa_cache.py

    # Verify cache contents
    modal run research/eval/generate_msa_cache.py --verify-only
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
)

app = modal.App("boltz-msa-cache-gen", image=boltz_image)
msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=True)


@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_volume},
)
def generate_cache() -> str:
    """Run Boltz once per test case with MSA server, extract and cache MSA files."""
    import yaml

    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        eval_config = yaml.safe_load(f)

    test_cases = eval_config.get("test_cases", [])
    results = {"cached": [], "errors": []}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["errors"].append(f"YAML not found: {tc_yaml}")
            continue

        print(f"\n{'='*60}")
        print(f"[msa-cache] Generating MSAs for {tc_name}...")
        print(f"{'='*60}")

        work_dir = Path(f"/tmp/boltz_msa_gen/{tc_name}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Run Boltz with --use_msa_server and minimal steps (we only need MSAs)
        wrapper = str(Path("/eval/boltz_wrapper.py"))
        cmd = [
            sys.executable, wrapper,
            str(tc_yaml),
            "--out_dir", str(work_dir),
            "--sampling_steps", "5",  # minimal — we only need MSA generation
            "--recycling_steps", "0",
            "--diffusion_samples", "1",
            "--override",
            "--use_msa_server",
            "--matmul_precision", "highest",
        ]

        print(f"[msa-cache] Running: {' '.join(cmd)}")
        t_start = time.perf_counter()

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        t_elapsed = time.perf_counter() - t_start
        print(f"[msa-cache] Completed in {t_elapsed:.1f}s (exit code {proc.returncode})")

        if proc.returncode != 0:
            print(f"[msa-cache] STDERR: {proc.stderr[-2000:]}")
            results["errors"].append(f"{tc_name}: exit code {proc.returncode}")
            continue

        # Find generated MSA files
        # Boltz stores MSAs in {out_dir}/boltz_results_{name}/msa/{name}_{entity_id}.csv
        target_name = tc_yaml.stem
        msa_search_dirs = [
            work_dir / f"boltz_results_{target_name}" / "msa",
            work_dir / "msa",
        ]

        msa_files = []
        for msa_dir in msa_search_dirs:
            if msa_dir.exists():
                msa_files = sorted(msa_dir.glob("*.csv"))
                if msa_files:
                    print(f"[msa-cache] Found MSA dir: {msa_dir}")
                    break

        if not msa_files:
            # Broader search
            print(f"[msa-cache] Searching for CSV files in {work_dir}...")
            msa_files = sorted(work_dir.rglob("*.csv"))
            # Filter out confidence JSONs etc
            msa_files = [f for f in msa_files if "confidence" not in f.name]

        if not msa_files:
            results["errors"].append(f"{tc_name}: no MSA CSV files found")
            # Debug: list directory structure
            for p in sorted(work_dir.rglob("*"))[:50]:
                print(f"  {p}")
            continue

        # Copy to volume
        cache_dir = Path(f"/msa_cache/{target_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        cached_files = []
        for msa_file in msa_files:
            dest = cache_dir / msa_file.name
            shutil.copy2(msa_file, dest)
            size_kb = msa_file.stat().st_size / 1024
            print(f"[msa-cache] Cached: {msa_file.name} ({size_kb:.1f} KB)")
            cached_files.append({"name": msa_file.name, "size_kb": round(size_kb, 1)})

        results["cached"].append({
            "test_case": tc_name,
            "target_name": target_name,
            "num_files": len(cached_files),
            "files": cached_files,
            "generation_time_s": round(t_elapsed, 1),
        })

    msa_volume.commit()

    print(f"\n{'='*60}")
    print(f"[msa-cache] Summary: cached {len(results['cached'])} targets, "
          f"{len(results['errors'])} errors")
    print(f"{'='*60}")

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
        return json.dumps({"error": "Volume empty — run without --verify-only first"})

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


@app.function(
    timeout=300,
    volumes={"/msa_cache": msa_volume},
)
def download_cache() -> dict[str, bytes]:
    """Download all cached MSA files as a dict of {path: bytes}."""
    files = {}
    cache_root = Path("/msa_cache")

    for target_dir in sorted(cache_root.iterdir()):
        if target_dir.is_dir():
            for f in sorted(target_dir.iterdir()):
                key = f"{target_dir.name}/{f.name}"
                files[key] = f.read_bytes()

    return files


@app.local_entrypoint()
def main(verify_only: bool = False, download: bool = False):
    """Generate or verify MSA cache.

    Usage:
        modal run research/eval/generate_msa_cache.py              # Generate
        modal run research/eval/generate_msa_cache.py --verify-only # Verify
        modal run research/eval/generate_msa_cache.py --download    # Download to local
    """
    if verify_only:
        print("[msa-cache] Verifying cache contents...")
        result = verify_cache.remote()
        print(json.dumps(json.loads(result), indent=2))
        return

    if download:
        print("[msa-cache] Downloading cached MSAs to research/eval/msa_cache/...")
        files = download_cache.remote()
        local_cache = EVAL_DIR / "msa_cache"
        local_cache.mkdir(parents=True, exist_ok=True)

        for relpath, data in files.items():
            dest = local_cache / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"  {relpath} ({len(data)/1024:.1f} KB)")

        print(f"\n[msa-cache] Downloaded {len(files)} files to {local_cache}")
        return

    # Generate cache
    print("[msa-cache] Generating MSA cache on Modal GPU...")
    result_json = generate_cache.remote()
    result = json.loads(result_json)
    print(json.dumps(result, indent=2))

    if result.get("errors"):
        print(f"\n[msa-cache] ERRORS: {result['errors']}")
        sys.exit(1)

    print(f"\n[msa-cache] Successfully cached MSAs for {len(result['cached'])} test cases")
    print("[msa-cache] Now run with --download to pull files locally, then re-run baseline.")
