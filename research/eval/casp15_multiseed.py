"""Multi-seed validation for CASP15 outlier targets.

Runs baseline and optimized configs with 3 seeds on specified targets
to distinguish real regression from chain-packing variance.

Usage:
    modal run research/eval/casp15_multiseed.py
    modal run research/eval/casp15_multiseed.py --targets T1123,T1147,T1176,T1185
"""
from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import modal

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
BYPASS_DIR = REPO_ROOT / "research" / "solutions" / "bypass-lightning"
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
    .add_local_dir(str(EVAL_DIR / "casp15"), remote_path="/casp15")
    .add_local_file(
        str(BYPASS_DIR / "boltz_bypass_wrapper.py"),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
    .add_local_file(
        str(STACKED_WRAPPER),
        remote_path="/eval/boltz_wrapper_stacked.py",
    )
)

app = modal.App("boltz-casp15-multiseed", image=boltz_image)
msa_volume = modal.Volume.from_name("boltz-msa-cache-casp15", create_if_missing=True)
gt_volume = modal.Volume.from_name("boltz-gt-casp15", create_if_missing=True)

BASELINE_CONFIG = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "diffusion_samples": 1,
}

OPTIMIZED_CONFIG = {
    "sampling_steps": 12,
    "recycling_steps": 3,
    "gamma_0": 0.0,
    "noise_scale": 1.003,
    "matmul_precision": "high",
    "bf16_trunk": True,
    "enable_kernels": True,
    "cuda_warmup": True,
    "diffusion_samples": 1,
}

SEEDS = [42, 43, 44]

DEFAULT_TARGETS = ["T1123", "T1147", "T1176", "T1185"]


# ---------------------------------------------------------------------------
# Helpers (same as casp15_eval.py)
# ---------------------------------------------------------------------------

def _inject_cached_msas(input_yaml, msa_cache_root, work_dir):
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
            if str(entity_idx) in entity_msa_map:
                seq_entry["protein"]["msa"] = entity_msa_map[str(entity_idx)]
                injected += 1
            entity_idx += 1
    if injected == 0:
        return None
    cached_yaml = work_dir / f"{target_name}_cached.yaml"
    with cached_yaml.open("w") as f:
        import yaml as _yaml
        _yaml.dump(data, f, default_flow_style=False)
    return cached_yaml


def _run_subprocess(cmd):
    result = {"wall_time_s": None, "predict_only_s": None, "quality": {}, "error": None}
    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        result["wall_time_s"] = time.perf_counter() - t_start
        if proc.returncode != 0:
            result["error"] = f"Exit code {proc.returncode}.\nSTDERR: {proc.stderr[-2000:]}"
            return result
        if proc.stderr:
            ps = pe = None
            for line in proc.stderr.split("\n"):
                if "[PHASE] predict_start=" in line:
                    ps = float(line.split("=")[1])
                elif "[PHASE] predict_end=" in line:
                    pe = float(line.split("=")[1])
            if ps and pe:
                result["predict_only_s"] = pe - ps
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout after 3600s"
    except Exception as e:
        result["error"] = str(e)
    return result


def _run_baseline(input_yaml, out_dir, config):
    cmd = [
        sys.executable, "/eval/boltz_wrapper.py",
        str(input_yaml), "--out_dir", str(out_dir),
        "--sampling_steps", str(config["sampling_steps"]),
        "--recycling_steps", str(config["recycling_steps"]),
        "--diffusion_samples", str(config["diffusion_samples"]),
        "--override", "--matmul_precision", config["matmul_precision"],
    ]
    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")
    if config.get("seed") is not None:
        cmd.extend(["--seed", str(config["seed"])])
    return _run_subprocess(cmd)


def _run_optimized(input_yaml, out_dir, config):
    cmd = [
        sys.executable, "/eval/boltz_bypass_wrapper.py",
        str(input_yaml), "--out_dir", str(out_dir),
        "--sampling_steps", str(config["sampling_steps"]),
        "--recycling_steps", str(config["recycling_steps"]),
        "--diffusion_samples", str(config["diffusion_samples"]),
        "--override",
        "--gamma_0", str(config["gamma_0"]),
        "--noise_scale", str(config["noise_scale"]),
        "--matmul_precision", config["matmul_precision"],
    ]
    if config.get("enable_kernels"):
        cmd.append("--enable_kernels")
    if config.get("bf16_trunk"):
        cmd.append("--bf16_trunk")
    if config.get("cuda_warmup"):
        cmd.append("--cuda_warmup")
    if not config.get("_msa_cached"):
        cmd.append("--use_msa_server")
    if config.get("seed") is not None:
        cmd.extend(["--seed", str(config["seed"])])
    return _run_subprocess(cmd)


def _parse_confidence(out_dir, input_yaml):
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
    conf_files = sorted(results_dir.glob("confidence_*.json"))
    if not conf_files:
        return {"error": "No confidence JSON"}
    with conf_files[0].open() as f:
        conf = json.load(f)
    return {k: conf[k] for k in ["complex_plddt", "iptm", "ptm"] if k in conf}


def _compare_structures(pred_dir, input_yaml, gt_path, chain_mapping):
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
        from Bio.SVDSuperimposer import SVDSuperimposer
        import numpy as np
        parser = MMCIFParser(QUIET=True)
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
        if not pred_cif:
            return {"error": "No predicted CIF"}
        gt_s = parser.get_structure("gt", str(gt_path))
        pred_s = parser.get_structure("pred", str(pred_cif))
        # Match by sequential position within each chain (not residue number)
        # because Boltz outputs 1-based indices while cryo-EM PDB structures
        # use deposited numbering (e.g. starting at 420).
        from collections import defaultdict
        gt_by_chain = defaultdict(list)
        for chain in gt_s[0]:
            for r in sorted(chain.get_residues(), key=lambda r: r.id[1]):
                if r.id[0] == " " and "CA" in r:
                    gt_by_chain[chain.id].append(r["CA"].get_vector().get_array())
        pred_by_chain = defaultdict(list)
        for chain in pred_s[0]:
            mapped = chain_mapping.get(chain.id, chain.id)
            for r in sorted(chain.get_residues(), key=lambda r: r.id[1]):
                if r.id[0] == " " and "CA" in r:
                    pred_by_chain[mapped].append(r["CA"].get_vector().get_array())
        matched_gt, matched_pred = [], []
        for cid in sorted(gt_by_chain):
            if cid in pred_by_chain:
                n = min(len(gt_by_chain[cid]), len(pred_by_chain[cid]))
                matched_gt.extend(gt_by_chain[cid][:n])
                matched_pred.extend(pred_by_chain[cid][:n])
        if len(matched_gt) < 10:
            return {"error": f"Too few CA atoms: {len(matched_gt)}", "matched": len(matched_gt)}
        sup = SVDSuperimposer()
        sup.set(np.array(matched_gt), np.array(matched_pred))
        sup.run()
        return {"ca_rmsd": round(float(sup.get_rms()), 3), "matched": len(matched_gt)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=21600,
    volumes={"/msa_cache": msa_volume, "/ground_truth": gt_volume},
)
def run_multiseed(target_names: list[str]) -> str:
    import yaml

    config_path = Path("/casp15/targets.yaml")
    with config_path.open() as f:
        all_targets = yaml.safe_load(f)["targets"]

    targets_by_name = {t["name"]: t for t in all_targets}
    targets = [targets_by_name[n] for n in target_names if n in targets_by_name]

    # Upload ground truth
    gt_local = Path("/casp15/ground_truth")
    gt_vol = Path("/ground_truth")
    if gt_local.exists():
        for cif in gt_local.glob("*.cif"):
            dest = gt_vol / cif.name
            if not dest.exists():
                shutil.copy2(cif, dest)
        gt_volume.commit()

    msa_root = Path("/msa_cache")
    results = {}

    for target in targets:
        tc_name = target["name"]
        tc_yaml = Path(f"/casp15/{target['yaml']}")
        gt_path = gt_vol / Path(target["ground_truth"]).name
        chain_mapping = target.get("chain_mapping", {})

        print(f"\n{'='*60}")
        print(f"  {tc_name} (PDB {target['pdb_id']}, {target['total_residues']} res, "
              f"{target['num_chains']} chains)")
        print(f"{'='*60}")

        target_results = {"baseline": [], "optimized": []}

        for config_name, config_base, run_fn in [
            ("baseline", BASELINE_CONFIG, _run_baseline),
            ("optimized", OPTIMIZED_CONFIG, _run_optimized),
        ]:
            for seed in SEEDS:
                config = {**config_base, "seed": seed}
                work_dir = Path(f"/tmp/ms/{tc_name}_{config_name}_{seed}_{uuid.uuid4().hex[:6]}")
                work_dir.mkdir(parents=True, exist_ok=True)

                effective_yaml = tc_yaml
                if msa_root.exists():
                    cached = _inject_cached_msas(tc_yaml, msa_root, work_dir)
                    if cached:
                        effective_yaml = cached
                        config["_msa_cached"] = True

                print(f"  {config_name} seed={seed}...", end=" ", flush=True)
                pred = run_fn(effective_yaml, work_dir, config)

                if pred["error"]:
                    print(f"ERROR: {pred['error'][:100]}")
                    target_results[config_name].append({
                        "seed": seed, "error": pred["error"][:300]
                    })
                    continue

                quality = _parse_confidence(work_dir, effective_yaml)
                structural = _compare_structures(work_dir, effective_yaml, gt_path, chain_mapping)
                plddt = quality.get("complex_plddt")
                rmsd = structural.get("ca_rmsd")
                t = pred.get("predict_only_s") or pred.get("wall_time_s")

                print(f"pLDDT={plddt:.4f}, RMSD={rmsd}Å, time={t:.1f}s")
                target_results[config_name].append({
                    "seed": seed, "plddt": plddt, "ca_rmsd": rmsd,
                    "predict_only_s": pred.get("predict_only_s"),
                    "wall_time_s": pred["wall_time_s"],
                })

        # Aggregate per target
        for cfg in ["baseline", "optimized"]:
            runs = [r for r in target_results[cfg] if "plddt" in r]
            if runs:
                plddts = [r["plddt"] for r in runs if r["plddt"] is not None]
                rmsds = [r["ca_rmsd"] for r in runs if r["ca_rmsd"] is not None]
                target_results[f"{cfg}_mean_plddt"] = sum(plddts) / len(plddts) if plddts else None
                target_results[f"{cfg}_mean_rmsd"] = sum(rmsds) / len(rmsds) if rmsds else None
                target_results[f"{cfg}_std_plddt"] = statistics.stdev(plddts) if len(plddts) > 1 else 0
                target_results[f"{cfg}_std_rmsd"] = statistics.stdev(rmsds) if len(rmsds) > 1 else 0

        results[tc_name] = target_results

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Multi-seed Summary (3 seeds per config)")
    print(f"{'='*60}")
    print(f"  {'Target':<10} {'BL pLDDT':>12} {'OPT pLDDT':>12} {'Δ pp':>8}  "
          f"{'BL RMSD':>12} {'OPT RMSD':>12} {'Δ Å':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}  {'-'*12} {'-'*12} {'-'*8}")

    for tc_name in target_names:
        if tc_name not in results:
            continue
        r = results[tc_name]
        bp = r.get("baseline_mean_plddt")
        op = r.get("optimized_mean_plddt")
        bs = r.get("baseline_std_plddt", 0)
        os_ = r.get("optimized_std_plddt", 0)
        br = r.get("baseline_mean_rmsd")
        or_ = r.get("optimized_mean_rmsd")
        brs = r.get("baseline_std_rmsd", 0)
        ors = r.get("optimized_std_rmsd", 0)

        bp_s = f"{bp:.4f}±{bs:.4f}" if bp else "N/A"
        op_s = f"{op:.4f}±{os_:.4f}" if op else "N/A"
        dp = f"{(op-bp)*100:+.2f}" if bp and op else "N/A"
        br_s = f"{br:.2f}±{brs:.2f}" if br else "N/A"
        or_s = f"{or_:.2f}±{ors:.2f}" if or_ else "N/A"
        dr = f"{or_-br:+.2f}" if br and or_ else "N/A"

        print(f"  {tc_name:<10} {bp_s:>12} {op_s:>12} {dp:>8}  {br_s:>12} {or_s:>12} {dr:>8}")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(targets: str = ""):
    target_list = targets.split(",") if targets else DEFAULT_TARGETS
    target_list = [t.strip() for t in target_list if t.strip()]

    print(f"[multiseed] Running 3-seed validation on {target_list}")
    result_json = run_multiseed.remote(target_list)
    result = json.loads(result_json)

    out_path = EVAL_DIR / "casp15" / "results" / "multiseed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[multiseed] Results saved to {out_path}")
