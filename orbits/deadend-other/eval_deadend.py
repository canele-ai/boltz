"""Evaluator for dead-end approaches stacked on winning config (eval-v5).

Tests three configurations:
  A) Control: bypass + ODE-12 + recycle=3 + TF32 + bf16 + warmup
  B) + Sparse triangle multiplication (W=32)
  C) + MSA skip on recycling passes 2+ (reuse MSA output from pass 1)

All with 3 seeds in parallel via Modal .map().
Includes CA RMSD structural comparison against PDB ground truth.

Usage:
    modal run orbits/deadend-other/eval_deadend.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
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

# Use eval-v5 test cases
MAIN_REPO = Path("/home/liambai/code/boltz")
EVAL_V5_DIR = MAIN_REPO / "research" / "eval"

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
    .add_local_dir(str(EVAL_V5_DIR), remote_path="/eval")
    .add_local_file(
        str(BYPASS_WRAPPER),
        remote_path="/eval/boltz_bypass_wrapper.py",
    )
)

app = modal.App("boltz-eval-deadend-other", image=boltz_image)

msa_volume = modal.Volume.from_name("boltz-msa-cache-v5", create_if_missing=False)
gt_volume = modal.Volume.from_name("boltz-ground-truth-v1", create_if_missing=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_CONFIG: dict[str, Any] = {
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

SEEDS = [42, 123, 7]

# Variants to test
VARIANTS = {
    "control": {},
    "sparse_w32": {"sparse_window": 32},
    "msa_skip": {"msa_skip_recycling": True},
}

# Chain mapping: predicted chain -> ground truth chain
CHAIN_MAPPINGS = {
    "small_complex": {"A": "A", "B": "D"},
    "medium_complex": {"A": "A", "B": "B", "C": "C"},
    "large_complex": {"A": "A", "B": "B", "C": "C", "D": "D"},
}

GT_FILES = {
    "small_complex": "1BRS.cif",
    "medium_complex": "1DQJ.cif",
    "large_complex": "2DN2.cif",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_cached_msas(input_yaml: Path, msa_cache_root: Path, work_dir: Path):
    """Inject cached MSA paths into input YAML."""
    import shutil
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
    print(f"[eval] MSA cache: injected {injected} cached MSA(s) for {target_name}")
    return cached_yaml


def _compare_structures(
    pred_dir: Path,
    input_yaml: Path,
    gt_root: Path,
    tc_name: str,
) -> dict[str, Any]:
    """Compare predicted structure against PDB ground truth via CA RMSD."""
    import numpy as np
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.SVDSuperimposer import SVDSuperimposer

    chain_mapping = CHAIN_MAPPINGS.get(tc_name, {})
    gt_file = GT_FILES.get(tc_name)
    if not gt_file:
        return {"error": f"No ground truth file for {tc_name}"}

    gt_path = gt_root / gt_file
    if not gt_path.exists():
        return {"error": f"Ground truth file not found: {gt_path}"}

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

        return {
            "ca_rmsd": round(float(ca_rmsd), 3),
            "matched_residues": len(matched_gt),
        }

    except Exception as e:
        return {"error": f"Structural comparison failed: {e}"}


def _write_variant_wrapper(work_dir: Path, variant: str, config: dict) -> str:
    """Write a variant-specific wrapper script that patches then runs bypass.

    Returns path to the wrapper script.
    """
    if variant == "sparse_w32":
        W = config.get("sparse_window", 32)
        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, "/eval")
            import torch

            from boltz.model.layers.triangular_mult import (
                TriangleMultiplicationOutgoing,
                TriangleMultiplicationIncoming,
            )
            W = {W}
            _kidx_cache = {{}}

            def _get_kidx(N, device):
                key = (N, str(device))
                if key not in _kidx_cache:
                    starts = torch.clamp(torch.arange(N, device=device) - W // 2, 0, N - W)
                    offsets = torch.arange(W, device=device)
                    _kidx_cache[key] = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N - 1)
                return _kidx_cache[key]

            def forward_outgoing_sparse(self, x, mask, use_kernels=False):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x, 2, dim=-1)
                B, Ni, Nk, D = a.shape
                if W >= Nk:
                    out = torch.einsum("bikd,bjkd->bijd", a, b)
                else:
                    kidx = _get_kidx(Nk, a.device)
                    kidx_exp = kidx.unsqueeze(0).unsqueeze(-1).expand(B, Ni, W, D)
                    a_sparse = torch.gather(a, 2, kidx_exp)
                    b_sparse = torch.gather(b, 2, kidx_exp)
                    out = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)
                out = self.p_out(self.norm_out(out)) * self.g_out(x_in).sigmoid()
                return out

            def forward_incoming_sparse(self, x, mask, use_kernels=False):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x, 2, dim=-1)
                B, Nk, Ni, D = a.shape
                if W >= Nk:
                    out = torch.einsum("bkid,bkjd->bijd", a, b)
                else:
                    kidx = _get_kidx(Nk, a.device)
                    kidx_exp = kidx.unsqueeze(0).unsqueeze(-1).expand(B, Ni, W, D)
                    a_t = a.transpose(1, 2)
                    b_t = b.transpose(1, 2)
                    a_sparse = torch.gather(a_t, 2, kidx_exp)
                    b_sparse = torch.gather(b_t, 2, kidx_exp)
                    out = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)
                out = self.p_out(self.norm_out(out)) * self.g_out(x_in).sigmoid()
                return out

            TriangleMultiplicationOutgoing.forward = forward_outgoing_sparse
            TriangleMultiplicationIncoming.forward = forward_incoming_sparse
            print(f"[deadend] Sparse tri-mult patch applied (W={W})", file=sys.stderr, flush=True)

            import boltz_bypass_wrapper
            boltz_bypass_wrapper.main()
        """)
    elif variant == "msa_skip":
        code = textwrap.dedent("""\
            import sys
            sys.path.insert(0, "/eval")
            import torch

            from boltz.model.modules.trunkv2 import MSAModule

            _orig_msa_forward = MSAModule.forward

            def _cached_msa_forward(self, z, emb, feats, use_kernels=False):
                if not hasattr(self, '_msa_cache_result'):
                    self._msa_cache_result = None
                    self._msa_call_count = 0

                self._msa_call_count += 1

                if self._msa_call_count == 1:
                    result = _orig_msa_forward(self, z, emb, feats, use_kernels=use_kernels)
                    self._msa_cache_result = result.detach().clone()
                    return result
                else:
                    print(f"[deadend] MSA skip: pass {self._msa_call_count}, reusing cached output",
                          file=sys.stderr, flush=True)
                    return self._msa_cache_result

            MSAModule.forward = _cached_msa_forward
            print("[deadend] MSA skip patch applied", file=sys.stderr, flush=True)

            import boltz_bypass_wrapper
            boltz_bypass_wrapper.main()
        """)
    else:
        return str(Path("/eval/boltz_bypass_wrapper.py"))

    wrapper_path = work_dir / f"wrapper_{variant}.py"
    wrapper_path.write_text(code)
    return str(wrapper_path)


def _run_boltz_bypass(
    input_yaml: Path,
    out_dir: Path,
    config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Run prediction using bypass-lightning wrapper with optional patches."""
    work_dir = out_dir.parent

    if variant in ("sparse_w32", "msa_skip"):
        effective_wrapper = _write_variant_wrapper(work_dir, variant, config)
    else:
        effective_wrapper = str(Path("/eval/boltz_bypass_wrapper.py"))

    cmd = [
        sys.executable, effective_wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
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
        "structural": {},
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

        # Parse confidence
        target_name = input_yaml.stem
        results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name
        if not results_dir.exists():
            pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
            if pred_base.exists():
                subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
                if subdirs:
                    results_dir = subdirs[0]

        if results_dir.exists():
            confidence_files = sorted(results_dir.glob("confidence_*.json"))
            if confidence_files:
                with confidence_files[0].open() as f:
                    conf = json.load(f)
                for key in ["confidence_score", "ptm", "iptm", "complex_plddt",
                             "complex_iplddt", "complex_pde", "complex_ipde"]:
                    if key in conf:
                        result["quality"][key] = conf[key]

    except subprocess.TimeoutExpired:
        result["error"] = "Prediction timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Modal function: evaluate one (variant, test_case, seed) triple
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/msa_cache": msa_volume, "/ground_truth": gt_volume},
)
def evaluate_one(variant: str, tc_name: str, seed: int) -> str:
    """Evaluate a single (variant, test_case, seed) combination."""
    config = dict(BASE_CONFIG)
    config["seed"] = seed

    variant_overrides = VARIANTS.get(variant, {})
    config.update(variant_overrides)

    msa_cache_root = Path("/msa_cache")
    gt_root = Path("/ground_truth")

    tc_yaml = Path(f"/eval/test_cases/{tc_name}.yaml")
    if not tc_yaml.exists():
        return json.dumps({"variant": variant, "tc_name": tc_name, "seed": seed,
                           "error": f"YAML not found: {tc_yaml}"})

    work_dir = Path(f"/tmp/boltz_eval/{variant}_{tc_name}_s{seed}_{uuid.uuid4().hex[:8]}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inject cached MSAs
    effective_yaml = tc_yaml
    if msa_cache_root.exists() and any(msa_cache_root.iterdir()):
        cached_yaml = _inject_cached_msas(tc_yaml, msa_cache_root, work_dir)
        if cached_yaml is not None:
            effective_yaml = cached_yaml
            config["_msa_cached"] = True
            print(f"[eval] Using cached MSAs for {tc_name}")
    else:
        print(f"[eval] WARNING: No MSA cache, using server for {tc_name}")

    print(f"[eval] Running variant={variant} {tc_name} seed={seed} "
          f"steps={config['sampling_steps']} recycle={config['recycling_steps']} "
          f"gamma_0={config['gamma_0']} warmup={config['cuda_warmup']}")

    pred_result = _run_boltz_bypass(effective_yaml, out_dir, config, variant)

    # Structural comparison (CA RMSD)
    if pred_result["error"] is None:
        struct_result = _compare_structures(out_dir, effective_yaml, gt_root, tc_name)
        pred_result["structural"] = struct_result
        if "ca_rmsd" in struct_result:
            print(f"[eval] {variant} {tc_name} seed={seed}: CA RMSD = {struct_result['ca_rmsd']}A")
        elif "error" in struct_result:
            print(f"[eval] {variant} {tc_name} seed={seed}: structural error: {struct_result['error']}")

    pred_result["variant"] = variant
    pred_result["tc_name"] = tc_name
    pred_result["seed"] = seed

    if pred_result["error"] is None:
        print(f"[eval] {variant} {tc_name} seed={seed}: wall={pred_result['wall_time_s']:.1f}s "
              f"predict={pred_result.get('predict_only_s', 'N/A')}s "
              f"pLDDT={pred_result['quality'].get('complex_plddt', 'N/A')}")

    return json.dumps(pred_result, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run dead-end evaluation: 3 variants x 3 test cases x 3 seeds = 27 jobs."""
    import statistics

    test_cases = ["small_complex", "medium_complex", "large_complex"]
    variant_names = list(VARIANTS.keys())

    # Build all (variant, tc, seed) triples
    jobs = [(v, tc, seed) for v in variant_names for tc in test_cases for seed in SEEDS]
    v_args = [j[0] for j in jobs]
    tc_args = [j[1] for j in jobs]
    seed_args = [j[2] for j in jobs]

    print(f"[eval] Launching {len(jobs)} parallel evaluations on Modal L40S...")
    print(f"[eval] Variants: {variant_names}")
    print(f"[eval] Config: {json.dumps(BASE_CONFIG, indent=2)}")
    print(f"[eval] Seeds: {SEEDS}")

    # Run all jobs in parallel via .map()
    results_raw = list(evaluate_one.map(v_args, tc_args, seed_args))

    # Parse results
    all_results = [json.loads(r) for r in results_raw]

    # Group by variant and test case
    by_variant: dict[str, dict[str, list]] = {}
    for r in all_results:
        v = r["variant"]
        tc = r["tc_name"]
        by_variant.setdefault(v, {}).setdefault(tc, []).append(r)

    # --- Baseline reference values ---
    eval_v5_baseline = {
        "small_complex": {"wall_time_s": 14.0, "plddt": 0.967, "ca_rmsd": 0.325},
        "medium_complex": {"wall_time_s": 33.5, "plddt": 0.962, "ca_rmsd": 5.243},
        "large_complex": {"wall_time_s": 41.9, "plddt": 0.966, "ca_rmsd": 0.474},
    }
    baseline_mean_wall = 29.78

    # --- Print results ---
    print("\n" + "=" * 90)
    print("DEAD-END APPROACH VALIDATION (eval-v5)")
    print("=" * 90)

    summary = {}

    for variant in variant_names:
        print(f"\n{'=' * 90}")
        print(f"VARIANT: {variant}")
        print(f"{'=' * 90}")

        variant_results = by_variant.get(variant, {})
        v_wall_times = []
        v_predict_times = []
        v_plddts = []
        v_rmsds = {}

        for tc in test_cases:
            tc_results = variant_results.get(tc, [])
            errors = [r for r in tc_results if r.get("error")]
            if errors:
                print(f"\n  {tc}: ERRORS")
                for e in errors:
                    print(f"    seed={e['seed']}: {e['error'][:300]}")
                continue

            wall_times = [r["wall_time_s"] for r in tc_results]
            predict_times = [r.get("predict_only_s") for r in tc_results
                             if r.get("predict_only_s")]
            plddts = [r["quality"]["complex_plddt"] for r in tc_results
                       if "complex_plddt" in r.get("quality", {})]
            ca_rmsds = [r["structural"]["ca_rmsd"] for r in tc_results
                        if "ca_rmsd" in r.get("structural", {})]

            bl = eval_v5_baseline[tc]
            mean_wall = statistics.mean(wall_times) if wall_times else None
            mean_pred = statistics.mean(predict_times) if predict_times else None
            mean_plddt = statistics.mean(plddts) if plddts else None
            mean_rmsd = statistics.mean(ca_rmsds) if ca_rmsds else None

            if mean_wall:
                v_wall_times.append(mean_wall)
            if predict_times:
                v_predict_times.extend(predict_times)
            if plddts:
                v_plddts.extend(plddts)
            if ca_rmsds:
                v_rmsds[tc] = ca_rmsds

            print(f"\n  --- {tc} (PDB: {GT_FILES[tc].replace('.cif', '')}) ---")
            print(f"  {'Seed':<8} {'Wall(s)':<10} {'Predict(s)':<12} {'pLDDT':<10} {'CA RMSD(A)':<12}")
            for r in tc_results:
                s = r["seed"]
                w = r.get("wall_time_s", 0)
                p = r.get("predict_only_s", 0) or 0
                pl = r.get("quality", {}).get("complex_plddt", 0)
                cr = r.get("structural", {}).get("ca_rmsd", "N/A")
                print(f"  {s:<8} {w:<10.1f} {p:<12.1f} {pl:<10.4f} {cr}")
            if mean_wall:
                print(f"  {'Mean':<8} {mean_wall:<10.1f} {mean_pred or 0:<12.1f} "
                      f"{mean_plddt or 0:<10.4f} {mean_rmsd or 'N/A'}")
                plddt_delta = (mean_plddt - bl['plddt']) * 100 if mean_plddt else None
                rmsd_delta = mean_rmsd - bl['ca_rmsd'] if mean_rmsd else None
                if plddt_delta is not None and rmsd_delta is not None:
                    print(f"  vs baseline: pLDDT {plddt_delta:+.1f}pp, "
                          f"CA RMSD {rmsd_delta:+.3f}A")

        # Aggregate for this variant
        if v_wall_times:
            mean_wall_all = statistics.mean(v_wall_times)
            speedup = baseline_mean_wall / mean_wall_all
            mean_plddt_all = statistics.mean(v_plddts) if v_plddts else None

            summary[variant] = {
                "mean_wall_s": round(mean_wall_all, 1),
                "speedup": round(speedup, 2),
                "mean_plddt": round(mean_plddt_all, 4) if mean_plddt_all else None,
                "plddt_delta_pp": round((mean_plddt_all - 0.9650) * 100, 1) if mean_plddt_all else None,
                "ca_rmsds": {tc: round(statistics.mean(rs), 3) for tc, rs in v_rmsds.items()},
            }

            print(f"\n  AGGREGATE: wall={mean_wall_all:.1f}s, speedup={speedup:.2f}x"
                  + (f", pLDDT={mean_plddt_all:.4f}" if mean_plddt_all else ""))

    # --- Final summary ---
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")
    header = (f"{'Variant':<15} {'Wall(s)':<10} {'Speedup':<10} {'pLDDT':<10} "
              f"{'dPLDDT(pp)':<12} {'1BRS RMSD':<12} {'1DQJ RMSD':<12} {'2DN2 RMSD':<12}")
    print(header)
    print(f"{'baseline':<15} {29.78:<10.1f} {'1.00x':<10} {0.9650:<10.4f} {'--':<12} "
          f"{0.325:<12.3f} {5.243:<12.3f} {0.474:<12.3f}")
    for v, s in summary.items():
        rmsds = s.get("ca_rmsds", {})
        r_small = rmsds.get("small_complex", "N/A")
        r_medium = rmsds.get("medium_complex", "N/A")
        r_large = rmsds.get("large_complex", "N/A")
        r_small_s = f"{r_small:<12.3f}" if isinstance(r_small, float) else f"{r_small:<12}"
        r_medium_s = f"{r_medium:<12.3f}" if isinstance(r_medium, float) else f"{r_medium:<12}"
        r_large_s = f"{r_large:<12.3f}" if isinstance(r_large, float) else f"{r_large:<12}"
        dp = s.get('plddt_delta_pp', 0)
        print(f"{v:<15} {s['mean_wall_s']:<10.1f} {s['speedup']:.2f}x{'':>4} "
              f"{s['mean_plddt'] or 0:<10.4f} {dp:+.1f}{'':>6} "
              f"{r_small_s} {r_medium_s} {r_large_s}")

    # --- Dump full JSON ---
    print("\n\n--- FULL JSON RESULTS ---")
    print(json.dumps(all_results, indent=2))
