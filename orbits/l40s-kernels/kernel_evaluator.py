"""Evaluator that tests cuequivariance_torch kernels on L40S.

Installs cuequivariance_torch and tests whether the custom kernels work.
Then runs a comparison: with kernels vs without kernels on the same config.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent

# Image with cuequivariance packages installed
boltz_image_with_kernels = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "boltz==2.2.1",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_dir(str(REPO_ROOT / "research" / "eval"), remote_path="/research_eval")
)

app = modal.App("boltz-kernel-test", image=boltz_image_with_kernels)


@app.function(
    gpu="L40S",
    timeout=3600,
)
def test_kernels():
    """Test cuequivariance_torch kernels on L40S.

    Returns JSON with:
    - Whether cuequivariance_torch is available
    - Whether kernels work on L40S (sm_89)
    - Timing comparison: with kernels vs without
    """
    import torch
    results = {}

    props = torch.cuda.get_device_properties(0)
    results["gpu"] = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
    }

    # Check cuequivariance_torch
    try:
        import cuequivariance_torch
        results["cuequivariance_torch"] = {
            "available": True,
            "version": getattr(cuequivariance_torch, "__version__", "unknown"),
        }

        # Test triangle_attention
        try:
            from cuequivariance_torch.primitives.triangle import triangle_attention
            results["cuequivariance_torch"]["triangle_attention_import"] = True

            # Try running it on a small input
            N = 64
            H = 4
            C = 32
            q = torch.randn(1, H, N, N, C, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, H, N, N, C, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, H, N, N, C, device="cuda", dtype=torch.bfloat16)
            tri_bias = torch.randn(1, H, N, N, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(1, N, N, device="cuda", dtype=torch.bool)

            out = triangle_attention(q, k, v, tri_bias, mask=mask, scale=1.0/C**0.5)
            results["cuequivariance_torch"]["triangle_attention_works"] = True
            results["cuequivariance_torch"]["output_shape"] = str(out.shape)
        except Exception as e:
            results["cuequivariance_torch"]["triangle_attention_works"] = False
            results["cuequivariance_torch"]["triangle_attention_error"] = str(e)

        # Test triangle_multiplicative_update
        try:
            from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
            results["cuequivariance_torch"]["triangle_mult_import"] = True
        except Exception as e:
            results["cuequivariance_torch"]["triangle_mult_import"] = False
            results["cuequivariance_torch"]["triangle_mult_error"] = str(e)

    except ImportError as e:
        results["cuequivariance_torch"] = {
            "available": False,
            "error": str(e),
        }

    # Check trifast
    try:
        import trifast
        results["trifast"] = {
            "available": True,
            "version": getattr(trifast, "__version__", "unknown"),
        }
    except ImportError as e:
        results["trifast"] = {"available": False, "error": str(e)}

    # --- Run end-to-end predictions ---
    # Test 1: Without kernels (baseline, TF32)
    # Test 2: With kernels (if available)
    configs = [
        {"name": "no_kernels_tf32", "no_kernels": True, "precision": "high"},
        {"name": "no_kernels_highest", "no_kernels": True, "precision": "highest"},
    ]

    # Only test with kernels if cuequivariance is available
    if results.get("cuequivariance_torch", {}).get("available"):
        configs.append({"name": "with_kernels_tf32", "no_kernels": False, "precision": "high"})

    e2e_results = {}
    test_cases = ["small_complex", "medium_complex", "large_complex"]

    for cfg in configs:
        cfg_results = {}
        for tc_name in test_cases:
            tc_yaml = f"/research_eval/test_cases/{tc_name}.yaml"
            if not Path(tc_yaml).exists():
                cfg_results[tc_name] = {"status": "missing"}
                continue

            work_dir = f"/tmp/boltz_kernel/{cfg['name']}_{tc_name}_{uuid.uuid4().hex[:8]}"
            os.makedirs(work_dir, exist_ok=True)

            cmd = [
                sys.executable, "/research_eval/boltz_wrapper.py",
                tc_yaml,
                "--out_dir", work_dir,
                "--sampling_steps", "20",
                "--recycling_steps", "0",
                "--diffusion_samples", "1",
                "--override",
                "--matmul_precision", cfg["precision"],
                "--use_msa_server",
            ]
            if cfg["no_kernels"]:
                cmd.append("--no_kernels")

            try:
                t_start = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                t_end = time.perf_counter()

                if proc.returncode == 0:
                    # Parse confidence scores
                    import glob
                    conf_files = sorted(glob.glob(f"{work_dir}/boltz_results_*/predictions/*/confidence_*.json"))
                    quality = {}
                    if conf_files:
                        with open(conf_files[0]) as f:
                            conf = json.load(f)
                        quality = {k: conf[k] for k in ["complex_plddt", "iptm", "confidence_score"] if k in conf}

                    cfg_results[tc_name] = {
                        "wall_time_s": round(t_end - t_start, 2),
                        "status": "success",
                        "quality": quality,
                    }
                else:
                    cfg_results[tc_name] = {
                        "status": "error",
                        "stderr": proc.stderr[-1000:] if proc.stderr else "",
                    }
            except Exception as e:
                cfg_results[tc_name] = {"status": "error", "error": str(e)}

        e2e_results[cfg["name"]] = cfg_results

    results["e2e"] = e2e_results

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print("[kernel-test] Starting kernel test on L40S...")
    result_json = test_kernels.remote()
    result = json.loads(result_json)
    print(json.dumps(result, indent=2))

    out_path = Path("orbits/l40s-kernels/kernel_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[kernel-test] Results saved to {out_path}")
