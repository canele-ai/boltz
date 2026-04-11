"""Profile evaluator: measures GPU time breakdown on L40S.

Runs the 20-step/0-recycle config and profiles where GPU time goes.
Uses torch profiler to get a per-operation breakdown.
"""
from __future__ import annotations

import json
import os
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
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "boltz==2.2.1",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_dir(str(REPO_ROOT / "research" / "eval"), remote_path="/research_eval")
)

app = modal.App("boltz-l40s-profile", image=boltz_image)


@app.function(
    gpu="L40S",
    timeout=3600,
)
def profile_trunk():
    """Profile the Boltz-2 trunk on L40S to understand time breakdown.

    Returns a JSON with:
    - GPU properties (memory bandwidth, compute capability)
    - torch.compile availability
    - cuequivariance_torch availability and version
    - TF32 matmul behavior
    - Quick timing of triangular ops at representative sizes
    """
    import torch
    import importlib
    results = {}

    # --- GPU info ---
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        results["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "multi_processor_count": props.multi_processor_count,
        }
    else:
        results["gpu"] = {"error": "No CUDA device"}
        return json.dumps(results, indent=2)

    # --- Check cuequivariance_torch ---
    try:
        import cuequivariance_torch
        results["cuequivariance_torch"] = {
            "available": True,
            "version": getattr(cuequivariance_torch, "__version__", "unknown"),
        }
        # Try importing the triangle primitives
        try:
            from cuequivariance_torch.primitives.triangle import triangle_attention
            results["cuequivariance_torch"]["triangle_attention"] = True
        except Exception as e:
            results["cuequivariance_torch"]["triangle_attention"] = str(e)
        try:
            from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
            results["cuequivariance_torch"]["triangle_mult"] = True
        except Exception as e:
            results["cuequivariance_torch"]["triangle_mult"] = str(e)
    except ImportError as e:
        results["cuequivariance_torch"] = {
            "available": False,
            "error": str(e),
        }

    # --- Check trifast ---
    try:
        import trifast
        results["trifast"] = {
            "available": True,
            "version": getattr(trifast, "__version__", "unknown"),
        }
    except ImportError as e:
        results["trifast"] = {
            "available": False,
            "error": str(e),
        }

    # --- Check torch.compile ---
    results["torch_compile"] = {
        "available": hasattr(torch, "compile"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    # --- TF32 matmul test ---
    torch.set_float32_matmul_precision("highest")
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")

    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t_highest = time.perf_counter() - t0

    torch.set_float32_matmul_precision("high")
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t_high = time.perf_counter() - t0

    results["tf32_matmul"] = {
        "time_highest_100iter": round(t_highest, 4),
        "time_high_100iter": round(t_high, 4),
        "speedup": round(t_highest / t_high, 3) if t_high > 0 else None,
    }

    # --- Profile triangular operations at realistic sizes ---
    # Boltz-2 Pairformer: token_z=128, typical seq length 200-600
    # Test with N=200 (small complex), N=400 (medium), N=600 (large)
    sizes = [200, 400]
    dim = 128
    n_heads = 4
    head_dim = 32

    timing_results = {}

    for N in sizes:
        torch.set_float32_matmul_precision("highest")

        # --- Triangular multiplication (outgoing) ---
        # einsum: bikd,bjkd->bijd
        x = torch.randn(1, N, N, dim, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(1, N, N, dim, device="cuda", dtype=torch.float32)
        b = torch.randn(1, N, N, dim, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            torch.einsum("bikd,bjkd->bijd", a, b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            torch.einsum("bikd,bjkd->bijd", a, b)
        torch.cuda.synchronize()
        t_tri_mul_highest = (time.perf_counter() - t0) / 10

        torch.set_float32_matmul_precision("high")
        for _ in range(3):
            torch.einsum("bikd,bjkd->bijd", a, b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            torch.einsum("bikd,bjkd->bijd", a, b)
        torch.cuda.synchronize()
        t_tri_mul_high = (time.perf_counter() - t0) / 10

        # bf16 version
        a16 = a.bfloat16()
        b16 = b.bfloat16()
        for _ in range(3):
            torch.einsum("bikd,bjkd->bijd", a16, b16)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            torch.einsum("bikd,bjkd->bijd", a16, b16)
        torch.cuda.synchronize()
        t_tri_mul_bf16 = (time.perf_counter() - t0) / 10

        # --- Triangular attention (Q*K matmul) ---
        # Shape: [B, H, N, N, C_hidden] for Q,K
        q = torch.randn(1, n_heads, N, N, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, n_heads, N, N, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, n_heads, N, N, head_dim, device="cuda", dtype=torch.bfloat16)

        # Standard attention: Q*K^T then softmax then *V
        for _ in range(3):
            scores = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        torch.cuda.synchronize()
        t_tri_attn = (time.perf_counter() - t0) / 10

        # Test F.scaled_dot_product_attention
        # Reshape to [B*H*N, N, head_dim] for standard SDPA
        q_sdpa = q.reshape(-1, N, head_dim)
        k_sdpa = k.reshape(-1, N, head_dim)
        v_sdpa = v.reshape(-1, N, head_dim)

        for _ in range(3):
            out_sdpa = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            out_sdpa = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        torch.cuda.synchronize()
        t_sdpa = (time.perf_counter() - t0) / 10

        timing_results[f"N={N}"] = {
            "tri_mul_fp32_highest_ms": round(t_tri_mul_highest * 1000, 2),
            "tri_mul_fp32_tf32_ms": round(t_tri_mul_high * 1000, 2),
            "tri_mul_bf16_ms": round(t_tri_mul_bf16 * 1000, 2),
            "tri_attn_manual_bf16_ms": round(t_tri_attn * 1000, 2),
            "tri_attn_sdpa_bf16_ms": round(t_sdpa * 1000, 2),
            "sdpa_vs_manual_speedup": round(t_tri_attn / t_sdpa, 2) if t_sdpa > 0 else None,
            "tf32_vs_highest_speedup": round(t_tri_mul_highest / t_tri_mul_high, 2) if t_tri_mul_high > 0 else None,
            "bf16_vs_highest_speedup": round(t_tri_mul_highest / t_tri_mul_bf16, 2) if t_tri_mul_bf16 > 0 else None,
        }

    results["op_timing"] = timing_results

    # --- End-to-end timing with different configs ---
    # Now run actual Boltz prediction with different settings
    configs_to_test = [
        {"name": "baseline_20s_0r", "matmul_precision": "highest", "extra_args": []},
        {"name": "tf32_20s_0r", "matmul_precision": "high", "extra_args": []},
    ]

    e2e_results = {}

    for cfg in configs_to_test:
        torch.set_float32_matmul_precision(cfg["matmul_precision"])

        wrapper = "/research_eval/boltz_wrapper.py"
        tc_yaml = "/research_eval/test_cases/small_complex.yaml"
        import uuid
        work_dir = f"/tmp/boltz_profile/{cfg['name']}_{uuid.uuid4().hex[:8]}"
        os.makedirs(work_dir, exist_ok=True)

        cmd = [
            sys.executable, wrapper,
            tc_yaml,
            "--out_dir", work_dir,
            "--sampling_steps", "20",
            "--recycling_steps", "0",
            "--diffusion_samples", "1",
            "--override",
            "--no_kernels",
            "--matmul_precision", cfg["matmul_precision"],
            "--use_msa_server",
        ] + cfg["extra_args"]

        try:
            t_start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            t_end = time.perf_counter()

            if proc.returncode == 0:
                e2e_results[cfg["name"]] = {
                    "wall_time_s": round(t_end - t_start, 2),
                    "status": "success",
                }
            else:
                e2e_results[cfg["name"]] = {
                    "status": "error",
                    "stderr": proc.stderr[-500:] if proc.stderr else "",
                }
        except Exception as e:
            e2e_results[cfg["name"]] = {"status": "error", "error": str(e)}

    results["e2e_timing"] = e2e_results

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print("[profile] Starting L40S profiling...")
    result_json = profile_trunk.remote()
    result = json.loads(result_json)
    print(json.dumps(result, indent=2))

    # Save to file
    out_path = Path("orbits/l40s-kernels/profile_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[profile] Results saved to {out_path}")
