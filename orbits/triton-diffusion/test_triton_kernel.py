"""Test Triton attention kernel correctness and benchmark speed.

Runs on Modal GPU (L40S) to validate:
1. Correctness: Triton output matches reference PyTorch implementation
2. Speed: Benchmark Triton vs einsum for typical Boltz-2 tensor sizes

Usage:
    modal run orbits/triton-diffusion/test_triton_kernel.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ORBIT_DIR = Path(__file__).resolve().parent

test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "triton>=2.2.0",
        "numpy>=1.26,<2.0",
    )
    .add_local_file(
        str(ORBIT_DIR / "triton_attention.py"),
        remote_path="/code/triton_attention.py",
    )
)

app = modal.App("triton-attention-test", image=test_image)


@app.function(gpu="L40S", timeout=600)
def test_correctness_and_benchmark() -> str:
    """Test Triton kernel correctness and benchmark speed."""
    import sys
    sys.path.insert(0, "/code")

    import torch
    from triton_attention import triton_attention_pair_bias, reference_attention_pair_bias

    results = {
        "env": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "correctness": [],
        "benchmarks": [],
    }

    # Test shapes matching Boltz-2 diffusion transformer
    # B=1, H=8, D=48 (384/8), varying S
    test_configs = [
        {"B": 1, "S": 32,  "H": 8, "D": 48, "label": "tiny"},
        {"B": 1, "S": 64,  "H": 8, "D": 48, "label": "small_window"},
        {"B": 1, "S": 128, "H": 8, "D": 48, "label": "medium_window"},
        {"B": 1, "S": 200, "H": 8, "D": 48, "label": "small_complex"},
        {"B": 1, "S": 400, "H": 8, "D": 48, "label": "medium_complex"},
        {"B": 1, "S": 600, "H": 8, "D": 48, "label": "large_complex"},
        # Also test with actual Boltz windowed attention sizes
        # AtomTransformer uses W=32 or W=64 windows
        {"B": 8,  "S": 32, "H": 8, "D": 48, "label": "batched_window_32"},
        {"B": 16, "S": 32, "H": 8, "D": 48, "label": "batched_window_32_x16"},
    ]

    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    for cfg in test_configs:
        B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
        label = cfg["label"]

        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        bias = torch.randn(B, H, S, S, device="cuda", dtype=torch.float32) * 0.1
        mask = torch.ones(B, S, device="cuda", dtype=torch.float32)
        # Mask out some positions to test masking
        if S > 10:
            mask[:, -3:] = 0.0

        # Reference
        ref = reference_attention_pair_bias(q, k, v, bias, mask)

        # Triton
        tri = triton_attention_pair_bias(q, k, v, bias, mask)

        # Compare
        max_diff = (ref - tri).abs().max().item()
        mean_diff = (ref - tri).abs().mean().item()

        # Check only non-masked positions
        valid_mask = mask[:, :, None, None].bool().expand_as(ref)
        max_diff_valid = (ref[valid_mask] - tri[valid_mask]).abs().max().item()

        passed = max_diff_valid < 1e-2  # relaxed for float32 tiling differences

        result = {
            "label": label,
            "shape": f"B={B}, S={S}, H={H}, D={D}",
            "max_abs_diff": max_diff,
            "max_abs_diff_valid": max_diff_valid,
            "mean_abs_diff": mean_diff,
            "passed": passed,
        }
        results["correctness"].append(result)

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: shape=({B},{S},{H},{D}), "
              f"max_diff={max_diff_valid:.6f}, mean_diff={mean_diff:.6f}")

    all_pass = all(r["passed"] for r in results["correctness"])
    results["all_correctness_pass"] = all_pass
    print(f"\nAll correctness tests: {'PASS' if all_pass else 'FAIL'}")

    # Benchmarks
    print("\n" + "=" * 60)
    print("SPEED BENCHMARKS")
    print("=" * 60)

    bench_configs = [
        {"B": 1, "S": 200, "H": 8, "D": 48, "label": "small_complex"},
        {"B": 1, "S": 400, "H": 8, "D": 48, "label": "medium_complex"},
        {"B": 1, "S": 600, "H": 8, "D": 48, "label": "large_complex"},
        {"B": 8, "S": 32,  "H": 8, "D": 48, "label": "windowed_B8"},
        {"B": 16, "S": 32, "H": 8, "D": 48, "label": "windowed_B16"},
    ]

    num_warmup = 10
    num_iters = 100

    for cfg in bench_configs:
        B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
        label = cfg["label"]

        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        bias = torch.randn(B, H, S, S, device="cuda", dtype=torch.float32) * 0.1
        mask = torch.ones(B, S, device="cuda", dtype=torch.float32)

        # Warmup reference
        for _ in range(num_warmup):
            _ = reference_attention_pair_bias(q, k, v, bias, mask)
        torch.cuda.synchronize()

        # Benchmark reference
        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = reference_attention_pair_bias(q, k, v, bias, mask)
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - t0) / num_iters * 1000  # ms

        # Warmup Triton
        for _ in range(num_warmup):
            _ = triton_attention_pair_bias(q, k, v, bias, mask)
        torch.cuda.synchronize()

        # Benchmark Triton
        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = triton_attention_pair_bias(q, k, v, bias, mask)
        torch.cuda.synchronize()
        tri_time = (time.perf_counter() - t0) / num_iters * 1000  # ms

        speedup = ref_time / tri_time if tri_time > 0 else 0

        bench = {
            "label": label,
            "shape": f"B={B}, S={S}, H={H}, D={D}",
            "ref_ms": round(ref_time, 3),
            "triton_ms": round(tri_time, 3),
            "speedup": round(speedup, 2),
        }
        results["benchmarks"].append(bench)

        print(f"  {label}: ref={ref_time:.3f}ms, triton={tri_time:.3f}ms, "
              f"speedup={speedup:.2f}x")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    result_json = test_correctness_and_benchmark.remote()
    result = json.loads(result_json)
    print("\n" + "=" * 60)
    print("FULL RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    if result.get("all_correctness_pass"):
        print("\nAll correctness tests PASSED.")
    else:
        print("\nSome correctness tests FAILED!")
