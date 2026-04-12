"""Profile compiled sparse triangle mult vs cuequivariance kernel.

Tests whether torch.compile can fuse the sparse path to compete
with cuequivariance kernels.

Usage:
    modal run orbits/sparse-tri-ops/profile_compiled_sparse.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
    )
    .pip_install("boltz==2.2.1")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
)

app = modal.App("profile-compiled-sparse", image=boltz_image)


def benchmark_fn(fn, warmup=10, repeats=30):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "median_ms": times[len(times)//2],
        "min_ms": times[0],
        "max_ms": times[-1],
        "mean_ms": sum(times) / len(times),
    }


@app.function(gpu="L40S", timeout=1800)
def profile_compiled() -> str:
    import torch
    import torch.nn as nn
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    results = {}
    device = "cuda"
    dim = 128
    N = 600
    B = 1

    tri_out = TriangleMultiplicationOutgoing(dim).to(device).eval()
    tri_in = TriangleMultiplicationIncoming(dim).to(device).eval()

    x = torch.randn(B, N, N, dim, device=device, dtype=torch.float32)
    mask = torch.ones(B, N, N, device=device, dtype=torch.float32)

    with torch.no_grad():
        # 1. cuequivariance kernel baseline
        results["cueq_out"] = benchmark_fn(lambda: tri_out(x, mask, use_kernels=True))
        results["cueq_in"] = benchmark_fn(lambda: tri_in(x, mask, use_kernels=True))
        results["cueq_total"] = results["cueq_out"]["median_ms"] + results["cueq_in"]["median_ms"]

        # 2. Create a sparse TriangleMult module that can be compiled
        class SparseTriMulOut(nn.Module):
            def __init__(self, base, W, N):
                super().__init__()
                self.norm_in = base.norm_in
                self.p_in = base.p_in
                self.g_in = base.g_in
                self.norm_out = base.norm_out
                self.p_out = base.p_out
                self.g_out = base.g_out
                self.W = W
                starts = torch.clamp(torch.arange(N, device=device) - W//2, 0, N-W)
                offsets = torch.arange(W, device=device)
                self.register_buffer("kidx", (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N-1))

            def forward(self, x, mask):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x, 2, dim=-1)
                B_, Ni, Nk, D = a.shape
                W = self.W
                kidx_exp = self.kidx.unsqueeze(0).unsqueeze(-1).expand(B_, Ni, W, D)
                a_sparse = torch.gather(a, 2, kidx_exp)
                b_sparse = torch.gather(b, 2, kidx_exp)
                x = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)
                x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
                return x

        class SparseTriMulIn(nn.Module):
            def __init__(self, base, W, N):
                super().__init__()
                self.norm_in = base.norm_in
                self.p_in = base.p_in
                self.g_in = base.g_in
                self.norm_out = base.norm_out
                self.p_out = base.p_out
                self.g_out = base.g_out
                self.W = W
                starts = torch.clamp(torch.arange(N, device=device) - W//2, 0, N-W)
                offsets = torch.arange(W, device=device)
                self.register_buffer("kidx", (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N-1))

            def forward(self, x, mask):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x, 2, dim=-1)
                B_, Nk, Ni, D = a.shape
                W = self.W
                kidx_exp = self.kidx.unsqueeze(0).unsqueeze(-1).expand(B_, Ni, W, D)
                a_t = a.transpose(1, 2)
                b_t = b.transpose(1, 2)
                a_sparse = torch.gather(a_t, 2, kidx_exp)
                b_sparse = torch.gather(b_t, 2, kidx_exp)
                x = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)
                x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
                return x

        # 3. Dense bf16 forward (no kernels, no float upcast)
        class DenseBf16TriMulOut(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.norm_in = base.norm_in
                self.p_in = base.p_in
                self.g_in = base.g_in
                self.norm_out = base.norm_out
                self.p_out = base.p_out
                self.g_out = base.g_out

            def forward(self, x, mask):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x, 2, dim=-1)
                x = torch.einsum("bikd,bjkd->bijd", a, b)
                x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
                return x

        for W in [64, 128, 256]:
            # Uncompiled sparse
            sparse_out = SparseTriMulOut(tri_out, W, N).eval()
            sparse_in = SparseTriMulIn(tri_in, W, N).eval()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                results[f"sparse_W{W}_uncompiled_out"] = benchmark_fn(lambda s=sparse_out: s(x, mask))
                results[f"sparse_W{W}_uncompiled_in"] = benchmark_fn(lambda s=sparse_in: s(x, mask))

            # Compiled sparse
            try:
                sparse_out_c = torch.compile(SparseTriMulOut(tri_out, W, N).eval(), dynamic=False)
                sparse_in_c = torch.compile(SparseTriMulIn(tri_in, W, N).eval(), dynamic=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Extra warmup for compilation
                    for _ in range(5):
                        sparse_out_c(x, mask)
                        sparse_in_c(x, mask)
                    results[f"sparse_W{W}_compiled_out"] = benchmark_fn(lambda s=sparse_out_c: s(x, mask))
                    results[f"sparse_W{W}_compiled_in"] = benchmark_fn(lambda s=sparse_in_c: s(x, mask))
                    results[f"sparse_W{W}_compiled_total"] = (
                        results[f"sparse_W{W}_compiled_out"]["median_ms"] +
                        results[f"sparse_W{W}_compiled_in"]["median_ms"]
                    )
            except Exception as e:
                results[f"sparse_W{W}_compiled_error"] = str(e)

        # Compiled dense bf16
        try:
            dense_out_c = torch.compile(DenseBf16TriMulOut(tri_out).eval(), dynamic=False)
            dense_in_c = torch.compile(DenseBf16TriMulOut(tri_in).eval(), dynamic=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(5):
                    dense_out_c(x, mask)
                    dense_in_c(x, mask)
                results["dense_bf16_compiled_out"] = benchmark_fn(lambda: dense_out_c(x, mask))
                results["dense_bf16_compiled_in"] = benchmark_fn(lambda: dense_in_c(x, mask))
                results["dense_bf16_compiled_total"] = (
                    results["dense_bf16_compiled_out"]["median_ms"] +
                    results["dense_bf16_compiled_in"]["median_ms"]
                )
        except Exception as e:
            results["dense_bf16_compiled_error"] = str(e)

    # Summary
    print("\n=== N=600 Summary (median ms) ===")
    print(f"  cueq (out+in):           {results['cueq_total']:.3f} ms")
    for W in [64, 128, 256]:
        k = f"sparse_W{W}_compiled_total"
        if k in results:
            print(f"  sparse W={W} compiled:    {results[k]:.3f} ms")
        k2 = f"sparse_W{W}_uncompiled_out"
        if k2 in results:
            t = results[k2]["median_ms"] + results[f"sparse_W{W}_uncompiled_in"]["median_ms"]
            print(f"  sparse W={W} uncompiled:  {t:.3f} ms")
    if "dense_bf16_compiled_total" in results:
        print(f"  dense bf16 compiled:     {results['dense_bf16_compiled_total']:.3f} ms")

    return json.dumps(results, indent=2, default=str)


@app.local_entrypoint()
def main():
    print("[profile] Starting compiled sparse profiling...")
    result_json = profile_compiled.remote()
    print(result_json)

    out_path = Path(__file__).parent / "profile_compiled_results.json"
    with open(out_path, "w") as f:
        f.write(result_json)
    print(f"\nResults saved to {out_path}")
