"""Profile triangle multiplication ops to understand time breakdown.

Runs standalone microbenchmarks of TriangleMultiplication{Outgoing,Incoming}
with and without cuequivariance kernels, plus sparse alternatives.

Usage:
    modal run orbits/sparse-tri-ops/profile_trimult.py
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

app = modal.App("profile-trimult", image=boltz_image)


def benchmark_fn(fn, warmup=5, repeats=20):
    """Benchmark a function, return median time in ms."""
    import torch
    # warmup
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


@app.function(gpu="L40S", timeout=1200)
def profile_triangle_ops() -> str:
    import torch
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    results = {}
    device = "cuda"
    dim = 128  # standard pair representation dimension

    for N in [200, 400, 600]:
        results[f"N={N}"] = {}
        B = 1

        # Create modules
        tri_out = TriangleMultiplicationOutgoing(dim).to(device).eval()
        tri_in = TriangleMultiplicationIncoming(dim).to(device).eval()

        # Create inputs
        x = torch.randn(B, N, N, dim, device=device, dtype=torch.float32)
        mask = torch.ones(B, N, N, device=device, dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x_bf16 = x.to(torch.bfloat16)

            # 1. Dense without kernels (fp32 path)
            def run_dense_out():
                return tri_out(x, mask, use_kernels=False)
            results[f"N={N}"]["dense_out_fp32"] = benchmark_fn(run_dense_out)

            def run_dense_in():
                return tri_in(x, mask, use_kernels=False)
            results[f"N={N}"]["dense_in_fp32"] = benchmark_fn(run_dense_in)

            # 2. Dense with cuequivariance kernels
            try:
                def run_kernel_out():
                    return tri_out(x, mask, use_kernels=True)
                results[f"N={N}"]["kernel_out"] = benchmark_fn(run_kernel_out)

                def run_kernel_in():
                    return tri_in(x, mask, use_kernels=True)
                results[f"N={N}"]["kernel_in"] = benchmark_fn(run_kernel_in)
            except Exception as e:
                results[f"N={N}"]["kernel_error"] = str(e)

            # 3. Just the einsum portion (to isolate contraction cost)
            x_normed = tri_out.norm_in(x)
            x_proj = tri_out.p_in(x_normed) * tri_out.g_in(x_normed).sigmoid()
            x_proj = x_proj * mask.unsqueeze(-1)
            a_fp32, b_fp32 = torch.chunk(x_proj.float(), 2, dim=-1)
            a_bf16, b_bf16 = torch.chunk(x_proj.bfloat16(), 2, dim=-1)

            def run_einsum_out_fp32():
                return torch.einsum("bikd,bjkd->bijd", a_fp32, b_fp32)
            results[f"N={N}"]["einsum_out_fp32"] = benchmark_fn(run_einsum_out_fp32)

            def run_einsum_out_bf16():
                return torch.einsum("bikd,bjkd->bijd", a_bf16, b_bf16)
            results[f"N={N}"]["einsum_out_bf16"] = benchmark_fn(run_einsum_out_bf16)

            # 4. Sparse (window-based) approach
            for W in [32, 64, 128, 256]:
                if W >= N:
                    continue

                def run_sparse_window_out(W=W):
                    # For each (i,j), k in [max(0,min(i,j)-W/2), min(N,max(i,j)+W/2)]
                    # Simplified: use bmm with gathered slices
                    # Actually, reshape to use block-diagonal-like structure
                    # For profiling, we'll use the gather approach
                    B_, Ni, Nj, D = a_bf16.shape

                    # Build per-row indices: for row i, gather k in [i-W//2, i+W//2]
                    indices = torch.arange(N, device=device).unsqueeze(0).expand(N, -1)  # (N, N)
                    centers = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)
                    # Window mask: |k - i| <= W//2
                    wmask = (indices - centers).abs() <= W // 2  # (N, N)
                    # Get topk indices per row (fixed W positions)
                    # Use topk on the mask to get the W indices
                    # Actually simpler: just create the window indices
                    starts = torch.clamp(torch.arange(N, device=device) - W//2, 0, N-W)
                    offsets = torch.arange(W, device=device)
                    kidx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (N, W)
                    kidx = kidx.clamp(0, N-1)

                    # Gather a[b,i,k,d] for k in window
                    kidx_a = kidx.unsqueeze(0).unsqueeze(-1).expand(B_, N, W, D)  # (B, N, W, D)
                    a_sparse = torch.gather(a_bf16, 2, kidx_a)  # (B, N, W, D)

                    # For outgoing: need b[b,j,k,d] with same k per (i,j)
                    # But k depends on i, so for each i, we gather b at those k for all j
                    kidx_b = kidx.unsqueeze(0).unsqueeze(-1).expand(B_, N, W, D)
                    b_sparse = torch.gather(b_bf16, 2, kidx_b)  # (B, N, W, D)

                    # Now compute a_sparse[b,i,w,d] * b_sparse[b,j,w,d] -> sum over w
                    # This is still O(N^2 * W * D) - same as dense but with W < N
                    # Use einsum
                    return torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)

                results[f"N={N}"][f"sparse_window_W{W}_out_bf16"] = benchmark_fn(run_sparse_window_out)

            # 5. Alternative: just use smaller matmul with masking
            # (zeros out distant contributions but still does full matmul)
            def run_masked_out_bf16():
                # Create distance-based mask
                idx = torch.arange(N, device=device)
                dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float()  # (N, N)
                wmask = (dist <= 64).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
                a_masked = a_bf16 * wmask
                return torch.einsum("bikd,bjkd->bijd", a_masked, b_bf16)
            results[f"N={N}"]["masked_W64_out_bf16"] = benchmark_fn(run_masked_out_bf16)

            # 6. BMM-based (reshape einsum to batched matmul for comparison)
            def run_bmm_out_bf16():
                # bikd,bjkd->bijd = (B*Ni, K, D) @ (B*Nj, K, D)^T reshaped
                # Actually: for each b, a[b] is (N,N,D), want a[b] @ b[b]^T along dim 1
                # einsum bikd,bjkd->bijd = bmm(a.permute(0,1,3,2), b.permute(0,3,1,2))... no
                # Better: a is (B,N,N,D), treat as (B*N, N, D)
                # result[b,i,j,d] = sum_k a[b,i,k,d]*b[b,j,k,d]
                # = for each d: (B,N,N) matmul-like: result[:,:,j] = sum_k a[:,:,k] * b[:,j,k]
                # This is really (B,N_i,K) @ (B,N_j,K)^T for each d... not great
                # Actually the standard approach: reshape to bmm
                B_, Ni, K_, D_ = a_bf16.shape
                # a: (B,Ni,K,D) -> (B*D, Ni, K)
                a_r = a_bf16.permute(0,3,1,2).reshape(B_*D_, Ni, K_)
                b_r = b_bf16.permute(0,3,1,2).reshape(B_*D_, Ni, K_)
                out = torch.bmm(a_r, b_r.transpose(1,2))  # (B*D, N, N)
                return out.reshape(B_, D_, Ni, Ni).permute(0,2,3,1)  # (B, N, N, D)

            results[f"N={N}"]["bmm_out_bf16"] = benchmark_fn(run_bmm_out_bf16)

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print("[profile] Starting triangle multiplication profiling...")
    result_json = profile_triangle_ops.remote()
    print(result_json)

    # Save results
    out_path = Path(__file__).parent / "profile_results.json"
    with open(out_path, "w") as f:
        f.write(result_json)
    print(f"\nResults saved to {out_path}")
