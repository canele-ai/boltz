"""Profile full forward pass of sparse vs dense triangle multiplication.

Compares:
1. cuequivariance kernel (full forward, fused)
2. Dense PyTorch (no kernels, fp32)
3. Dense PyTorch bf16 (no float upcast)
4. Sparse window bf16 (various W sizes, full forward including norms/projections)

Usage:
    modal run orbits/sparse-tri-ops/profile_full_forward.py
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

app = modal.App("profile-full-forward", image=boltz_image)


def benchmark_fn(fn, warmup=5, repeats=20):
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


def make_sparse_forward_outgoing(module, W, N, device):
    """Create a sparse forward function for outgoing triangle mult."""
    import torch

    # Precompute window indices
    starts = torch.clamp(torch.arange(N, device=device) - W//2, 0, N-W)
    offsets = torch.arange(W, device=device)
    kidx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N-1)  # (N, W)

    def sparse_forward(x, mask, use_kernels=False):
        x = module.norm_in(x)
        x_in = x
        x = module.p_in(x) * module.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        # NO .float() upcast
        a, b = torch.chunk(x, 2, dim=-1)
        B, Ni, Nk, D = a.shape

        # Gather sparse window
        kidx_exp = kidx.unsqueeze(0).unsqueeze(-1).expand(B, N, W, D)
        a_sparse = torch.gather(a, 2, kidx_exp)  # (B, N, W, D)
        b_sparse = torch.gather(b, 2, kidx_exp)  # (B, N, W, D)

        # Sparse einsum
        x = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)

        x = module.p_out(module.norm_out(x)) * module.g_out(x_in).sigmoid()
        return x

    return sparse_forward


def make_sparse_forward_incoming(module, W, N, device):
    """Create a sparse forward function for incoming triangle mult."""
    import torch

    starts = torch.clamp(torch.arange(N, device=device) - W//2, 0, N-W)
    offsets = torch.arange(W, device=device)
    kidx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N-1)  # (N, W)

    def sparse_forward(x, mask, use_kernels=False):
        x = module.norm_in(x)
        x_in = x
        x = module.p_in(x) * module.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        B, Nk, Ni, D = a.shape

        # For incoming: einsum("bkid,bkjd->bijd")
        # k is dim 1, i is dim 2. We want to sparsify over k for each (i,j)
        # Gather along k (dim 1) for each i
        kidx_exp = kidx.unsqueeze(0).unsqueeze(-1).expand(B, N, W, D)

        # a[b,k,i,d] -> gather k for each i: a_sparse[b,i,w,d] = a[b,kidx[i,w],i,d]
        # Need to gather along dim=1
        # Reshape: transpose to (B,i,k,d) then gather along k
        a_t = a.transpose(1, 2)  # (B, N_i, N_k, D)
        b_t = b.transpose(1, 2)  # (B, N_j, N_k, D)

        a_sparse = torch.gather(a_t, 2, kidx_exp)  # (B, N, W, D)
        b_sparse = torch.gather(b_t, 2, kidx_exp)  # (B, N, W, D)

        # einsum over sparse window
        x = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)

        x = module.p_out(module.norm_out(x)) * module.g_out(x_in).sigmoid()
        return x

    return sparse_forward


@app.function(gpu="L40S", timeout=1200)
def profile_full() -> str:
    import torch
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    results = {}
    device = "cuda"
    dim = 128

    for N in [200, 400, 600]:
        results[f"N={N}"] = {}
        B = 1

        tri_out = TriangleMultiplicationOutgoing(dim).to(device).eval()
        tri_in = TriangleMultiplicationIncoming(dim).to(device).eval()

        x = torch.randn(B, N, N, dim, device=device, dtype=torch.float32)
        mask = torch.ones(B, N, N, device=device, dtype=torch.float32)

        with torch.no_grad():
            # 1. cuequivariance kernel
            try:
                results[f"N={N}"]["cueq_out"] = benchmark_fn(
                    lambda: tri_out(x, mask, use_kernels=True))
                results[f"N={N}"]["cueq_in"] = benchmark_fn(
                    lambda: tri_in(x, mask, use_kernels=True))
                results[f"N={N}"]["cueq_total"] = {
                    "median_ms": results[f"N={N}"]["cueq_out"]["median_ms"] +
                                 results[f"N={N}"]["cueq_in"]["median_ms"]
                }
            except Exception as e:
                results[f"N={N}"]["cueq_error"] = str(e)

            # 2. Dense bf16
            with torch.autocast("cuda", dtype=torch.bfloat16):
                def dense_bf16_out():
                    xn = tri_out.norm_in(x)
                    x_in = xn
                    xp = tri_out.p_in(xn) * tri_out.g_in(xn).sigmoid()
                    xp = xp * mask.unsqueeze(-1)
                    a, b = torch.chunk(xp, 2, dim=-1)
                    out = torch.einsum("bikd,bjkd->bijd", a, b)
                    return tri_out.p_out(tri_out.norm_out(out)) * tri_out.g_out(x_in).sigmoid()

                def dense_bf16_in():
                    xn = tri_in.norm_in(x)
                    x_in = xn
                    xp = tri_in.p_in(xn) * tri_in.g_in(xn).sigmoid()
                    xp = xp * mask.unsqueeze(-1)
                    a, b = torch.chunk(xp, 2, dim=-1)
                    out = torch.einsum("bkid,bkjd->bijd", a, b)
                    return tri_in.p_out(tri_in.norm_out(out)) * tri_in.g_out(x_in).sigmoid()

                results[f"N={N}"]["dense_bf16_out"] = benchmark_fn(dense_bf16_out)
                results[f"N={N}"]["dense_bf16_in"] = benchmark_fn(dense_bf16_in)
                results[f"N={N}"]["dense_bf16_total"] = {
                    "median_ms": results[f"N={N}"]["dense_bf16_out"]["median_ms"] +
                                 results[f"N={N}"]["dense_bf16_in"]["median_ms"]
                }

            # 3. Sparse window, various W, bf16
            for W in [32, 64, 128, 256]:
                if W >= N:
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sparse_out = make_sparse_forward_outgoing(tri_out, W, N, device)
                    sparse_in = make_sparse_forward_incoming(tri_in, W, N, device)

                    def run_sparse_out(f=sparse_out):
                        return f(x, mask)
                    def run_sparse_in(f=sparse_in):
                        return f(x, mask)

                    results[f"N={N}"][f"sparse_W{W}_out"] = benchmark_fn(run_sparse_out)
                    results[f"N={N}"][f"sparse_W{W}_in"] = benchmark_fn(run_sparse_in)
                    results[f"N={N}"][f"sparse_W{W}_total"] = {
                        "median_ms": results[f"N={N}"][f"sparse_W{W}_out"]["median_ms"] +
                                     results[f"N={N}"][f"sparse_W{W}_in"]["median_ms"]
                    }

    # Summary table
    print("\n=== SUMMARY (median ms, out+in total) ===")
    for N in [200, 400, 600]:
        k = f"N={N}"
        print(f"\n{k}:")
        for method in ["cueq", "dense_bf16", "sparse_W32", "sparse_W64", "sparse_W128", "sparse_W256"]:
            tk = f"{method}_total"
            if tk in results[k]:
                print(f"  {method:20s}: {results[k][tk]['median_ms']:.3f} ms")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print("[profile] Starting full forward profiling...")
    result_json = profile_full.remote()
    print(result_json)

    out_path = Path(__file__).parent / "profile_full_results.json"
    with open(out_path, "w") as f:
        f.write(result_json)
    print(f"\nResults saved to {out_path}")
