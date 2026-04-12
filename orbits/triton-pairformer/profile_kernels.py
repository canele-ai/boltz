"""Profile and validate Triton triangle multiplication kernels.

Compares three implementations:
1. PyTorch einsum (reference)
2. Batched matmul (torch.bmm permutation trick)
3. Triton kernel

Also profiles cuequivariance if available.

Usage:
    modal run orbits/triton-pairformer/profile_kernels.py
    modal run orbits/triton-pairformer/profile_kernels.py --mode correctness
    modal run orbits/triton-pairformer/profile_kernels.py --mode benchmark
    modal run orbits/triton-pairformer/profile_kernels.py --mode profile-full
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ORBIT_DIR = Path(__file__).resolve().parent

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "triton>=2.2.0",
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
    .add_local_file(
        str(ORBIT_DIR / "triton_triangle_mul.py"),
        remote_path="/work/triton_triangle_mul.py",
    )
)

app = modal.App("triton-pairformer-profile", image=boltz_image)


def _benchmark_fn(fn, warmup=5, repeats=20):
    """Benchmark a function using CUDA events."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": sorted(times)[len(times) // 2],
    }


@app.function(gpu="L40S", timeout=1200)
def run_correctness():
    """Validate Triton kernels match PyTorch einsum reference."""
    import sys
    sys.path.insert(0, "/work")
    import torch
    from triton_triangle_mul import (
        triton_triangle_mul_outgoing,
        triton_triangle_mul_incoming,
        triangle_mul_matmul_outgoing,
        triangle_mul_matmul_incoming,
    )

    results = {}
    torch.manual_seed(42)

    for N in [64, 200, 400, 600]:
        B, D = 1, 128
        K = N  # In TriangleMult, K=N (pair representation is NxN)

        for dtype_name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
            key = f"N={N},dtype={dtype_name}"
            print(f"\n--- Correctness: {key} ---")

            # Outgoing: bikd,bjkd->bijd
            a = torch.randn(B, N, K, D, device="cuda", dtype=dtype)
            b = torch.randn(B, N, K, D, device="cuda", dtype=dtype)

            ref = torch.einsum("bikd,bjkd->bijd", a.float(), b.float())
            matmul_out = triangle_mul_matmul_outgoing(a.float(), b.float())
            triton_out = triton_triangle_mul_outgoing(a.float(), b.float())

            # Check matmul
            matmul_err = (ref - matmul_out).abs().max().item()
            matmul_rel = ((ref - matmul_out).abs() / (ref.abs() + 1e-8)).max().item()

            # Check triton
            triton_err = (ref - triton_out).abs().max().item()
            triton_rel = ((ref - triton_out).abs() / (ref.abs() + 1e-8)).max().item()

            entry = {
                "outgoing": {
                    "matmul_max_abs_err": matmul_err,
                    "matmul_max_rel_err": matmul_rel,
                    "triton_max_abs_err": triton_err,
                    "triton_max_rel_err": triton_rel,
                    "matmul_pass": matmul_err < 1e-3,
                    "triton_pass": triton_err < 1e-3,
                },
            }

            print(f"  Outgoing matmul: abs_err={matmul_err:.2e}, rel_err={matmul_rel:.2e}")
            print(f"  Outgoing triton: abs_err={triton_err:.2e}, rel_err={triton_rel:.2e}")

            # Incoming: bkid,bkjd->bijd
            a_in = torch.randn(B, K, N, D, device="cuda", dtype=dtype)
            b_in = torch.randn(B, K, N, D, device="cuda", dtype=dtype)

            ref_in = torch.einsum("bkid,bkjd->bijd", a_in.float(), b_in.float())
            matmul_in = triangle_mul_matmul_incoming(a_in.float(), b_in.float())
            triton_in = triton_triangle_mul_incoming(a_in.float(), b_in.float())

            matmul_err_in = (ref_in - matmul_in).abs().max().item()
            triton_err_in = (ref_in - triton_in).abs().max().item()

            entry["incoming"] = {
                "matmul_max_abs_err": matmul_err_in,
                "triton_max_abs_err": triton_err_in,
                "matmul_pass": matmul_err_in < 1e-3,
                "triton_pass": triton_err_in < 1e-3,
            }

            print(f"  Incoming matmul: abs_err={matmul_err_in:.2e}")
            print(f"  Incoming triton: abs_err={triton_err_in:.2e}")

            results[key] = entry

    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=1200)
def run_benchmark():
    """Benchmark all implementations on typical shapes."""
    import sys
    sys.path.insert(0, "/work")
    import torch
    from triton_triangle_mul import (
        triton_triangle_mul_outgoing,
        triton_triangle_mul_incoming,
        triangle_mul_matmul_outgoing,
        triangle_mul_matmul_incoming,
    )

    results = {}
    torch.manual_seed(42)

    for N in [200, 400, 600]:
        B, D = 1, 128
        K = N

        key = f"N={N}"
        print(f"\n=== Benchmark: {key} ===")
        entry = {}

        # --- Outgoing ---
        a = torch.randn(B, N, K, D, device="cuda", dtype=torch.float32)
        b = torch.randn(B, N, K, D, device="cuda", dtype=torch.float32)

        print("  Outgoing einsum...")
        entry["outgoing_einsum"] = _benchmark_fn(
            lambda: torch.einsum("bikd,bjkd->bijd", a, b)
        )
        print(f"    {entry['outgoing_einsum']['mean_ms']:.2f} ms")

        print("  Outgoing matmul...")
        entry["outgoing_matmul"] = _benchmark_fn(
            lambda: triangle_mul_matmul_outgoing(a, b)
        )
        print(f"    {entry['outgoing_matmul']['mean_ms']:.2f} ms")

        print("  Outgoing triton...")
        entry["outgoing_triton"] = _benchmark_fn(
            lambda: triton_triangle_mul_outgoing(a, b)
        )
        print(f"    {entry['outgoing_triton']['mean_ms']:.2f} ms")

        # --- Incoming ---
        a_in = torch.randn(B, K, N, D, device="cuda", dtype=torch.float32)
        b_in = torch.randn(B, K, N, D, device="cuda", dtype=torch.float32)

        print("  Incoming einsum...")
        entry["incoming_einsum"] = _benchmark_fn(
            lambda: torch.einsum("bkid,bkjd->bijd", a_in, b_in)
        )
        print(f"    {entry['incoming_einsum']['mean_ms']:.2f} ms")

        print("  Incoming matmul...")
        entry["incoming_matmul"] = _benchmark_fn(
            lambda: triangle_mul_matmul_incoming(a_in, b_in)
        )
        print(f"    {entry['incoming_matmul']['mean_ms']:.2f} ms")

        print("  Incoming triton...")
        entry["incoming_triton"] = _benchmark_fn(
            lambda: triton_triangle_mul_incoming(a_in, b_in)
        )
        print(f"    {entry['incoming_triton']['mean_ms']:.2f} ms")

        # Try cuequivariance full-module comparison if available
        try:
            from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
            # We can't easily benchmark the fused kernel without the full module weights
            # Just note it's available
            entry["cuequivariance_available"] = True
        except ImportError:
            entry["cuequivariance_available"] = False

        results[key] = entry

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (mean ms)")
    print("=" * 80)
    print(f"{'N':>4} | {'Dir':>8} | {'einsum':>10} | {'matmul':>10} | {'triton':>10} | {'triton/ein':>10}")
    print("-" * 80)
    for N in [200, 400, 600]:
        key = f"N={N}"
        if key not in results:
            continue
        e = results[key]
        for direction in ["outgoing", "incoming"]:
            ein = e[f"{direction}_einsum"]["mean_ms"]
            mat = e[f"{direction}_matmul"]["mean_ms"]
            tri = e[f"{direction}_triton"]["mean_ms"]
            ratio = tri / ein
            print(f"{N:>4} | {direction:>8} | {ein:>10.2f} | {mat:>10.2f} | {tri:>10.2f} | {ratio:>10.2f}x")

    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=1800)
def run_full_profile():
    """Profile a full Boltz-2 forward pass to measure cuequivariance kernel time.

    This loads the actual model and hooks into the Pairformer layers to measure
    the time spent in each triangle operation.
    """
    import sys
    sys.path.insert(0, "/work")
    import torch
    import time as time_mod

    results = {"env": {}}

    results["env"]["torch_version"] = torch.__version__
    results["env"]["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        results["env"]["gpu_name"] = torch.cuda.get_device_name(0)

    try:
        import cuequivariance_torch
        results["env"]["cuequivariance_torch"] = cuequivariance_torch.__version__
    except ImportError:
        results["env"]["cuequivariance_torch"] = None

    # Profile the triangle ops in isolation with model-realistic shapes
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )
    from triton_triangle_mul import (
        triton_triangle_mul_outgoing,
        triton_triangle_mul_incoming,
        triangle_mul_matmul_outgoing,
        triangle_mul_matmul_incoming,
    )

    torch.manual_seed(42)
    profile_results = {}

    for N in [200, 400, 600]:
        B, D = 1, 128
        key = f"N={N}"
        print(f"\n=== Full module profile: {key} ===")

        # Create module instances
        out_module = TriangleMultiplicationOutgoing(dim=D).cuda().eval()
        in_module = TriangleMultiplicationIncoming(dim=D).cuda().eval()

        x = torch.randn(B, N, N, D, device="cuda")
        mask = torch.ones(B, N, N, device="cuda")

        entry = {}

        # Benchmark cuequivariance path (use_kernels=True)
        try:
            with torch.no_grad():
                print("  Cuequivariance outgoing...")
                entry["cueq_outgoing"] = _benchmark_fn(
                    lambda: out_module(x, mask, use_kernels=True),
                    warmup=3, repeats=10,
                )
                print(f"    {entry['cueq_outgoing']['mean_ms']:.2f} ms")

                print("  Cuequivariance incoming...")
                entry["cueq_incoming"] = _benchmark_fn(
                    lambda: in_module(x, mask, use_kernels=True),
                    warmup=3, repeats=10,
                )
                print(f"    {entry['cueq_incoming']['mean_ms']:.2f} ms")
        except Exception as e:
            print(f"  Cuequivariance failed: {e}")
            entry["cueq_error"] = str(e)

        # Benchmark PyTorch path (use_kernels=False)
        with torch.no_grad():
            print("  PyTorch outgoing...")
            entry["pytorch_outgoing"] = _benchmark_fn(
                lambda: out_module(x, mask, use_kernels=False),
                warmup=3, repeats=10,
            )
            print(f"    {entry['pytorch_outgoing']['mean_ms']:.2f} ms")

            print("  PyTorch incoming...")
            entry["pytorch_incoming"] = _benchmark_fn(
                lambda: in_module(x, mask, use_kernels=False),
                warmup=3, repeats=10,
            )
            print(f"    {entry['pytorch_incoming']['mean_ms']:.2f} ms")

        # Benchmark with matmul replacement
        # Monkey-patch the einsum with matmul
        import types

        def _make_outgoing_matmul_forward(mod):
            def forward(self, x, mask, use_kernels=False):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x.float(), 2, dim=-1)
                x = triangle_mul_matmul_outgoing(a, b)
                x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
                return x
            return types.MethodType(forward, mod)

        def _make_incoming_matmul_forward(mod):
            def forward(self, x, mask, use_kernels=False):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x.float(), 2, dim=-1)
                x = triangle_mul_matmul_incoming(a, b)
                x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
                return x
            return types.MethodType(forward, mod)

        def _make_outgoing_triton_forward(mod):
            def forward(self, x, mask, use_kernels=False):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x.float(), 2, dim=-1)
                x = triton_triangle_mul_outgoing(a, b)
                x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
                return x
            return types.MethodType(forward, mod)

        def _make_incoming_triton_forward(mod):
            def forward(self, x, mask, use_kernels=False):
                x = self.norm_in(x)
                x_in = x
                x = self.p_in(x) * self.g_in(x).sigmoid()
                x = x * mask.unsqueeze(-1)
                a, b = torch.chunk(x.float(), 2, dim=-1)
                x = triton_triangle_mul_incoming(a, b)
                x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
                return x
            return types.MethodType(forward, mod)

        # Create fresh modules for matmul test
        out_matmul = TriangleMultiplicationOutgoing(dim=D).cuda().eval()
        in_matmul = TriangleMultiplicationIncoming(dim=D).cuda().eval()
        out_matmul.forward = _make_outgoing_matmul_forward(out_matmul)
        in_matmul.forward = _make_incoming_matmul_forward(in_matmul)

        with torch.no_grad():
            print("  Matmul outgoing (full module)...")
            entry["matmul_outgoing"] = _benchmark_fn(
                lambda: out_matmul(x, mask),
                warmup=3, repeats=10,
            )
            print(f"    {entry['matmul_outgoing']['mean_ms']:.2f} ms")

            print("  Matmul incoming (full module)...")
            entry["matmul_incoming"] = _benchmark_fn(
                lambda: in_matmul(x, mask),
                warmup=3, repeats=10,
            )
            print(f"    {entry['matmul_incoming']['mean_ms']:.2f} ms")

        # Create fresh modules for triton test
        out_triton = TriangleMultiplicationOutgoing(dim=D).cuda().eval()
        in_triton = TriangleMultiplicationIncoming(dim=D).cuda().eval()
        out_triton.forward = _make_outgoing_triton_forward(out_triton)
        in_triton.forward = _make_incoming_triton_forward(in_triton)

        with torch.no_grad():
            print("  Triton outgoing (full module)...")
            entry["triton_outgoing"] = _benchmark_fn(
                lambda: out_triton(x, mask),
                warmup=3, repeats=10,
            )
            print(f"    {entry['triton_outgoing']['mean_ms']:.2f} ms")

            print("  Triton incoming (full module)...")
            entry["triton_incoming"] = _benchmark_fn(
                lambda: in_triton(x, mask),
                warmup=3, repeats=10,
            )
            print(f"    {entry['triton_incoming']['mean_ms']:.2f} ms")

        profile_results[key] = entry

    # Summary table
    print("\n" + "=" * 90)
    print("FULL MODULE PROFILE SUMMARY (mean ms)")
    print("=" * 90)
    print(f"{'N':>4} | {'Dir':>8} | {'cueq':>8} | {'pytorch':>8} | {'matmul':>8} | {'triton':>8} | {'tri/cueq':>8}")
    print("-" * 90)
    for N in [200, 400, 600]:
        key = f"N={N}"
        if key not in profile_results:
            continue
        e = profile_results[key]
        for direction in ["outgoing", "incoming"]:
            cueq = e.get(f"cueq_{direction}", {}).get("mean_ms", float("nan"))
            pytorch = e.get(f"pytorch_{direction}", {}).get("mean_ms", float("nan"))
            matmul = e.get(f"matmul_{direction}", {}).get("mean_ms", float("nan"))
            triton = e.get(f"triton_{direction}", {}).get("mean_ms", float("nan"))
            ratio = triton / cueq if cueq and cueq > 0 else float("nan")
            print(f"{N:>4} | {direction:>8} | {cueq:>8.2f} | {pytorch:>8.2f} | {matmul:>8.2f} | {triton:>8.2f} | {ratio:>8.2f}x")

    results["profile"] = profile_results
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(mode: str = "all"):
    if mode in ("correctness", "all"):
        print("\n" + "=" * 60)
        print("CORRECTNESS VALIDATION")
        print("=" * 60)
        result = run_correctness.remote()
        print(result)

    if mode in ("benchmark", "all"):
        print("\n" + "=" * 60)
        print("KERNEL BENCHMARK (einsum only)")
        print("=" * 60)
        result = run_benchmark.remote()
        print(result)

    if mode in ("profile-full", "all"):
        print("\n" + "=" * 60)
        print("FULL MODULE PROFILE")
        print("=" * 60)
        result = run_full_profile.remote()
        print(result)
