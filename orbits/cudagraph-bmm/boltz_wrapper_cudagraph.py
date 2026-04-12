"""Boltz wrapper with CUDA graph capture via torch.compile(mode='reduce-overhead').

Extends the triton-pairformer bmm approach to enable CUDA graph capture on the
Pairformer module. With cuequivariance disabled and bmm replacing the einsum
contractions, the entire Pairformer is fully traceable by torch.compile.

Optimizations stacked:
1. ODE sampling (gamma_0=0) -- deterministic first-order Euler solver
2. TF32 matmul precision (matmul_precision="high")
3. bf16 trunk -- removes .float() upcast in triangular_mult.py
4. BMM triangle multiplication -- replaces cuequivariance fused kernel (traceable)
5. torch.compile(mode="reduce-overhead") on Pairformer -- CUDA graph capture

Key insight: mode="reduce-overhead" in torch.compile triggers Inductor's CUDA
graph capture. With cuequivariance gone (no @torch.compiler.disable) and bmm
replacing einsum, the Pairformer has no graph breaks, enabling full CUDA graph
capture of the most expensive module (~26% of GPU time).
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, "/eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch_triangle_mul_bmm_bf16():
    """Replace TriangleMultiplication with bmm + bf16 (no upcast, traceable)."""
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    def triangle_out_fn(a, b):
        B, N, K, D = a.shape
        a_t = a.permute(0, 3, 1, 2).reshape(B * D, N, K)
        b_t = b.permute(0, 3, 1, 2).reshape(B * D, N, K)
        c = torch.bmm(a_t, b_t.transpose(1, 2))
        return c.reshape(B, D, N, N).permute(0, 2, 3, 1)

    def triangle_in_fn(a, b):
        B, K, N, D = a.shape
        a_t = a.permute(0, 3, 1, 2).reshape(B * D, K, N)
        b_t = b.permute(0, 3, 1, 2).reshape(B * D, K, N)
        c = torch.bmm(a_t.transpose(1, 2), b_t)
        return c.reshape(B, D, N, N).permute(0, 2, 3, 1)

    def forward_outgoing(self, x, mask, use_kernels=False):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = triangle_out_fn(a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    def forward_incoming(self, x, mask, use_kernels=False):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = triangle_in_fn(a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    TriangleMultiplicationOutgoing.forward = forward_outgoing
    TriangleMultiplicationIncoming.forward = forward_incoming
    print("[cudagraph-wrapper] Triangle multiplication patched with bmm+bf16")


def patch_compile_pairformer(compile_mode="reduce-overhead", dynamic=False):
    """Hook into model loading to compile the PairformerModule.

    We monkey-patch the PairformerModule.forward and PairformerNoSeqModule.forward
    to wrap them in torch.compile. This is called AFTER model is loaded.
    """
    from boltz.model.layers.pairformer import PairformerModule as PFModule
    from boltz.model.layers.pairformer import PairformerNoSeqModule as PFNoSeqModule

    # Compile the forward methods
    original_pf_forward = PFModule.forward
    compiled_pf_forward = torch.compile(
        original_pf_forward,
        mode=compile_mode,
        dynamic=dynamic,
        fullgraph=False,  # Allow graph breaks if needed, but we expect none
    )
    PFModule.forward = compiled_pf_forward
    print(f"[cudagraph-wrapper] PairformerModule.forward compiled (mode={compile_mode}, dynamic={dynamic})")

    original_pfns_forward = PFNoSeqModule.forward
    compiled_pfns_forward = torch.compile(
        original_pfns_forward,
        mode=compile_mode,
        dynamic=dynamic,
        fullgraph=False,
    )
    PFNoSeqModule.forward = compiled_pfns_forward
    print(f"[cudagraph-wrapper] PairformerNoSeqModule.forward compiled (mode={compile_mode}, dynamic={dynamic})")


def patch_compile_score_model(compile_mode="reduce-overhead", dynamic=False):
    """Compile the entire DiffusionModule (score model).

    More aggressive: compiles the whole score model, not just pairformer.
    This captures more operations in CUDA graphs but may have more graph breaks.
    """
    from boltz.model.modules.diffusion import DiffusionModule

    original_forward = DiffusionModule.forward
    compiled_forward = torch.compile(
        original_forward,
        mode=compile_mode,
        dynamic=dynamic,
        fullgraph=False,
    )
    DiffusionModule.forward = compiled_forward
    print(f"[cudagraph-wrapper] DiffusionModule.forward compiled (mode={compile_mode}, dynamic={dynamic})")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true",
                       help="Keep bf16, no .float() upcast in triangle mul")
    parser.add_argument("--use_bmm", action="store_true",
                       help="Use batched matmul for triangle multiplication")
    parser.add_argument("--compile_pairformer", action="store_true",
                       help="torch.compile the Pairformer (mode=reduce-overhead for CUDA graphs)")
    parser.add_argument("--compile_score", action="store_true",
                       help="torch.compile the entire score model")
    parser.add_argument("--compile_mode", default="reduce-overhead",
                       choices=["reduce-overhead", "default", "max-autotune"],
                       help="torch.compile mode")
    parser.add_argument("--compile_dynamic", action="store_true",
                       help="Enable dynamic shapes in torch.compile")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz and monkey-patch
    import boltz.main as boltz_main
    from dataclasses import dataclass

    # Monkey-patch Boltz2DiffusionParams for ODE mode
    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.0
        noise_scale: float = our_args.noise_scale
        rho: float = 7
        step_scale: float = 1.5
        sigma_min: float = 0.0001
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True

    boltz_main.Boltz2DiffusionParams = PatchedBoltz2DiffusionParams

    @dataclass
    class PatchedBoltzDiffusionParams:
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.107
        noise_scale: float = our_args.noise_scale
        rho: float = 8
        step_scale: float = 1.638
        sigma_min: float = 0.0004
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True
        use_inference_model_cache: bool = True

    boltz_main.BoltzDiffusionParams = PatchedBoltzDiffusionParams

    # Apply bmm + bf16 patches
    if our_args.use_bmm:
        if our_args.bf16_trunk:
            patch_triangle_mul_bmm_bf16()
        else:
            # bmm without bf16 -- still patch but keep float upcast behavior
            # For simplicity, use bf16 version (bmm doesn't need float upcast)
            patch_triangle_mul_bmm_bf16()
            print("[cudagraph-wrapper] NOTE: bmm path inherently avoids float upcast")

    # Apply compile patches
    if our_args.compile_pairformer:
        patch_compile_pairformer(
            compile_mode=our_args.compile_mode,
            dynamic=our_args.compile_dynamic,
        )
    if our_args.compile_score:
        patch_compile_score_model(
            compile_mode=our_args.compile_mode,
            dynamic=our_args.compile_dynamic,
        )

    # Disable cuequivariance (we use bmm instead)
    if our_args.use_bmm:
        boltz_args.append("--no_kernels")
        print("[cudagraph-wrapper] cuequivariance kernels DISABLED (using bmm)")
    else:
        boltz_args.append("--no_kernels")
        print("[cudagraph-wrapper] cuequivariance kernels DISABLED")

    print(f"[cudagraph-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"bmm={our_args.use_bmm}, "
          f"compile_pf={our_args.compile_pairformer}, "
          f"compile_score={our_args.compile_score}, "
          f"compile_mode={our_args.compile_mode}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
