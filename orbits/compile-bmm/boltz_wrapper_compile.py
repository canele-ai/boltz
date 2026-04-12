"""Boltz wrapper with torch.compile on bmm-based Pairformer.

Extends the triton-pairformer approach: replaces cuequivariance TriangleMultiplication
with batched matmul (fully traceable), then applies torch.compile to the Pairformer.

Optimizations stacked:
1. ODE sampling (gamma_0=0) -- deterministic first-order Euler solver
2. TF32 matmul precision (matmul_precision="high")
3. bf16 trunk -- removes .float() upcast in triangular_mult.py
4. bmm-based triangle multiplication -- replaces cuequivariance fused kernel
5. torch.compile on Pairformer -- fuses LayerNorm+Linear+sigmoid+bmm+gate

The hypothesis: with cuequivariance removed (its @torch.compiler.disable blocked
tracing), inductor can now trace the full Pairformer and fuse ops to recover
the ~8% gap from using bmm instead of cuequivariance kernels.
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch_triangle_mul_bmm_bf16():
    """Replace TriangleMultiplication with bmm-based implementation (bf16, no upcast).

    This is the same bmm path from triton-pairformer, kept inline to avoid
    cross-orbit imports. The bmm path is fully traceable by torch.compile.
    """
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
    print("[compile-bmm] Triangle multiplication patched with bmm+bf16 (traceable)")


def apply_compile_to_pairformer(model_module, compile_mode="default"):
    """Apply torch.compile to the pairformer module after model loading.

    Args:
        model_module: The loaded Boltz2 model
        compile_mode: One of "default", "reduce-overhead", "max-autotune"
    """
    import torch._dynamo

    # Reset dynamo state for clean compilation
    torch._dynamo.reset()

    pairformer = model_module.pairformer_module
    print(f"[compile-bmm] Compiling pairformer with mode='{compile_mode}'...")
    print(f"[compile-bmm] Pairformer type: {type(pairformer).__name__}")

    compiled = torch.compile(
        pairformer,
        mode=compile_mode,
        dynamic=False,
        fullgraph=False,  # Allow graph breaks if needed
    )
    model_module.pairformer_module = compiled
    model_module.is_pairformer_compiled = True
    print(f"[compile-bmm] Pairformer compiled successfully (mode={compile_mode})")
    return model_module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true",
                       help="Keep bf16, no .float() upcast in triangle mul")
    parser.add_argument("--compile_mode", default="none",
                       choices=["none", "default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode for the Pairformer")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz and monkey-patch
    import boltz.main as boltz_main
    from dataclasses import dataclass

    # Monkey-patch diffusion params for ODE mode
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

    # Apply bmm triangle multiplication patch
    if our_args.bf16_trunk:
        patch_triangle_mul_bmm_bf16()

    # Always disable cuequivariance kernels (we use bmm instead)
    boltz_args.append("--no_kernels")
    print("[compile-bmm] cuequivariance kernels DISABLED (using bmm)")

    # Hook into predict to apply torch.compile AFTER model loading
    compile_mode = our_args.compile_mode
    if compile_mode != "none":
        _orig_predict = boltz_main.predict

        def _patched_predict(*args, **kwargs):
            # Monkey-patch Boltz2's eval() to apply compile after model is loaded
            # This is cleaner than intercepting load_from_checkpoint
            from boltz.model.models.boltz2 import Boltz2
            _orig_eval = Boltz2.eval

            def _compiled_eval(self):
                result = _orig_eval(self)
                if not getattr(self, '_compile_applied', False):
                    self._compile_applied = True
                    apply_compile_to_pairformer(self, compile_mode=compile_mode)
                return result

            Boltz2.eval = _compiled_eval
            try:
                return _orig_predict(*args, **kwargs)
            finally:
                Boltz2.eval = _orig_eval

        boltz_main.predict = _patched_predict

    print(f"[compile-bmm] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"compile_mode={compile_mode}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
