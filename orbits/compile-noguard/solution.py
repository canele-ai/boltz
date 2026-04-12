"""Boltz wrapper with compile-noguard: torch.compile without inference guards.

Extends the eval-v2-winner stacked wrapper (ODE + TF32 + bf16) with:
4. torch.compile on pairformer, MSA, structure (score) model, and confidence model
   WITHOUT the _orig_mod fallback that normally undoes compilation at inference time.

The Boltz2 model has a critical anti-pattern: during inference, all compiled modules
revert to _orig_mod, explicitly undoing torch.compile. This wrapper fixes that by
applying torch.compile AFTER model loading, without setting the is_*_compiled flags
that trigger the fallback.

Architecture:
  - Monkey-patches Boltz2's eval() method to apply torch.compile post-load
  - The is_*_compiled flags stay False, so forward() uses compiled modules directly
  - Inherits ODE (gamma_0=0), TF32, and bf16 trunk from parent orbit
"""
import sys
import argparse
import torch


def patch_triangular_mult_bf16():
    """Remove .float() upcast in triangular_mult.py for bf16 trunk.

    The standard boltz triangular_mult.py casts to float32 before the
    einsum in both TriangleMultiplicationOutgoing and
    TriangleMultiplicationIncoming. When running in bf16 mixed precision,
    this upcast is unnecessary and slows down the trunk.
    """
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    def forward_outgoing_bf16(self, x, mask, use_kernels=False):
        if use_kernels:
            from boltz.model.layers.triangular_mult import kernel_triangular_mult
            return kernel_triangular_mult(
                x,
                direction="outgoing",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = torch.einsum("bikd,bjkd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    def forward_incoming_bf16(self, x, mask, use_kernels=False):
        if use_kernels:
            from boltz.model.layers.triangular_mult import kernel_triangular_mult
            return kernel_triangular_mult(
                x,
                direction="incoming",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        x = torch.einsum("bkid,bkjd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    TriangleMultiplicationOutgoing.forward = forward_outgoing_bf16
    TriangleMultiplicationIncoming.forward = forward_incoming_bf16
    print("[compile-noguard] bf16 trunk patch applied")


def patch_compile_noguard(compile_pairformer=False, compile_msa=False,
                          compile_structure=False, compile_confidence=False,
                          compile_mode="default"):
    """Monkey-patch Boltz2 to apply torch.compile without inference guards.

    Strategy: We wrap Boltz2.eval() to apply torch.compile AFTER the model
    is loaded from checkpoint. The key insight is that we do NOT set the
    is_*_compiled flags, so the forward() method uses the compiled modules
    directly instead of falling back to _orig_mod.

    Parameters
    ----------
    compile_pairformer : bool
        Compile the PairformerModule (trunk, runs recycling_steps+1 times)
    compile_msa : bool
        Compile the MSAModule (trunk)
    compile_structure : bool
        Compile the DiffusionModule score model (runs sampling_steps times)
    compile_confidence : bool
        Compile the ConfidenceModule
    compile_mode : str
        torch.compile mode: "default", "reduce-overhead", or "max-autotune"
    """
    from boltz.model.models.boltz2 import Boltz2

    _original_eval = Boltz2.eval
    _compile_applied = [False]  # mutable to allow closure modification

    def patched_eval(self):
        """Apply torch.compile on first eval() call, then delegate to original."""
        result = _original_eval(self)

        if not _compile_applied[0]:
            _compile_applied[0] = True
            compile_kwargs = dict(dynamic=False, fullgraph=False, mode=compile_mode)

            if compile_pairformer and hasattr(self, 'pairformer_module'):
                print(f"[compile-noguard] Compiling pairformer_module (mode={compile_mode})")
                self.pairformer_module = torch.compile(
                    self.pairformer_module, **compile_kwargs
                )
                # Do NOT set is_pairformer_compiled=True -- that triggers _orig_mod fallback

            if compile_msa and hasattr(self, 'msa_module'):
                print(f"[compile-noguard] Compiling msa_module (mode={compile_mode})")
                self.msa_module = torch.compile(
                    self.msa_module, **compile_kwargs
                )

            if compile_structure and hasattr(self, 'structure_module'):
                # The score model is inside structure_module.score_model
                if hasattr(self.structure_module, 'score_model'):
                    print(f"[compile-noguard] Compiling structure score_model (mode={compile_mode})")
                    self.structure_module.score_model = torch.compile(
                        self.structure_module.score_model, **compile_kwargs
                    )

            if compile_confidence and hasattr(self, 'confidence_module'):
                print(f"[compile-noguard] Compiling confidence_module (mode={compile_mode})")
                self.confidence_module = torch.compile(
                    self.confidence_module, **compile_kwargs
                )

        return result

    Boltz2.eval = patched_eval
    print(f"[compile-noguard] Patched Boltz2.eval() for compile without guards")
    print(f"[compile-noguard]   pairformer={compile_pairformer}, msa={compile_msa}, "
          f"structure={compile_structure}, confidence={compile_confidence}, mode={compile_mode}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true",
                       help="Remove .float() upcast in triangular_mult for bf16")
    parser.add_argument("--enable_kernels", action="store_true",
                       help="Enable cuequivariance CUDA kernels")
    parser.add_argument("--no_kernels_flag", action="store_true",
                       help="Explicitly disable kernels")
    # Compile flags
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--compile_mode", default="default",
                       choices=["default", "reduce-overhead", "max-autotune"])

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Increase dynamo cache for large models
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512

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

    # Apply bf16 trunk patch if requested
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Apply compile patches
    any_compile = (our_args.compile_pairformer or our_args.compile_msa or
                   our_args.compile_structure or our_args.compile_confidence)
    if any_compile:
        patch_compile_noguard(
            compile_pairformer=our_args.compile_pairformer,
            compile_msa=our_args.compile_msa,
            compile_structure=our_args.compile_structure,
            compile_confidence=our_args.compile_confidence,
            compile_mode=our_args.compile_mode,
        )

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[compile-noguard] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False
        print("[compile-noguard] cuequivariance_torch NOT available")

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[compile-noguard] Kernels DISABLED")
    elif our_args.enable_kernels and kernels_available:
        print("[compile-noguard] Kernels ENABLED")
    else:
        if not kernels_available:
            boltz_args.append("--no_kernels")
            print("[compile-noguard] Kernels DISABLED (not installed)")
        else:
            print("[compile-noguard] Kernels ENABLED (default)")

    print(f"[compile-noguard] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"compile_mode={our_args.compile_mode}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
