"""Boltz wrapper with torch.compile on score model + stacked optimizations.

Extends the eval-v2-winner stacked wrapper to apply torch.compile to ONLY
the diffusion score model (the hot inner loop). This avoids the guard
failures that plagued earlier compile attempts which targeted multiple
modules with different dynamic shapes.

The score model has fixed tensor shapes at every diffusion step, making it
an ideal compile target.

Optimizations applied:
1. ODE sampling (gamma_0=0) — deterministic first-order Euler solver
2. TF32 matmul precision (matmul_precision="high")
3. bf16 trunk — removes .float() upcast in triangular_mult.py
4. torch.compile on score model — compiles the 24-layer token transformer
"""
import sys
import argparse
import torch


def patch_triangular_mult_bf16():
    """Remove .float() upcast in triangular_mult.py for bf16 trunk."""
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
    print("[compile-wrapper] bf16 trunk patch applied")


def patch_compile_score_model(compile_mode="default", suppress_errors=False):
    """Monkey-patch Boltz to apply torch.compile to the score model after loading.

    Instead of modifying the checkpoint loading, we patch the predict function
    to compile the score model after the model is loaded.

    Parameters
    ----------
    compile_mode : str
        "default" — torch.compile(dynamic=False, fullgraph=False)
        "reduce-overhead" — uses CUDA graphs for minimal kernel launch overhead
        "max-autotune" — maximum optimization (longer compile time)
    suppress_errors : bool
        If True, set torch._dynamo.config.suppress_errors = True
    """
    import os
    import boltz.main as boltz_main

    # Enable FX graph cache for persistent compiled graph caching across processes.
    # This avoids recompilation when the same shapes are seen again.
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_compile_cache")
    print(f"[compile-wrapper] FX graph cache enabled at {os.environ['TORCHINDUCTOR_CACHE_DIR']}")

    if suppress_errors:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        print("[compile-wrapper] dynamo suppress_errors=True")

    original_predict = boltz_main.predict

    def patched_predict(*args, **kwargs):
        # Intercept the Trainer.predict call to compile the score model
        from pytorch_lightning import Trainer

        original_trainer_predict = Trainer.predict

        def compiled_trainer_predict(self, model, *a, **kw):
            import torch as _torch
            # model is the LightningModule (Boltz2). Compile its score model.
            if hasattr(model, 'structure_module') and hasattr(model.structure_module, 'score_model'):
                print(f"[compile-wrapper] Compiling score model with mode={compile_mode}")
                compile_kwargs = {
                    "dynamic": False,
                    "fullgraph": False,
                }
                if compile_mode != "default":
                    compile_kwargs["mode"] = compile_mode

                model.structure_module.score_model = _torch.compile(
                    model.structure_module.score_model,
                    **compile_kwargs,
                )
                print("[compile-wrapper] Score model compiled successfully")
            else:
                print("[compile-wrapper] WARNING: could not find score_model on model")

            return original_trainer_predict(self, model, *a, **kw)

        Trainer.predict = compiled_trainer_predict
        try:
            return original_predict(*args, **kwargs)
        finally:
            Trainer.predict = original_trainer_predict

    boltz_main.predict = patched_predict


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
    parser.add_argument("--compile_score", action="store_true",
                       help="Apply torch.compile to the diffusion score model")
    parser.add_argument("--compile_mode", default="default",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode")
    parser.add_argument("--suppress_errors", action="store_true",
                       help="Set torch._dynamo.config.suppress_errors=True")
    parser.add_argument("--warmup_runs", type=int, default=0,
                       help="Number of warmup runs before measured run (for compile amortization)")

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

    # Apply bf16 trunk patch if requested
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Apply torch.compile to score model if requested
    if our_args.compile_score:
        patch_compile_score_model(
            compile_mode=our_args.compile_mode,
            suppress_errors=our_args.suppress_errors,
        )

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[compile-wrapper] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False
        print("[compile-wrapper] cuequivariance_torch NOT available")

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[compile-wrapper] Kernels DISABLED")
    elif our_args.enable_kernels and kernels_available:
        print("[compile-wrapper] Kernels ENABLED")
    else:
        if not kernels_available:
            boltz_args.append("--no_kernels")
            print("[compile-wrapper] Kernels DISABLED (not installed)")
        else:
            print("[compile-wrapper] Kernels ENABLED (default)")

    print(f"[compile-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"compile_score={our_args.compile_score}, "
          f"compile_mode={our_args.compile_mode}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
