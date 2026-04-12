"""Boltz wrapper with ODE sampler support and cuequivariance kernel support.

Extends the standard boltz_wrapper.py to accept --gamma_0 and --noise_scale
flags for ODE sampling. When gamma_0=0, the diffusion sampler becomes a
deterministic first-order Euler ODE solver.

This wrapper does NOT pass --no_kernels, allowing cuequivariance CUDA kernels
to be used when the environment supports them.
"""
import sys
import argparse
import torch


def main():
    # Extract our custom flags before passing the rest to boltz
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--enable_kernels", action="store_true",
                       help="Enable cuequivariance CUDA kernels (do not pass --no_kernels)")
    parser.add_argument("--no_kernels_flag", action="store_true",
                       help="Explicitly disable kernels (pass --no_kernels to boltz)")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz and monkey-patch the diffusion params
    import boltz.main as boltz_main
    from dataclasses import dataclass

    # Monkey-patch Boltz2DiffusionParams to use our gamma_0 and noise_scale
    @dataclass
    class PatchedBoltz2DiffusionParams:
        """Patched diffusion params with custom gamma_0 and noise_scale."""
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

    # Also patch Boltz1 params in case model=boltz1 is used
    @dataclass
    class PatchedBoltzDiffusionParams:
        """Patched diffusion params for Boltz1."""
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

    # Check if cuequivariance is available
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[kernels-wrapper] cuequivariance_torch available: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False
        print("[kernels-wrapper] cuequivariance_torch NOT available")

    # If --no_kernels_flag is explicitly set, pass it through
    # If --enable_kernels is set and kernels are available, do NOT pass --no_kernels
    # Default behavior: pass --no_kernels if cuequivariance is not available
    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[kernels-wrapper] Kernels DISABLED (--no_kernels_flag)")
    elif our_args.enable_kernels and kernels_available:
        # Do NOT add --no_kernels — kernels are enabled
        print("[kernels-wrapper] Kernels ENABLED via cuequivariance")
    else:
        # Default: disable if not available
        if not kernels_available:
            boltz_args.append("--no_kernels")
            print("[kernels-wrapper] Kernels DISABLED (cuequivariance not installed)")
        else:
            # cuequivariance available but --enable_kernels not set: enable by default
            print("[kernels-wrapper] Kernels ENABLED (cuequivariance available)")

    print(f"[kernels-wrapper] gamma_0={our_args.gamma_0}, noise_scale={our_args.noise_scale}")
    print(f"[kernels-wrapper] matmul_precision={our_args.matmul_precision}")

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()


if __name__ == "__main__":
    main()
