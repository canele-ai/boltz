"""Boltz wrapper with ODE sampler support.

Extends the standard boltz_wrapper.py to accept --gamma_0 and --noise_scale
flags. When gamma_0=0, the diffusion sampler becomes a deterministic first-order
Euler ODE solver (no stochastic noise injection).

The monkey-patch works by overriding Boltz2DiffusionParams defaults before
boltz.main.predict() constructs the params dataclass.
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

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz and monkey-patch the diffusion params
    import boltz.main as boltz_main
    from dataclasses import dataclass, field, fields, asdict

    # Monkey-patch Boltz2DiffusionParams to use our gamma_0 and noise_scale
    OrigParams = boltz_main.Boltz2DiffusionParams

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

    print(f"[ode-wrapper] gamma_0={our_args.gamma_0}, noise_scale={our_args.noise_scale}")

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()


if __name__ == "__main__":
    main()
