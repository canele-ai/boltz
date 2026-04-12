"""Boltz wrapper with fast model loading via safetensors.

Extends the stacked wrapper (ODE + TF32 + bf16) with:
- Safetensors loading from pre-saved weights on Modal volume
- Direct GPU loading (skip CPU->GPU copy)
- Falls back to standard checkpoint if safetensors not available

Usage:
    python boltz_wrapper_fast.py input.yaml --out_dir out \
        --sampling_steps 12 --recycling_steps 0 \
        --gamma_0 0.0 --matmul_precision high --bf16_trunk \
        --weights_dir /weights
"""
import sys
import argparse
import time
import json
from pathlib import Path

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
                x, direction="outgoing", mask=mask,
                norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight, g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight,
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
                x, direction="incoming", mask=mask,
                norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight, g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight,
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
    print("[fast-wrapper] bf16 trunk patch applied")


def patch_model_loading(weights_dir: str):
    """Monkey-patch Boltz2.load_from_checkpoint to use safetensors.

    This replaces the standard Lightning checkpoint loading with:
    1. safetensors load_file (zero-copy mmap, direct GPU)
    2. Boltz2(**hparams) + load_state_dict

    Falls back to standard loading if safetensors not available.
    """
    from boltz.model.models.boltz2 import Boltz2

    sf_path = Path(weights_dir) / "boltz2.safetensors"
    hparams_path = Path(weights_dir) / "boltz2_hparams.json"

    if not sf_path.exists() or not hparams_path.exists():
        print(f"[fast-wrapper] Safetensors not found at {weights_dir}, using standard loading")
        return

    with open(hparams_path) as f:
        saved_hparams = json.load(f)

    original_load = Boltz2.load_from_checkpoint

    @classmethod
    def fast_load_from_checkpoint(cls, checkpoint_path, strict=True, map_location="cpu", **kwargs):
        """Fast loading via safetensors."""
        from safetensors.torch import load_file

        t0 = time.perf_counter()

        # Merge saved hparams with runtime kwargs
        hparams = dict(saved_hparams)
        for k, v in kwargs.items():
            if k in hparams:
                hparams[k] = v

        # Determine device
        device = "cuda:0" if map_location != "cpu" else "cpu"

        # Load safetensors (zero-copy for CPU, direct for GPU)
        sd = load_file(str(sf_path), device=device)
        t_load = time.perf_counter() - t0

        # Instantiate model and load state dict
        model = cls(**hparams)
        model.load_state_dict(sd, strict=strict)

        t_total = time.perf_counter() - t0
        print(f"[fast-wrapper] Safetensors load: {t_load:.2f}s (tensors) + "
              f"{t_total - t_load:.2f}s (model init) = {t_total:.2f}s total")
        return model

    Boltz2.load_from_checkpoint = fast_load_from_checkpoint
    print(f"[fast-wrapper] Patched Boltz2.load_from_checkpoint -> safetensors ({sf_path})")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="high",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.0)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--weights_dir", default="/weights",
                       help="Directory with safetensors weights")

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

    # Apply bf16 trunk patch
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Patch model loading to use safetensors
    patch_model_loading(our_args.weights_dir)

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
    except ImportError:
        kernels_available = False

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
    elif not our_args.enable_kernels or not kernels_available:
        if not kernels_available:
            boltz_args.append("--no_kernels")

    print(f"[fast-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"weights_dir={our_args.weights_dir}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
