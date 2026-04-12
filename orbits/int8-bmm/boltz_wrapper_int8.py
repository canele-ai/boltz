"""Boltz wrapper with INT8 weight-only quantization via torchao.

Stacked optimizations:
1. ODE sampling (gamma_0=0) -- deterministic first-order Euler solver
2. TF32 matmul precision (matmul_precision="high")
3. bf16 trunk -- removes .float() upcast in triangular_mult.py
4. BMM triangle multiplication -- replaces cuequivariance fused kernel
5. INT8 weight-only quantization via torchao on all Linear layers

Key insight: cuequivariance kernels produce AffineQuantizedTensor-incompatible
custom tensors. With bmm replacement (from triton-pairformer orbit),
cuequivariance is fully bypassed, so ALL Linear layers become standard
nn.Linear and can accept INT8 quantized weights.
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, "/eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch_triangle_mul_bmm_bf16():
    """Replace TriangleMultiplication with batched matmul, keep bf16."""
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
    print("[int8-wrapper] Triangle multiplication patched with bmm+bf16")


def apply_int8_quantization(model, skip_patterns=None, mode="weight_only"):
    """Apply INT8 quantization to all eligible Linear layers.

    Args:
        model: The Boltz model to quantize
        skip_patterns: List of submodule name patterns to skip
        mode: "weight_only" for W8A16, "dynamic" for W8A8 dynamic quantization
    Returns:
        dict with quantization stats
    """
    from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight

    skip_patterns = skip_patterns or []

    # Count layers before
    total_linear = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            total_linear += 1

    # Define filter function to skip certain layers if needed
    def filter_fn(module, fqn):
        for pattern in skip_patterns:
            if pattern in fqn:
                return False
        return True

    if mode == "dynamic":
        quant_config = int8_dynamic_activation_int8_weight()
        mode_label = "W8A8-dynamic"
    else:
        quant_config = int8_weight_only()
        mode_label = "W8A16-weight-only"

    try:
        quantize_(model, quant_config, filter_fn=filter_fn)
        status = "success"
        error = None
    except Exception as e:
        status = "failed"
        error = str(e)

    # Count quantized layers
    quantized = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
            if hasattr(w, 'layout_type') or 'AffineQuantized' in type(w).__name__:
                quantized += 1

    stats = {
        "total_linear": total_linear,
        "quantized": quantized,
        "mode": mode_label,
        "status": status,
        "error": error,
    }
    print(f"[int8-wrapper] Quantization ({mode_label}): {quantized}/{total_linear} layers, status={status}")
    if error:
        print(f"[int8-wrapper] Error: {error}")
    return stats


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--apply_int8", action="store_true",
                       help="Apply INT8 weight-only quantization via torchao")
    parser.add_argument("--int8_mode", type=str, default="weight_only",
                       choices=["weight_only", "dynamic"],
                       help="INT8 mode: weight_only (W8A16) or dynamic (W8A8)")
    parser.add_argument("--skip_patterns", type=str, default="",
                       help="Comma-separated patterns of layer names to skip for INT8")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Import boltz and monkey-patch
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

    # Apply BMM triangle multiplication patch (bypasses cuequivariance)
    if our_args.bf16_trunk:
        patch_triangle_mul_bmm_bf16()

    # Disable cuequivariance kernels
    boltz_args.append("--no_kernels")
    print("[int8-wrapper] cuequivariance kernels DISABLED (using bmm)")

    # If INT8 requested, hook into model loading to quantize after load.
    # Boltz uses LightningModule.load_from_checkpoint -> model.eval().
    # We monkey-patch Boltz2.load_from_checkpoint to intercept.
    if our_args.apply_int8:
        skip_patterns = [p.strip() for p in our_args.skip_patterns.split(",") if p.strip()]

        from boltz.model.models.boltz2 import Boltz2

        _original_load_from_checkpoint = Boltz2.load_from_checkpoint

        def _patched_load_from_checkpoint(*args, **kwargs):
            print("[int8-wrapper] Intercepting model load for INT8 quantization...")
            model = _original_load_from_checkpoint(*args, **kwargs)
            print("[int8-wrapper] Model loaded on CPU, moving to CUDA for quantization...")
            model = model.cuda()
            stats = apply_int8_quantization(model, skip_patterns=skip_patterns, mode=our_args.int8_mode)
            print(f"[int8-wrapper] INT8 stats: {stats}")
            # Move back to CPU so Lightning's device placement works normally
            model = model.cpu()
            return model

        Boltz2.load_from_checkpoint = _patched_load_from_checkpoint
        print(f"[int8-wrapper] INT8 quantization ENABLED (skip: {skip_patterns})")

    print(f"[int8-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"int8={our_args.apply_int8}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
