"""Boltz wrapper with SDPA attention + stacked optimizations (ODE + TF32 + bf16).

Extends the eval-v2-winner wrapper to also monkey-patch AttentionPairBias
in both attention.py and attentionv2.py to use
torch.nn.functional.scaled_dot_product_attention.

The pair bias prevents FlashAttention-2 dispatch, but SDPA's memory-efficient
backend (xformers-style) can still be faster than manual einsum when using bf16
inputs due to fused kernel execution.

Key insight vs flash-sdpa orbit (#5): eval-v3 uses cached MSAs, so GPU-level
speedups from SDPA are now cleanly measurable without MSA latency noise.
"""
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from torch import Tensor


def patch_attention_sdpa():
    """Replace AttentionPairBias.forward in both attention.py and attentionv2.py with SDPA."""
    import boltz.model.layers.attention as attn_v1
    import boltz.model.layers.attentionv2 as attn_v2

    # --- Patch attention.py (Pairformer / trunk) ---
    def forward_v1_sdpa(self, s: Tensor, z: Tensor, mask: Tensor,
                        multiplicity: int = 1, to_keys=None, model_cache=None) -> Tensor:
        B = s.shape[0]

        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s

        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        if model_cache is None or "z" not in model_cache:
            z = self.proj_z(z)
            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        z = z.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            # Transpose to (B, H, N, D) for SDPA
            q_t = q.bfloat16().transpose(1, 2)
            k_t = k.bfloat16().transpose(1, 2)
            v_t = v.bfloat16().transpose(1, 2)

            # Build attention bias: pair bias + padding mask
            attn_bias = z.bfloat16() + (1 - mask[:, None, None].bfloat16()) * -self.inf

            o = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_bias,
                dropout_p=0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            o = o.transpose(1, 2).to(v.dtype)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o

    attn_v1.AttentionPairBias.forward = forward_v1_sdpa
    print("[sdpa-wrapper] Patched attention.AttentionPairBias -> SDPA bf16")

    # --- Patch attentionv2.py (diffusion transformer / score model) ---
    def forward_v2_sdpa(self, s: Tensor, z: Tensor, mask: Tensor,
                        k_in: Tensor, multiplicity: int = 1) -> Tensor:
        B = s.shape[0]

        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            q_t = q.bfloat16().transpose(1, 2)
            k_t = k.bfloat16().transpose(1, 2)
            v_t = v.bfloat16().transpose(1, 2)

            attn_bias = bias.bfloat16() + (1 - mask[:, None, None].bfloat16()) * -self.inf

            o = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_bias,
                dropout_p=0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            o = o.transpose(1, 2).to(v.dtype)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o

    attn_v2.AttentionPairBias.forward = forward_v2_sdpa
    print("[sdpa-wrapper] Patched attentionv2.AttentionPairBias -> SDPA bf16")


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
                p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight, eps=1e-5,
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
                p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight, eps=1e-5,
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
    print("[sdpa-wrapper] bf16 trunk patch applied (removed .float() upcast)")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--sdpa", action="store_true",
                       help="Enable SDPA attention replacement")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")

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

    # Apply SDPA patch if requested
    if our_args.sdpa:
        patch_attention_sdpa()

    # Apply bf16 trunk patch if requested
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Handle kernels
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[sdpa-wrapper] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[sdpa-wrapper] Kernels DISABLED")
    elif our_args.enable_kernels and kernels_available:
        print("[sdpa-wrapper] Kernels ENABLED")
    else:
        if not kernels_available:
            boltz_args.append("--no_kernels")
            print("[sdpa-wrapper] Kernels DISABLED (not installed)")
        else:
            print("[sdpa-wrapper] Kernels ENABLED (default)")

    print(f"[sdpa-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"sdpa={our_args.sdpa}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
