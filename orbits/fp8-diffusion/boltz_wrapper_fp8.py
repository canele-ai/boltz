"""Boltz wrapper: ODE + TF32 + bf16 trunk + bf16 attention.

Extends eval-v2-winner's stacked wrapper with:
1. ODE sampling (gamma_0=0) -- deterministic first-order Euler solver
2. TF32 matmul precision (matmul_precision="high")
3. bf16 trunk -- removes .float() upcast in triangular_mult.py
4. bf16 attention -- removes .float() upcast and autocast(enabled=False)
   in both attention.py and attentionv2.py (score model + trunk)

The hypothesis: AttentionPairBias explicitly casts Q,K,V,Z to float32
and disables autocast. This is called 480 times per prediction (24 layers
x 20 steps). On L40S, bf16 tensor cores deliver ~2x the FLOPS of fp32.
Removing the upcast lets attention run in bf16, halving the compute.
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
    print("[fp8-wrapper] bf16 trunk patch applied")


def patch_attention_bf16():
    """Remove .float() upcast in AttentionPairBias for bf16 attention.

    Patches both attention.py (used by trunk/pairformer) and
    attentionv2.py (used by score model transformer).

    The key change: remove `torch.autocast("cuda", enabled=False)` and
    all `.float()` casts. Let the attention run in whatever precision the
    model is in (bf16 under autocast). Softmax in bf16 is numerically
    stable for typical sequence lengths (<2048 tokens).
    """
    # Patch attentionv2.py (score model -- this is the hot path)
    from boltz.model.layers.attentionv2 import (
        AttentionPairBias as AttentionPairBiasV2,
    )

    def forward_v2_bf16(self, s, z, mask, k_in, multiplicity=1):
        B = s.shape[0]
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        # NO autocast(enabled=False), NO .float() casts
        attn = torch.einsum("bihd,bjhd->bhij", q, k)
        attn = attn / (self.head_dim**0.5) + bias
        attn = attn + (1 - mask[:, None, None].to(attn.dtype)) * -self.inf
        attn = attn.softmax(dim=-1)
        o = torch.einsum("bhij,bjhd->bihd", attn, v)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o

    AttentionPairBiasV2.forward = forward_v2_bf16
    print("[fp8-wrapper] bf16 attention patch applied (attentionv2)")

    # Patch attention.py (trunk pairformer sequence attention)
    from boltz.model.layers.attention import (
        AttentionPairBias as AttentionPairBiasV1,
    )

    def forward_v1_bf16(self, s, z, mask, multiplicity=1, to_keys=None, model_cache=None):
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

        # NO autocast(enabled=False), NO .float() casts
        attn = torch.einsum("bihd,bjhd->bhij", q, k)
        attn = attn / (self.head_dim**0.5) + z
        attn = attn + (1 - mask[:, None, None].to(attn.dtype)) * -self.inf
        attn = attn.softmax(dim=-1)
        o = torch.einsum("bhij,bjhd->bihd", attn, v)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o

    AttentionPairBiasV1.forward = forward_v1_bf16
    print("[fp8-wrapper] bf16 attention patch applied (attention v1)")


def patch_pairformer_bf16():
    """Remove .float() upcast in pairformer sequence stack.

    The pairformer has its own autocast(enabled=False) block around the
    sequence attention + transition. This patch removes it.
    """
    from boltz.model.layers import pairformer

    def pairformer_block_forward_bf16(
        self, s, z, mask, pair_mask,
        chunk_size_tri_attn=None,
        use_kernels=False,
        use_cuequiv_attn=False,
    ):
        # Pair stack (unchanged)
        dropout = self.training

        z = z + self.tri_mul_out(z, mask=pair_mask, use_kernels=use_kernels)
        z = z + self.tri_mul_in(z, mask=pair_mask, use_kernels=use_kernels)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )
        z = z + self.transition_z(z)

        # Sequence stack -- NO autocast(enabled=False), NO .float() casts
        s_normed = self.pre_norm_s(s)
        s = s + self.attention(
            s=s_normed, z=z, mask=mask, k_in=s_normed
        )
        s = s + self.transition_s(s)
        s = self.s_post_norm(s)

        return s, z

    pairformer.PairformerLayer.forward = pairformer_block_forward_bf16
    print("[fp8-wrapper] bf16 pairformer sequence stack patch applied")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true",
                       help="Remove .float() upcast in triangular_mult")
    parser.add_argument("--bf16_attention", action="store_true",
                       help="Remove .float() upcast in AttentionPairBias")
    parser.add_argument("--bf16_pairformer", action="store_true",
                       help="Remove .float() upcast in pairformer seq stack")
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

    # Apply patches
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    if our_args.bf16_attention:
        patch_attention_bf16()

    if our_args.bf16_pairformer:
        patch_pairformer_bf16()

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[fp8-wrapper] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False
        print("[fp8-wrapper] cuequivariance_torch NOT available")

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[fp8-wrapper] Kernels DISABLED")
    elif our_args.enable_kernels and kernels_available:
        print("[fp8-wrapper] Kernels ENABLED")
    else:
        if not kernels_available:
            boltz_args.append("--no_kernels")
            print("[fp8-wrapper] Kernels DISABLED (not installed)")
        else:
            print("[fp8-wrapper] Kernels ENABLED (default)")

    print(f"[fp8-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"bf16_attention={our_args.bf16_attention}, "
          f"bf16_pairformer={our_args.bf16_pairformer}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
