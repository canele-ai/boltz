"""Boltz wrapper with SDPA attention + ODE + TF32 + bf16 trunk.

Uses PyTorch's built-in scaled_dot_product_attention (which dispatches to
FlashAttention2 or Memory-Efficient attention) instead of the manual einsum
implementation. This avoids materializing the full S*S attention matrix.

The pair bias is added via the attn_mask parameter of SDPA.

All optimizations:
1. ODE sampling (gamma_0=0)
2. TF32 matmul precision (matmul_precision="high")
3. bf16 trunk (no .float() upcast in triangular_mult)
4. SDPA fused attention (FlashAttention2 or mem-efficient backend)
"""
import sys
import argparse
import torch
import torch.nn.functional as F


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
    print("[sdpa-wrapper] bf16 trunk patch applied")


def patch_attention_sdpa():
    """Monkey-patch AttentionPairBias to use torch SDPA.

    SDPA supports attn_mask which we use to inject the pair bias.
    This dispatches to FlashAttention2 or Memory-Efficient attention
    depending on the backend availability and tensor properties.
    """
    # Patch v1 attention
    import boltz.model.layers.attention as attn_v1

    def sdpa_forward_v1(self, s, z, mask, multiplicity=1, to_keys=None, model_cache=None):
        B = s.shape[0]

        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s

        # (B, S, H, D) -> need (B, H, S, D) for SDPA
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if model_cache is None or "z" not in model_cache:
            z = self.proj_z(z)
            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        z = z.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        # Build attn_mask: pair_bias + padding mask combined
        # pair_bias z: (B, H, Sq, Sk) - already in the right shape
        # mask: (B, Sk) -> (B, 1, 1, Sk)
        attn_mask = z.float() + (1 - mask[:, None, None].float()) * -self.inf

        with torch.autocast("cuda", enabled=False):
            # SDPA with custom attn_mask (includes pair bias)
            # scale=1/sqrt(D) is applied by SDPA automatically
            o = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            ).to(v.dtype)

        # (B, H, S, D) -> (B, S, H*D)
        o = o.transpose(1, 2).reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o

    attn_v1.AttentionPairBias.forward = sdpa_forward_v1
    print("[sdpa-wrapper] SDPA attention patch applied to v1")

    # Patch v2 attention
    try:
        import boltz.model.layers.attentionv2 as attn_v2

        def sdpa_forward_v2(self, s, z, mask, k_in, multiplicity=1):
            B = s.shape[0]

            q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

            bias = self.proj_z(z)
            bias = bias.repeat_interleave(multiplicity, 0)

            g = self.proj_g(s).sigmoid()

            # Build combined attn_mask
            attn_mask = bias.float() + (1 - mask[:, None, None].float()) * -self.inf

            with torch.autocast("cuda", enabled=False):
                o = F.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(),
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                ).to(v.dtype)

            o = o.transpose(1, 2).reshape(B, -1, self.c_s)
            o = self.proj_o(g * o)
            return o

        attn_v2.AttentionPairBias.forward = sdpa_forward_v2
        print("[sdpa-wrapper] SDPA attention patch applied to v2")
    except Exception as e:
        print(f"[sdpa-wrapper] Warning: could not patch v2 attention: {e}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--sdpa_attention", action="store_true")

    our_args, boltz_args = parser.parse_known_args()

    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Apply SDPA patch BEFORE boltz import
    if our_args.sdpa_attention:
        patch_attention_sdpa()

    import boltz.main as boltz_main
    from dataclasses import dataclass

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

    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[sdpa-wrapper] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
    elif our_args.enable_kernels and kernels_available:
        pass
    elif not kernels_available:
        boltz_args.append("--no_kernels")

    print(f"[sdpa-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"sdpa_attention={our_args.sdpa_attention}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
