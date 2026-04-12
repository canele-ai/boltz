"""Boltz wrapper with Triton fused attention + ODE + TF32 + bf16 trunk.

Extends the eval-v2-winner stacked wrapper to also monkey-patch
AttentionPairBias.forward with a custom Triton kernel that fuses:
  Q@K^T + scaling + pair_bias + masking + softmax + @V
into a single tiled kernel, avoiding the full S*S attention matrix.

All optimizations:
1. ODE sampling (gamma_0=0) - deterministic first-order Euler solver
2. TF32 matmul precision (matmul_precision="high")
3. bf16 trunk - removes .float() upcast in triangular_mult.py
4. Triton fused attention with pair bias

Usage:
    python boltz_wrapper_triton.py input.yaml --out_dir out --sampling_steps 20 \
        --matmul_precision high --gamma_0 0.0 --bf16_trunk --triton_attention
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
    print("[triton-wrapper] bf16 trunk patch applied")


def patch_attention_triton():
    """Monkey-patch AttentionPairBias to use Triton fused kernel.

    Patches both v1 (attention.py) and v2 (attentionv2.py) attention modules.
    """
    from triton_attention import triton_attention_pair_bias

    # Patch v1 attention (used by DiffusionTransformer)
    import boltz.model.layers.attention as attn_v1

    original_forward_v1 = attn_v1.AttentionPairBias.forward

    def triton_forward_v1(self, s, z, mask, multiplicity=1, to_keys=None, model_cache=None):
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

        # Use Triton fused attention
        o = triton_attention_pair_bias(
            q.float(), k.float(), v.float(),
            z.float(), mask.float(),
            inf=self.inf,
        ).to(v.dtype)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o

    attn_v1.AttentionPairBias.forward = triton_forward_v1
    print("[triton-wrapper] Triton attention patch applied to v1")

    # Patch v2 attention (used by Boltz2)
    try:
        import boltz.model.layers.attentionv2 as attn_v2

        def triton_forward_v2(self, s, z, mask, k_in, multiplicity=1):
            B = s.shape[0]

            q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
            k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
            v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

            bias = self.proj_z(z)
            bias = bias.repeat_interleave(multiplicity, 0)

            g = self.proj_g(s).sigmoid()

            # mask shape: (B, Sk) where Sk = k_in seq length
            # For v2, mask is already the right shape for key positions
            mask_float = mask.float() if mask.dtype != torch.float32 else mask

            o = triton_attention_pair_bias(
                q.float(), k.float(), v.float(),
                bias.float(), mask_float,
                inf=self.inf,
            ).to(v.dtype)

            o = o.reshape(B, -1, self.c_s)
            o = self.proj_o(g * o)
            return o

        attn_v2.AttentionPairBias.forward = triton_forward_v2
        print("[triton-wrapper] Triton attention patch applied to v2")
    except Exception as e:
        print(f"[triton-wrapper] Warning: could not patch v2 attention: {e}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true",
                       help="Remove .float() upcast in triangular_mult for bf16")
    parser.add_argument("--enable_kernels", action="store_true",
                       help="Enable cuequivariance CUDA kernels")
    parser.add_argument("--no_kernels_flag", action="store_true",
                       help="Explicitly disable kernels")
    parser.add_argument("--triton_attention", action="store_true",
                       help="Use Triton fused attention kernel")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Apply Triton attention patch before boltz import if requested
    if our_args.triton_attention:
        # Import triton_attention module (should be on PYTHONPATH or in /eval/)
        patch_attention_triton()

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

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[triton-wrapper] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False
        print("[triton-wrapper] cuequivariance_torch NOT available")

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[triton-wrapper] Kernels DISABLED")
    elif our_args.enable_kernels and kernels_available:
        print("[triton-wrapper] Kernels ENABLED")
    else:
        if not kernels_available:
            boltz_args.append("--no_kernels")
            print("[triton-wrapper] Kernels DISABLED (not installed)")
        else:
            print("[triton-wrapper] Kernels ENABLED (default)")

    print(f"[triton-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"triton_attention={our_args.triton_attention}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
