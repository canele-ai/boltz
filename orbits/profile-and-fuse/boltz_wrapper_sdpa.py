"""Boltz wrapper with SDPA attention + all stacked optimizations.

Extends the eval-v2-winner stacked wrapper to add:
1. F.scaled_dot_product_attention replacing manual einsum attention
2. ODE sampling (gamma_0=0)
3. TF32 matmul precision (with fix for predict() override bug)
4. bf16 trunk (remove .float() upcast in triangular_mult.py)
5. bf16 outer product mean (remove .float() upcast)

Key fix discovered by profile-and-fuse orbit:
  boltz/main.py:predict() calls torch.set_float32_matmul_precision("highest")
  on line 1096, which OVERRIDES the wrapper's TF32 setting. We monkey-patch
  predict() to skip that reset, ensuring TF32 actually takes effect.
"""
import sys
import argparse
import torch
import torch.nn.functional as F


def patch_attention_sdpa():
    """Replace manual einsum attention with F.scaled_dot_product_attention.

    The original AttentionPairBias.forward does:
        1. torch.autocast("cuda", enabled=False) to force fp32
        2. einsum("bihd,bjhd->bhij", q.float(), k.float()) for Q*K^T
        3. manual scaling, bias addition, masking, softmax
        4. einsum("bhij,bjhd->bihd", attn, v.float()) for attn*V

    This is 4 separate global memory round-trips in float32.

    F.scaled_dot_product_attention fuses all of this into a single kernel,
    and on L40S (Ada Lovelace with SM89), dispatches to FlashAttention-2
    or memory-efficient attention which runs in the input dtype.
    """
    from boltz.model.layers.attentionv2 import AttentionPairBias

    def forward_sdpa(self, s, z, mask, k_in, multiplicity=1):
        B = s.shape[0]

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        # SDPA expects (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build attention mask: combine pair bias with padding mask
        attn_bias = bias.to(q.dtype)
        padding_mask = (1 - mask[:, None, None].to(q.dtype)) * -self.inf
        attn_bias = attn_bias + padding_mask

        # Use SDPA -- dispatches to FlashAttention-2 or mem-efficient attention
        o = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            scale=1.0 / (self.head_dim ** 0.5),
        )

        # (B, H, S, D) -> (B, S, H*D)
        o = o.transpose(1, 2).reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o

    AttentionPairBias.forward = forward_sdpa
    print("[sdpa-wrapper] SDPA attention patch applied")


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


def patch_outer_product_mean_bf16():
    """Remove .float() upcast in outer_product_mean.py.

    The non-chunked path (line 92) casts a and b to float32 for the einsum.
    Under bf16-mixed autocast, this is unnecessary.
    """
    from boltz.model.layers.outer_product_mean import OuterProductMean

    def forward_opm_bf16(self, m, mask, chunk_size=None):
        mask = mask.unsqueeze(-1).to(m)
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask

        if chunk_size is not None and not self.training:
            # Chunked path (unchanged)
            for i in range(0, mask.shape[1], 64):
                if i == 0:
                    num_mask = (
                        mask[:, i:i+64, None, :] * mask[:, i:i+64, :, None]
                    ).sum(1)
                else:
                    num_mask += (
                        mask[:, i:i+64, None, :] * mask[:, i:i+64, :, None]
                    ).sum(1)
            num_mask = num_mask.clamp(min=1)

            for i in range(0, self.c_hidden, chunk_size):
                a_chunk = a[:, :, :, i:i+chunk_size]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, i * self.c_hidden:(i + chunk_size) * self.c_hidden
                ]
                z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
                z = z.reshape(*z.shape[:3], -1)
                z = z / num_mask
                if i == 0:
                    z_out = z.to(m) @ sliced_weight_proj_o.T
                else:
                    z_out = z_out + z.to(m) @ sliced_weight_proj_o.T
            z_out = z_out + self.proj_o.bias
            return z_out
        else:
            mask = mask[:, :, None, :] * mask[:, :, :, None]
            num_mask = mask.sum(1).clamp(min=1)
            # NO .float() upcast -- keep in current dtype (bf16 under autocast)
            z = torch.einsum("bsic,bsjd->bijcd", a, b)
            z = z.reshape(*z.shape[:3], -1)
            z = z / num_mask
            z = self.proj_o(z.to(m))
            return z

    OuterProductMean.forward = forward_opm_bf16
    print("[sdpa-wrapper] bf16 outer product mean patch applied")


def patch_compile_score_model():
    """Compile the diffusion score model for inference speedup.

    The score model runs 20 times per prediction (once per diffusion step),
    so compilation overhead amortizes. We monkey-patch Boltz2's predict_step
    to compile the score model on first invocation.

    torch.compile with mode='reduce-overhead' uses CUDA graphs which
    eliminate kernel launch overhead for the repeated diffusion loop.
    """
    from boltz.model.models.boltz2 import Boltz2

    _original_predict_step = Boltz2.predict_step
    _compiled = [False]

    def predict_step_with_compile(self, batch, batch_idx, dataloader_idx=0):
        if not _compiled[0]:
            # Compile the score model inside AtomDiffusion
            if hasattr(self, 'structure_module') and hasattr(self.structure_module, 'score_model'):
                print("[sdpa-wrapper] Compiling score model with torch.compile...")
                self.structure_module.score_model = torch.compile(
                    self.structure_module.score_model,
                    dynamic=False,
                    fullgraph=False,
                )
                print("[sdpa-wrapper] Score model compiled")
            _compiled[0] = True
        return _original_predict_step(self, batch, batch_idx, dataloader_idx)

    Boltz2.predict_step = predict_step_with_compile
    print("[sdpa-wrapper] Score model compile patch registered")


def patch_predict_matmul_precision(target_precision):
    """Prevent boltz.main.predict() from resetting matmul precision.

    boltz/main.py line 1096 calls torch.set_float32_matmul_precision("highest"),
    which overrides any TF32 setting from the wrapper. We monkey-patch
    torch.set_float32_matmul_precision to ignore calls to "highest" when
    we want a different precision.
    """
    _original_set_precision = torch.set_float32_matmul_precision

    def guarded_set_precision(precision):
        if precision == "highest" and target_precision != "highest":
            print(f"[sdpa-wrapper] Blocked matmul precision reset to 'highest', "
                  f"keeping '{target_precision}'")
            return
        return _original_set_precision(precision)

    torch.set_float32_matmul_precision = guarded_set_precision


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
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--bf16_opm", action="store_true",
                       help="Remove .float() upcast in outer_product_mean.py")
    parser.add_argument("--sdpa_attention", action="store_true",
                       help="Replace manual einsum attention with F.scaled_dot_product_attention")
    parser.add_argument("--compile_score", action="store_true",
                       help="Compile the diffusion score model with torch.compile")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Prevent predict() from resetting matmul precision to "highest"
    if our_args.matmul_precision != "highest":
        patch_predict_matmul_precision(our_args.matmul_precision)

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

    # Apply bf16 outer product mean patch
    if our_args.bf16_opm:
        patch_outer_product_mean_bf16()

    # Apply SDPA attention patch
    if our_args.sdpa_attention:
        patch_attention_sdpa()

    # Apply score model compilation
    if our_args.compile_score:
        patch_compile_score_model()

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[sdpa-wrapper] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False
        print("[sdpa-wrapper] cuequivariance_torch NOT available")

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
          f"bf16_opm={our_args.bf16_opm}, "
          f"sdpa_attention={our_args.sdpa_attention}, "
          f"compile_score={our_args.compile_score}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
