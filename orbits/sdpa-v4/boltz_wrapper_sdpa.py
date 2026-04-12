"""Boltz wrapper extending eval-v4's boltz_wrapper.py with SDPA attention.

Adds --sdpa flag to replace AttentionPairBias's manual einsum with
torch.nn.functional.scaled_dot_product_attention. All other features
(gamma_0, noise_scale, bf16_trunk, phase timestamps) are inherited from
the base wrapper.

Usage:
    python boltz_wrapper_sdpa.py input.yaml --out_dir out --sampling_steps 12 \
        --recycling_steps 0 --gamma_0 0.0 --matmul_precision high --bf16_trunk --sdpa
"""
import sys
import time
import argparse
import torch


def patch_attention_sdpa():
    """Replace manual einsum attention with torch SDPA in AttentionPairBias.

    The original forward() does:
        attn = einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = attn / sqrt(head_dim) + z.float()
        attn = attn + (1 - mask) * -inf
        attn = softmax(attn)
        o = einsum("bhij,bjhd->bihd", attn, v.float())

    SDPA fuses scale, bias addition, softmax, and V multiplication into a
    single optimized kernel (FlashAttention-2 or memory-efficient backend).
    We pass the pair bias z + mask bias as the attn_mask argument.
    """
    import torch.nn.functional as F
    from boltz.model.layers.attention import AttentionPairBias

    def forward_sdpa(
        self,
        s,
        z,
        mask,
        multiplicity=1,
        to_keys=None,
        model_cache=None,
    ):
        B = s.shape[0]

        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s

        # Compute projections: (B, S, H, D)
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        # Pair bias z: (B, H, N, N)
        if model_cache is None or "z" not in model_cache:
            z = self.proj_z(z)
            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        z = z.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        # Build combined attention bias: pair bias + mask
        with torch.autocast("cuda", enabled=False):
            attn_bias = z.float() + (1 - mask[:, None, None].float()) * -self.inf

            # SDPA expects (B, H, S, D) layout
            q_sdpa = q.float().permute(0, 2, 1, 3)  # (B, H, S_q, D)
            k_sdpa = k.float().permute(0, 2, 1, 3)  # (B, H, S_k, D)
            v_sdpa = v.float().permute(0, 2, 1, 3)  # (B, H, S_k, D)

            # torch SDPA: fused scale + softmax + V multiplication
            o = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_bias,
                dropout_p=0.0,
                scale=1.0 / (self.head_dim ** 0.5),
            )
            # o is (B, H, S_q, D) -> transpose back to (B, S_q, H, D)
            o = o.permute(0, 2, 1, 3).to(v.dtype)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o

    AttentionPairBias.forward = forward_sdpa
    print("[sdpa-wrapper] SDPA attention patch applied to AttentionPairBias",
          file=sys.stderr, flush=True)


def main():
    # Extract our custom flags before passing the rest to boltz
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--gamma_0", type=float, default=None,
                       help="Diffusion gamma_0 (0.0 = ODE mode)")
    parser.add_argument("--noise_scale", type=float, default=None,
                       help="Diffusion noise scale")
    parser.add_argument("--bf16_trunk", action="store_true",
                       help="Remove .float() upcast in triangular_mult for bf16")
    parser.add_argument("--sdpa", action="store_true",
                       help="Replace AttentionPairBias with SDPA kernel")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    import boltz.main as boltz_main
    from dataclasses import dataclass, field as _field
    from pytorch_lightning import Trainer as _OrigTrainer

    # Monkey-patch diffusion params if gamma_0 or noise_scale specified
    if our_args.gamma_0 is not None or our_args.noise_scale is not None:
        _g0 = our_args.gamma_0 if our_args.gamma_0 is not None else 0.8
        _ns = our_args.noise_scale if our_args.noise_scale is not None else 1.003

        @dataclass
        class PatchedBoltz2DiffusionParams:
            gamma_0: float = _field(default_factory=lambda: _g0)
            gamma_min: float = 1.0
            noise_scale: float = _field(default_factory=lambda: _ns)
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
        print(f"[sdpa-wrapper] Patched diffusion: gamma_0={_g0}, noise_scale={_ns}",
              file=sys.stderr, flush=True)

    # bf16 trunk patch
    if our_args.bf16_trunk:
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
        print("[sdpa-wrapper] bf16 trunk patch applied", file=sys.stderr, flush=True)

    # Apply SDPA attention patch if requested
    if our_args.sdpa:
        patch_attention_sdpa()

    # Monkey-patch Trainer.predict to emit phase timestamps (eval-v4)
    _orig_trainer_predict = _OrigTrainer.predict

    def _timed_predict(self, *args, **kwargs):
        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)
        result = _orig_trainer_predict(self, *args, **kwargs)
        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)
        return result

    _OrigTrainer.predict = _timed_predict

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)
    boltz_main.predict()
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
