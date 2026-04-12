"""Boltz wrapper with BF16 trunk and ODE sampler support.

Extends the ODE sampler wrapper to also monkey-patch the triangular multiply
layers so they stay in bf16 instead of upcasting to fp32 before the einsum.

The Boltz codebase does `x.float()` (line 116 in triangular_mult.py for Outgoing,
line 204 for Incoming) before the einsum contraction. On L40S with bf16, the
isolated triangular multiply einsum is 1.94x faster than fp32 (orbit/l40s-kernels).
This wrapper removes that upcast.

Additionally patches:
- ODE sampler (gamma_0=0, deterministic sampling)
- Optionally: outer_product_mean bf16 (also has .float() before einsum)
- Optionally: pairformer sequence attention bf16

Usage:
    python boltz_wrapper_bf16.py input.yaml --out_dir out --sampling_steps 20 \
        --recycling_steps 0 --gamma_0 0.0 --bf16_trunk --matmul_precision highest
"""
import sys
import argparse
import torch


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
                       help="Remove fp32 upcast in triangular multiply (keep bf16)")
    parser.add_argument("--bf16_opm", action="store_true",
                       help="Remove fp32 upcast in outer product mean")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz and apply patches
    import boltz.main as boltz_main
    from dataclasses import dataclass

    # --- ODE sampler patch (from orbit/ode-sampler) ---
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

    # --- BF16 trunk patch: triangular multiply ---
    if our_args.bf16_trunk:
        from boltz.model.layers import triangular_mult as tri_mod

        # Save original forwards
        _orig_outgoing_forward = tri_mod.TriangleMultiplicationOutgoing.forward
        _orig_incoming_forward = tri_mod.TriangleMultiplicationIncoming.forward

        def _bf16_outgoing_forward(self, x, mask, use_kernels=False):
            """Patched forward: skip .float() upcast, keep bf16."""
            if use_kernels:
                return tri_mod.kernel_triangular_mult(
                    x, direction="outgoing", mask=mask,
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
            # Input gating
            x = self.norm_in(x)
            x_in = x
            x = self.p_in(x) * self.g_in(x).sigmoid()
            # Apply mask
            x = x * mask.unsqueeze(-1)
            # Split input -- NO .float() upcast (the key change)
            a, b = torch.chunk(x, 2, dim=-1)
            # Triangular projection (stays in bf16)
            x = torch.einsum("bikd,bjkd->bijd", a, b)
            # Output gating
            x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
            return x

        def _bf16_incoming_forward(self, x, mask, use_kernels=False):
            """Patched forward: skip .float() upcast, keep bf16."""
            if use_kernels:
                return tri_mod.kernel_triangular_mult(
                    x, direction="incoming", mask=mask,
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
            # Input gating
            x = self.norm_in(x)
            x_in = x
            x = self.p_in(x) * self.g_in(x).sigmoid()
            # Apply mask
            x = x * mask.unsqueeze(-1)
            # Split input -- NO .float() upcast (the key change)
            a, b = torch.chunk(x, 2, dim=-1)
            # Triangular projection (stays in bf16)
            x = torch.einsum("bkid,bkjd->bijd", a, b)
            # Output gating
            x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
            return x

        tri_mod.TriangleMultiplicationOutgoing.forward = _bf16_outgoing_forward
        tri_mod.TriangleMultiplicationIncoming.forward = _bf16_incoming_forward
        print("[bf16-wrapper] Patched triangular multiply to stay in bf16")

    # --- BF16 outer product mean patch ---
    if our_args.bf16_opm:
        from boltz.model.layers import outer_product_mean as opm_mod

        _orig_opm_forward = opm_mod.OuterProductMean.forward

        def _bf16_opm_forward(self, m, mask, chunk_size=None):
            """Patched forward: skip .float() in outer product einsum."""
            # If chunk_size is set, use the original chunked path (no .float() there)
            if chunk_size is not None and not self.training:
                # The chunked path doesn't have .float(), delegate to original
                return _orig_opm_forward(self, m, mask, chunk_size)

            # Expand mask
            mask = mask.unsqueeze(-1).to(m)

            # Keep the original logic but remove .float() on a and b
            m = self.norm(m)
            a = self.proj_a(m) * mask
            b = self.proj_b(m) * mask

            mask_2d = mask[:, :, None, :] * mask[:, :, :, None]
            num_mask = mask_2d.sum(1).clamp(min=1)

            # NO .float() upcast on a and b
            z = torch.einsum("bsic,bsjd->bijcd", a, b)
            z = z.reshape(*z.shape[:3], -1)
            z = z / num_mask

            z = self.proj_o(z.to(m))
            return z

        opm_mod.OuterProductMean.forward = _bf16_opm_forward
        print("[bf16-wrapper] Patched outer product mean to stay in bf16")

    flags = []
    if our_args.bf16_trunk:
        flags.append("bf16_tri_mult")
    if our_args.bf16_opm:
        flags.append("bf16_opm")
    print(f"[bf16-wrapper] gamma_0={our_args.gamma_0}, noise_scale={our_args.noise_scale}, patches={flags}")

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()


if __name__ == "__main__":
    main()
