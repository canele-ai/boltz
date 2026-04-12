"""Boltz wrapper with head pruning + stacked optimizations (ODE + TF32 + bf16).

Extends the stacked wrapper to support zeroing out attention heads in the
Pairformer's sequence attention (AttentionPairBias).

Usage:
    python boltz_wrapper_headprune.py input.yaml --out_dir out \
        --sampling_steps 20 --recycling_steps 0 \
        --gamma_0 0.0 --matmul_precision high --bf16_trunk \
        --enable_kernels --prune_fraction 0.25
"""
import sys
import time
import argparse
import json
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
    print("[headprune-wrapper] bf16 trunk patch applied", file=sys.stderr, flush=True)


def measure_head_importance_and_rank(model):
    """Measure head importance and return sorted list (least important first).

    Returns list of (block_idx, head_idx, importance_score).
    """
    pairformer = model.pairformer_module
    all_heads = []
    for block_idx, layer in enumerate(pairformer.layers):
        attn = layer.attention
        num_heads = attn.num_heads
        head_dim = attn.head_dim

        with torch.no_grad():
            w_o = attn.proj_o.weight.float()
            w_v = attn.proj_v.weight.float()
            w_g = attn.proj_g.weight.float()

            for h in range(num_heads):
                start = h * head_dim
                end = (h + 1) * head_dim
                o_norm = w_o[:, start:end].norm().item()
                v_norm = w_v[start:end, :].norm().item()
                g_norm = w_g[start:end, :].norm().item()
                head_imp = o_norm * v_norm * g_norm
                all_heads.append((block_idx, h, head_imp))

    all_heads.sort(key=lambda x: x[2])
    return all_heads


def prune_heads_by_zeroing(model, heads_to_prune):
    """Zero out weights for specified (block_idx, head_idx) pairs."""
    pairformer = model.pairformer_module

    for block_idx, head_idx in heads_to_prune:
        layer = pairformer.layers[block_idx]
        attn = layer.attention
        head_dim = attn.head_dim

        with torch.no_grad():
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim

            attn.proj_o.weight[:, start:end].zero_()
            attn.proj_q.weight[start:end, :].zero_()
            if attn.proj_q.bias is not None:
                attn.proj_q.bias[start:end].zero_()
            attn.proj_k.weight[start:end, :].zero_()
            attn.proj_v.weight[start:end, :].zero_()
            attn.proj_g.weight[start:end, :].zero_()

            # Zero pair bias for this head
            # proj_z: Sequential(LayerNorm, Linear(c_z, num_heads), Rearrange)
            proj_z_linear = attn.proj_z[1]
            proj_z_linear.weight[head_idx, :].zero_()


def apply_head_pruning_hook(prune_fraction):
    """Monkey-patch Boltz2 to apply head pruning after model loading.

    This hooks into the model's predict_step or forward to apply pruning
    on the first call, after the model has been loaded to GPU.
    """
    from boltz.model.models.boltz2 import Boltz2

    _original_predict_step = Boltz2.predict_step
    _pruned = [False]

    def patched_predict_step(self, batch, batch_idx):
        if not _pruned[0]:
            _pruned[0] = True
            print(f"[headprune-wrapper] Applying head pruning: {prune_fraction*100:.0f}%",
                  file=sys.stderr, flush=True)

            ranked = measure_head_importance_and_rank(self)
            n_prune = int(len(ranked) * prune_fraction)
            heads_to_prune = [(b, h) for b, h, _ in ranked[:n_prune]]
            prune_heads_by_zeroing(self, heads_to_prune)

            print(f"[headprune-wrapper] Pruned {n_prune}/{len(ranked)} heads",
                  file=sys.stderr, flush=True)

        return _original_predict_step(self, batch, batch_idx)

    Boltz2.predict_step = patched_predict_step


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--prune_fraction", type=float, default=0.0,
                       help="Fraction of heads to prune (0.0-1.0)")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

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

    # Apply bf16 trunk
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Apply head pruning hook (activates on first predict_step)
    if our_args.prune_fraction > 0:
        apply_head_pruning_hook(our_args.prune_fraction)
        print(f"[headprune-wrapper] Head pruning hook installed: {our_args.prune_fraction*100:.0f}%",
              file=sys.stderr, flush=True)

    # Handle kernels
    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
    elif not our_args.enable_kernels:
        try:
            import cuequivariance_torch
        except ImportError:
            boltz_args.append("--no_kernels")

    # Emit phase timestamps
    from pytorch_lightning import Trainer as _OrigTrainer
    _orig_predict = _OrigTrainer.predict

    def _timed_predict(self, *args, **kwargs):
        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)
        result = _orig_predict(self, *args, **kwargs)
        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)
        return result

    _OrigTrainer.predict = _timed_predict

    print(f"[headprune-wrapper] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"prune_fraction={our_args.prune_fraction}",
          file=sys.stderr, flush=True)

    sys.argv = [sys.argv[0]] + boltz_args
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)
    boltz_main.predict()
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
