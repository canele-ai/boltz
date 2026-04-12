"""Boltz wrapper for bmm-nokernels eval-v4 measurement.

Combines:
1. ODE sampling (gamma_0=0.0) via diffusion param monkey-patch
2. TF32 matmul precision
3. bf16 trunk (no .float() upcast in triangle multiplication)
4. Batched matmul replacement for triangle multiplication einsum
5. use_kernels=False (disables cuequivariance)
6. Phase timestamps for predict_only_s (eval-v4 compatible)

The key hypothesis: bf16 bmm avoids the float32 upcast that cuequivariance
kernels require, giving ~20% speedup on medium/large complexes.

Usage:
    python boltz_wrapper_bmm.py input.yaml --out_dir out --sampling_steps 12 \
        --matmul_precision high --bf16_trunk --use_bmm --no_kernels \
        --gamma_0 0.0
"""
import sys
import os
import time
import argparse
import torch

# Ensure the directory containing this file is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch_triangle_mul_bmm():
    """Replace TriangleMultiplication einsum with batched matmul.

    Monkey-patches both TriangleMultiplicationOutgoing and
    TriangleMultiplicationIncoming to use torch.bmm instead of
    torch.einsum or cuequivariance kernels.

    The patched forward methods:
    - Skip cuequivariance entirely (ignore use_kernels flag)
    - Use batched matmul for the contraction (bf16-native, no upcast)
    - Keep bf16 throughout (no .float() upcast)
    """
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    def forward_outgoing_bmm(self, x, mask, use_kernels=False):
        # ALWAYS use bmm, ignoring use_kernels flag
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        # bmm path: "bikd,bjkd->bijd"
        B, N, K, D = a.shape
        a_t = a.permute(0, 3, 1, 2).reshape(B * D, N, K)
        b_t = b.permute(0, 3, 1, 2).reshape(B * D, N, K)
        c = torch.bmm(a_t, b_t.transpose(1, 2))
        x = c.reshape(B, D, N, N).permute(0, 2, 3, 1)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    def forward_incoming_bmm(self, x, mask, use_kernels=False):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        # bmm path: "bkid,bkjd->bijd"
        B, K, N, D = a.shape
        a_t = a.permute(0, 3, 1, 2).reshape(B * D, K, N)
        b_t = b.permute(0, 3, 1, 2).reshape(B * D, K, N)
        c = torch.bmm(a_t.transpose(1, 2), b_t)
        x = c.reshape(B, D, N, N).permute(0, 2, 3, 1)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x

    TriangleMultiplicationOutgoing.forward = forward_outgoing_bmm
    TriangleMultiplicationIncoming.forward = forward_incoming_bmm
    print("[bmm-wrapper] Triangle multiplication patched with bmm+bf16",
          file=sys.stderr, flush=True)


def patch_triangle_mul_bf16_einsum():
    """bf16 trunk patch using original einsum (no bmm, no cuequivariance).

    Used for the control config with cuequivariance kernels enabled.
    """
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
    print("[bmm-wrapper] bf16 trunk patch applied (einsum, cuequivariance passthrough)",
          file=sys.stderr, flush=True)


def main():
    # Extract our custom flags before passing the rest to boltz
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=None,
                       help="Diffusion gamma_0 (0.0 = ODE mode)")
    parser.add_argument("--noise_scale", type=float, default=None,
                       help="Diffusion noise scale")
    parser.add_argument("--bf16_trunk", action="store_true",
                       help="Remove .float() upcast in triangular_mult for bf16")
    parser.add_argument("--no_kernels", action="store_true",
                       help="Disable cuequivariance kernels (use_kernels=False)")
    parser.add_argument("--use_bmm", action="store_true",
                       help="Replace triangle mul einsum with batched matmul")

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
        print(f"[bmm-wrapper] Patched diffusion: gamma_0={_g0}, noise_scale={_ns}",
              file=sys.stderr, flush=True)

    # Apply triangle multiplication patches
    if our_args.use_bmm:
        # bmm replaces einsum AND cuequivariance — always force no_kernels
        patch_triangle_mul_bmm()
        our_args.no_kernels = True
    elif our_args.bf16_trunk:
        # bf16 trunk without bmm — use einsum, allow cuequivariance passthrough
        patch_triangle_mul_bf16_einsum()

    # Handle kernel flags
    if our_args.no_kernels:
        # Append boltz's --no_kernels flag (not our flag)
        # Check if it's not already there
        if "--no_kernels" not in boltz_args:
            boltz_args.append("--no_kernels")
        print("[bmm-wrapper] cuequivariance kernels DISABLED",
              file=sys.stderr, flush=True)
    else:
        print("[bmm-wrapper] cuequivariance kernels ENABLED",
              file=sys.stderr, flush=True)

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

    print(f"[bmm-wrapper] matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"use_bmm={our_args.use_bmm}, "
          f"no_kernels={our_args.no_kernels}",
          file=sys.stderr, flush=True)

    # Emit start timestamp
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)

    # Run boltz predict
    boltz_main.predict()

    # Emit wrapper done timestamp
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
