"""Boltz wrapper with sparse triangular multiplication.

Extends the eval-v4 wrapper to add sparse window-based triangle
multiplication. The sparse approach replaces the k-dimension contraction
in TriangleMult from O(N) to O(W) per (i,j) pair.

Emits phase timestamps for prediction-only timing (eval-v4 compatible):
    [PHASE] predict_start=<timestamp>
    [PHASE] predict_end=<timestamp>

Usage:
    python boltz_wrapper_sparse.py input.yaml --out_dir out \
        --sampling_steps 12 --recycling_steps 0 --gamma_0 0.0 \
        --matmul_precision high --bf16_trunk --sparse_window 128
"""
import sys
import time
import argparse
import torch


def patch_triangular_mult_sparse(window_size: int):
    """Replace TriangleMultiplication with sparse window version."""
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
    )

    W = window_size
    _kidx_cache = {}

    def _get_kidx(N, device):
        key = (N, str(device))
        if key not in _kidx_cache:
            starts = torch.clamp(torch.arange(N, device=device) - W // 2, 0, N - W)
            offsets = torch.arange(W, device=device)
            _kidx_cache[key] = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N - 1)
        return _kidx_cache[key]

    def forward_outgoing_sparse(self, x, mask, use_kernels=False):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        B, Ni, Nk, D = a.shape

        if W >= Nk:
            out = torch.einsum("bikd,bjkd->bijd", a, b)
        else:
            kidx = _get_kidx(Nk, a.device)
            kidx_exp = kidx.unsqueeze(0).unsqueeze(-1).expand(B, Ni, W, D)
            a_sparse = torch.gather(a, 2, kidx_exp)
            b_sparse = torch.gather(b, 2, kidx_exp)
            out = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)

        out = self.p_out(self.norm_out(out)) * self.g_out(x_in).sigmoid()
        return out

    def forward_incoming_sparse(self, x, mask, use_kernels=False):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x, 2, dim=-1)
        B, Nk, Ni, D = a.shape

        if W >= Nk:
            out = torch.einsum("bkid,bkjd->bijd", a, b)
        else:
            kidx = _get_kidx(Nk, a.device)
            kidx_exp = kidx.unsqueeze(0).unsqueeze(-1).expand(B, Ni, W, D)
            a_t = a.transpose(1, 2)
            b_t = b.transpose(1, 2)
            a_sparse = torch.gather(a_t, 2, kidx_exp)
            b_sparse = torch.gather(b_t, 2, kidx_exp)
            out = torch.einsum("biwd,bjwd->bijd", a_sparse, b_sparse)

        out = self.p_out(self.norm_out(out)) * self.g_out(x_in).sigmoid()
        return out

    TriangleMultiplicationOutgoing.forward = forward_outgoing_sparse
    TriangleMultiplicationIncoming.forward = forward_incoming_sparse
    print(f"[wrapper] Sparse tri-mult patch applied (W={W})", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--gamma_0", type=float, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--sparse_window", type=int, default=0,
                       help="Window size for sparse tri-mult (0=disabled)")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    import boltz.main as boltz_main
    from dataclasses import dataclass, field as _field
    from pytorch_lightning import Trainer as _OrigTrainer

    # Monkey-patch diffusion params for ODE mode
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
        print(f"[wrapper] Patched diffusion: gamma_0={_g0}, noise_scale={_ns}",
              file=sys.stderr, flush=True)

    # Apply sparse tri-mult patch (includes bf16 - no .float() upcast)
    if our_args.sparse_window > 0:
        patch_triangular_mult_sparse(our_args.sparse_window)
    elif our_args.bf16_trunk:
        # Standalone bf16 patch (no sparse)
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
        print("[wrapper] bf16 trunk patch applied", file=sys.stderr, flush=True)

    # Phase timing: monkey-patch Trainer.predict to emit timestamps
    _orig_trainer_predict = _OrigTrainer.predict

    def _timed_predict(self, *args, **kwargs):
        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)
        result = _orig_trainer_predict(self, *args, **kwargs)
        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)
        return result

    _OrigTrainer.predict = _timed_predict

    sys.argv = [sys.argv[0]] + boltz_args
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)
    boltz_main.predict()
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
