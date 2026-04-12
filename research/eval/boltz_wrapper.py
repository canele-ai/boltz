"""Thin wrapper around boltz predict that supports matmul_precision and compile flags.

Emits phase timestamps to stderr so the evaluator can compute prediction-only
time (excluding model loading). Format:
    [PHASE] model_load_done=<timestamp>
    [PHASE] predict_done=<timestamp>

Usage:
    python boltz_wrapper.py predict input.yaml --out_dir out --sampling_steps 20 \
        --matmul_precision high --compile_pairformer --compile_structure
"""
import sys
import time
import argparse
import torch

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
        print(f"[wrapper] Patched diffusion: gamma_0={_g0}, noise_scale={_ns}",
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
        print("[wrapper] bf16 trunk patch applied", file=sys.stderr, flush=True)

    # Monkey-patch Trainer.predict to emit a timestamp just before inference starts.
    # This fires AFTER model loading, data processing, and DataModule setup —
    # right before the actual GPU prediction loop.
    _orig_trainer_predict = _OrigTrainer.predict

    def _timed_predict(self, *args, **kwargs):
        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)
        result = _orig_trainer_predict(self, *args, **kwargs)
        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)
        return result

    _OrigTrainer.predict = _timed_predict

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Emit start timestamp
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)

    # Run boltz predict
    boltz_main.predict()

    # Emit wrapper done timestamp
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
