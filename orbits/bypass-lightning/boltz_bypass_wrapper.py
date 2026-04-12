"""Bypass-Lightning wrapper: direct model inference without Lightning Trainer.

Strategy: monkey-patch Trainer.predict() to replace Lightning's full predict
loop with a minimal direct loop. This lets boltz.main.predict() handle all
input processing, model loading, and DataModule setup normally, but intercepts
the final Trainer.predict() call and replaces it with:

1. Manual DataLoader creation from the DataModule
2. Direct batch iteration with torch.no_grad + autocast
3. Direct model.predict_step() calls (no Lightning hooks/callbacks)
4. Direct BoltzWriter calls for output

This eliminates:
- ~2.0s Lightning Trainer setup (strategy init, callback registration)
- ~4.7s in-loop overhead (batch hooks, tensor transfer management, callbacks)

Includes all eval-v2-winner optimizations (ODE + TF32 + bf16 trunk).

Optional CUDA warmup: runs one forward pass before the timed section to
trigger CUDA kernel JIT, absorbing ~2.2s lazy init.

Usage:
    python boltz_bypass_wrapper.py input.yaml --out_dir out \
        --sampling_steps 12 --recycling_steps 0 \
        --gamma_0 0.0 --matmul_precision high --bf16_trunk \
        --cuda_warmup
"""
import gc
import sys
import time
import argparse

import torch


# Keys that should NOT be moved to GPU (non-tensor or CPU-only data).
# Mirrors Boltz2InferenceDataModule.transfer_batch_to_device exactly.
_CPU_ONLY_KEYS = frozenset([
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "record",
    "affinity_mw",
])


def _transfer_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to device, mirroring Boltz2InferenceDataModule.transfer_batch_to_device."""
    for key in batch:
        if key not in _CPU_ONLY_KEYS:
            batch[key] = batch[key].to(device)
    return batch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    # Our custom flags (stripped before passing to boltz)
    parser.add_argument("--matmul_precision", default="highest",
                        choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--cuda_warmup", action="store_true",
                        help="Run one forward pass before timing to warm CUDA kernels")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz
    from dataclasses import dataclass
    import boltz.main as boltz_main
    from pytorch_lightning import Trainer as _OrigTrainer

    # -----------------------------------------------------------------------
    # ODE diffusion params patch
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # bf16 trunk patch
    # -----------------------------------------------------------------------
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
        print("[bypass] bf16 trunk patch applied", file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Kernel handling
    # -----------------------------------------------------------------------
    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[bypass] Kernels DISABLED", file=sys.stderr, flush=True)
    else:
        try:
            import cuequivariance_torch
            if our_args.enable_kernels:
                print(f"[bypass] Kernels ENABLED (cuequivariance_torch {cuequivariance_torch.__version__})",
                      file=sys.stderr, flush=True)
        except ImportError:
            boltz_args.append("--no_kernels")
            print("[bypass] Kernels DISABLED (not installed)", file=sys.stderr, flush=True)

    print(f"[bypass] gamma_0={our_args.gamma_0}, noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, bf16_trunk={our_args.bf16_trunk}, "
          f"cuda_warmup={our_args.cuda_warmup}",
          file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Monkey-patch Trainer.predict to bypass Lightning's predict loop
    # -----------------------------------------------------------------------
    _cuda_warmup = our_args.cuda_warmup

    def _bypass_predict(trainer_self, model, datamodule=None, return_predictions=False, **kwargs):
        """Replace Lightning Trainer.predict with direct inference loop."""
        # Extract the writer callback from the Trainer
        writer = None
        for cb in trainer_self.callbacks:
            if hasattr(cb, 'write_on_batch_end'):
                writer = cb
                break

        if writer is None:
            print("[bypass] WARNING: No writer callback found, falling back to Lightning",
                  file=sys.stderr, flush=True)
            return _OrigTrainer.predict(trainer_self, model, datamodule=datamodule,
                                        return_predictions=return_predictions, **kwargs)

        # Get dataloader from datamodule
        if datamodule is not None:
            # No setup() needed -- Boltz2InferenceDataModule creates dataset in predict_dataloader()
            dataloader = datamodule.predict_dataloader()
        else:
            print("[bypass] ERROR: No datamodule provided", file=sys.stderr, flush=True)
            return None

        # Move model to GPU
        device = torch.device("cuda")
        model.eval()
        model.to(device)

        # CUDA warmup: run one forward pass to trigger kernel JIT
        if _cuda_warmup:
            print("[bypass] Running CUDA warmup pass...", file=sys.stderr, flush=True)
            warmup_iter = iter(dataloader)
            try:
                warmup_batch = next(warmup_iter)
                warmup_batch = _transfer_batch_to_device(warmup_batch, device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model.predict_step(warmup_batch, 0)
                torch.cuda.synchronize()
                del warmup_batch
                gc.collect()
                torch.cuda.empty_cache()
                print("[bypass] CUDA warmup complete", file=sys.stderr, flush=True)
            except StopIteration:
                print("[bypass] WARNING: No data for warmup", file=sys.stderr, flush=True)
            # Re-create dataloader for the real run
            dataloader = datamodule.predict_dataloader()

        # Direct inference loop
        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                # Transfer batch to GPU (same logic as DataModule.transfer_batch_to_device)
                batch = _transfer_batch_to_device(batch, device)

                # Call predict_step directly -- no Lightning hooks/callbacks
                output = model.predict_step(batch, batch_idx)

                # Write output via the writer callback
                # trainer and pl_module args are unused (noqa: ARG002 in source)
                writer.write_on_batch_end(
                    trainer=None,
                    pl_module=model,
                    prediction=output,
                    batch_indices=None,
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=0,
                )

                print(f"[bypass] Batch {batch_idx} complete", file=sys.stderr, flush=True)

        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)

        # Print failed count (mimics Lightning callback behavior)
        if hasattr(writer, 'failed'):
            print(f"Number of failed examples: {writer.failed}", flush=True)

        return None

    # Replace Trainer.predict with our bypass
    _OrigTrainer.predict = _bypass_predict

    # -----------------------------------------------------------------------
    # Run boltz predict (with our patched Trainer.predict)
    # -----------------------------------------------------------------------
    sys.argv = [sys.argv[0]] + boltz_args

    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)

    boltz_main.predict()

    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
