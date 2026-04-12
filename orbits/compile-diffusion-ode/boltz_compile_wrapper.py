"""Compile-diffusion-ODE wrapper: torch.compile on the score network in the ODE loop.

Extends bypass-lightning with torch.compile applied to the score model (DiffusionModule)
that gets called 12 times during the ODE sampling loop. The compilation cost is amortized
over all 12 calls per forward pass, and kernel fusion / CUDA graph benefits multiply.

Strategy:
1. Monkey-patch Trainer.predict() (same as bypass-lightning)
2. After model is on GPU, apply torch.compile to model.structure_module.score_model
3. Run CUDA warmup pass to trigger compilation (excluded from timing)
4. Run timed inference with the compiled score model

Compile modes tested:
- "default": standard torch.compile
- "reduce-overhead": CUDA graphs for reduced kernel launch overhead
- "max-autotune": aggressive kernel tuning

Usage:
    python boltz_compile_wrapper.py input.yaml --out_dir out \
        --sampling_steps 12 --recycling_steps 0 \
        --gamma_0 0.0 --matmul_precision high --bf16_trunk \
        --compile_score --compile_mode default
"""
import gc
import sys
import time
import argparse

import torch


# Keys that should NOT be moved to GPU (non-tensor or CPU-only data).
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
    """Move batch tensors to device."""
    for key in batch:
        if key not in _CPU_ONLY_KEYS:
            batch[key] = batch[key].to(device)
    return batch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    # Custom flags (stripped before passing to boltz)
    parser.add_argument("--matmul_precision", default="highest",
                        choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--msa_directory", type=str, default=None)
    parser.add_argument("--cuda_warmup", action="store_true",
                        help="Run one forward pass before timing to warm CUDA + compile")
    # torch.compile flags
    parser.add_argument("--compile_score", action="store_true",
                        help="Apply torch.compile to score_model in the diffusion loop")
    parser.add_argument("--compile_transformer", action="store_true",
                        help="Apply torch.compile to just the token_transformer inside score_model")
    parser.add_argument("--compile_mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")

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
        print("[compile-wrapper] bf16 trunk patch applied", file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Kernel handling
    # -----------------------------------------------------------------------
    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
        print("[compile-wrapper] Kernels DISABLED", file=sys.stderr, flush=True)
    else:
        try:
            import cuequivariance_torch
            if our_args.enable_kernels:
                print(f"[compile-wrapper] Kernels ENABLED (cuequivariance_torch {cuequivariance_torch.__version__})",
                      file=sys.stderr, flush=True)
        except ImportError:
            boltz_args.append("--no_kernels")
            print("[compile-wrapper] Kernels DISABLED (not installed)", file=sys.stderr, flush=True)

    print(f"[compile-wrapper] gamma_0={our_args.gamma_0}, noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, bf16_trunk={our_args.bf16_trunk}, "
          f"compile_score={our_args.compile_score}, compile_transformer={our_args.compile_transformer}, "
          f"compile_mode={our_args.compile_mode}, cuda_warmup={our_args.cuda_warmup}",
          file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Monkey-patch Trainer.predict to bypass Lightning + compile score model
    # -----------------------------------------------------------------------
    _cuda_warmup = our_args.cuda_warmup
    _compile_score = our_args.compile_score
    _compile_transformer = our_args.compile_transformer
    _compile_mode = our_args.compile_mode

    def _bypass_predict(trainer_self, model, datamodule=None, return_predictions=False, **kwargs):
        """Replace Lightning Trainer.predict with direct inference + torch.compile."""
        writer = None
        for cb in trainer_self.callbacks:
            if hasattr(cb, 'write_on_batch_end'):
                writer = cb
                break

        if writer is None:
            print("[compile-wrapper] WARNING: No writer callback found, falling back",
                  file=sys.stderr, flush=True)
            return _OrigTrainer.predict(trainer_self, model, datamodule=datamodule,
                                        return_predictions=return_predictions, **kwargs)

        if datamodule is not None:
            dataloader = datamodule.predict_dataloader()
        else:
            print("[compile-wrapper] ERROR: No datamodule provided", file=sys.stderr, flush=True)
            return None

        # Move model to GPU
        device = torch.device("cuda")
        model.eval()
        model.to(device)

        # ---------------------------------------------------------------
        # Apply torch.compile to the score model
        # ---------------------------------------------------------------
        if _compile_score:
            print(f"[compile-wrapper] Compiling score_model with mode='{_compile_mode}'...",
                  file=sys.stderr, flush=True)
            t_compile_start = time.perf_counter()

            # The score model is inside AtomDiffusion (structure_module)
            # It gets called 12 times per ODE sampling loop
            model.structure_module.score_model = torch.compile(
                model.structure_module.score_model,
                mode=_compile_mode,
                dynamic=False,
                fullgraph=False,  # allow graph breaks if needed
            )

            t_compile_wrap = time.perf_counter()
            print(f"[compile-wrapper] torch.compile wrapper created in "
                  f"{t_compile_wrap - t_compile_start:.2f}s",
                  file=sys.stderr, flush=True)

        elif _compile_transformer:
            print(f"[compile-wrapper] Compiling token_transformer with mode='{_compile_mode}'...",
                  file=sys.stderr, flush=True)
            t_compile_start = time.perf_counter()

            # Compile just the DiffusionTransformer (24-layer, heaviest submodule)
            model.structure_module.score_model.token_transformer = torch.compile(
                model.structure_module.score_model.token_transformer,
                mode=_compile_mode,
                dynamic=False,
                fullgraph=False,
            )

            t_compile_wrap = time.perf_counter()
            print(f"[compile-wrapper] torch.compile wrapper (transformer) created in "
                  f"{t_compile_wrap - t_compile_start:.2f}s",
                  file=sys.stderr, flush=True)

        # CUDA warmup: run one forward pass to trigger kernel JIT + compilation
        if _cuda_warmup:
            print("[compile-wrapper] Running CUDA warmup pass (includes compile)...",
                  file=sys.stderr, flush=True)
            t_warmup_start = time.perf_counter()
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
                t_warmup_end = time.perf_counter()
                print(f"[compile-wrapper] Warmup complete in {t_warmup_end - t_warmup_start:.1f}s",
                      file=sys.stderr, flush=True)
            except StopIteration:
                print("[compile-wrapper] WARNING: No data for warmup",
                      file=sys.stderr, flush=True)
            # Re-create dataloader for the real run
            dataloader = datamodule.predict_dataloader()

        # Direct inference loop
        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                batch = _transfer_batch_to_device(batch, device)
                output = model.predict_step(batch, batch_idx)
                writer.write_on_batch_end(
                    trainer=None,
                    pl_module=model,
                    prediction=output,
                    batch_indices=None,
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=0,
                )
                print(f"[compile-wrapper] Batch {batch_idx} complete",
                      file=sys.stderr, flush=True)

        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)

        if hasattr(writer, 'failed'):
            print(f"Number of failed examples: {writer.failed}", flush=True)

        return None

    _OrigTrainer.predict = _bypass_predict

    # -----------------------------------------------------------------------
    # Run boltz predict
    # -----------------------------------------------------------------------
    sys.argv = [sys.argv[0]] + boltz_args

    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)
    boltz_main.predict()
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
