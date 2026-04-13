"""Bypass-Lightning wrapper with layer pruning and token merging support.

Extends the bypass-lightning wrapper with:
- DiffusionTransformer layer pruning (24 -> K layers)
- Pairformer block pruning (64 -> K blocks)
- Token Merging (ToMe) for Pairformer

All stacked on: ODE + TF32 + bf16 + bypass Lightning + recycling_steps=3.
"""
import gc
import sys
import time
import argparse

import torch


# Keys that should NOT be moved to GPU
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
    for key in batch:
        if key not in _CPU_ONLY_KEYS:
            batch[key] = batch[key].to(device)
    return batch


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
    print("[deadend] bf16 trunk patch applied", file=sys.stderr, flush=True)


def patch_layer_pruning(diff_transformer_k=None, pairformer_k=None):
    """Monkey-patch to use fewer layers in DiffusionTransformer and/or Pairformer."""
    if diff_transformer_k is not None:
        from boltz.model.modules.transformers import DiffusionTransformer

        def pruned_forward(self, a, s, z, mask=None, to_keys=None, multiplicity=1, model_cache=None):
            k = min(diff_transformer_k, len(self.layers))
            for i, layer in enumerate(self.layers[:k]):
                layer_cache = None
                if model_cache is not None:
                    prefix_cache = "layer_" + str(i)
                    if prefix_cache not in model_cache:
                        model_cache[prefix_cache] = {}
                    layer_cache = model_cache[prefix_cache]
                a = layer(
                    a, s, z,
                    mask=mask,
                    to_keys=to_keys,
                    multiplicity=multiplicity,
                    layer_cache=layer_cache,
                )
            return a

        DiffusionTransformer.forward = pruned_forward
        print(f"[deadend] DiffusionTransformer pruned to K={diff_transformer_k} layers (of 24)",
              file=sys.stderr, flush=True)

    if pairformer_k is not None:
        from boltz.model.layers.pairformer import PairformerModule
        from boltz.data import const

        def pruned_pf_forward(self, s, z, mask, pair_mask, use_kernels=False):
            if not self.training:
                if z.shape[1] > const.chunk_size_threshold:
                    chunk_size_tri_attn = 128
                else:
                    chunk_size_tri_attn = 512
            else:
                chunk_size_tri_attn = None

            k = getattr(self, '_prune_k', len(self.layers))
            k = min(k, len(self.layers))
            for layer in self.layers[:k]:
                if self.activation_checkpointing and self.training:
                    s, z = torch.utils.checkpoint.checkpoint(
                        layer, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels,
                    )
                else:
                    s, z = layer(s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
            return s, z

        PairformerModule.forward = pruned_pf_forward

        import boltz.model.models.boltz2 as boltz2_mod
        original_eval = boltz2_mod.Boltz2.eval

        def patched_eval(self_model):
            result = original_eval(self_model)
            if hasattr(self_model, 'pairformer_module') and not getattr(self_model, '_pf_tagged', False):
                pf = self_model.pairformer_module
                if hasattr(pf, '_orig_mod'):
                    pf._orig_mod._prune_k = pairformer_k
                    total = len(pf._orig_mod.layers)
                else:
                    pf._prune_k = pairformer_k
                    total = len(pf.layers)
                self_model._pf_tagged = True
                print(f"[deadend] Tagged trunk pairformer: K={pairformer_k} of {total} blocks",
                      file=sys.stderr, flush=True)
            return result

        boltz2_mod.Boltz2.eval = patched_eval
        print(f"[deadend] PairformerModule trunk-only pruning to K={pairformer_k}",
              file=sys.stderr, flush=True)


def patch_token_merging(tome_ratio=0.0, tome_merge_after_layer=0):
    """Monkey-patch PairformerModule with Token Merging."""
    if tome_ratio <= 0:
        return

    from boltz.model.layers.pairformer import PairformerModule
    from boltz.data import const
    from torch import Tensor
    import numpy as np

    def bipartite_soft_matching(s, r, mask):
        B, N, D = s.shape
        device = s.device
        a_idx = torch.arange(0, N, 2, device=device)
        b_idx = torch.arange(1, N, 2, device=device)
        n_a = a_idx.shape[0]
        n_b = b_idx.shape[0]
        r = min(r, n_b)
        if r < 1:
            return None, None, N

        s_a_n = torch.nn.functional.normalize(s[:, a_idx].float(), dim=-1)
        s_b_n = torch.nn.functional.normalize(s[:, b_idx].float(), dim=-1)
        sim = torch.bmm(s_b_n, s_a_n.transpose(1, 2))

        mask_a = mask[:, a_idx]
        mask_b = mask[:, b_idx]
        sim.masked_fill_((mask_b[:, :, None] * mask_a[:, None, :]) == 0, -1e9)

        best_sim, best_a_local = sim.max(dim=-1)
        _, topk_b_local = best_sim.topk(r, dim=-1)
        topk_a_local = torch.gather(best_a_local, 1, topk_b_local)

        new_n = N - r
        orig_to_merged = torch.zeros(B, N, device=device, dtype=torch.long)
        orig_to_merged[:, a_idx] = torch.arange(n_a, device=device).unsqueeze(0)

        b_merged = torch.zeros(B, n_b, device=device, dtype=torch.bool)
        b_merged.scatter_(1, topk_b_local, True)

        kept_b_cumidx = (~b_merged).long().cumsum(dim=1) - 1
        kept_b_pos = kept_b_cumidx + n_a
        b_idx_exp = b_idx.unsqueeze(0).expand(B, n_b)
        orig_to_merged.scatter_(1, b_idx_exp, kept_b_pos)

        topk_b_global = b_idx[topk_b_local]
        orig_to_merged.scatter_(1, topk_b_global, topk_a_local)

        return orig_to_merged, orig_to_merged, new_n

    def merge_sz(s, z, mask, orig_to_merged, new_n):
        B, N, D_s = s.shape
        D_z = z.shape[-1]
        device = s.device

        idx_s = orig_to_merged.unsqueeze(-1).expand(B, N, D_s)
        s_m = torch.zeros(B, new_n, D_s, device=device, dtype=torch.float32)
        s_m.scatter_add_(1, idx_s, s.float())
        counts_s = torch.zeros(B, new_n, 1, device=device, dtype=torch.float32)
        counts_s.scatter_add_(1, orig_to_merged.unsqueeze(-1),
                              torch.ones(B, N, 1, device=device, dtype=torch.float32))
        s_m = (s_m / counts_s.clamp(min=1)).to(s.dtype)

        idx_i = orig_to_merged.unsqueeze(2).expand(B, N, N)
        idx_j = orig_to_merged.unsqueeze(1).expand(B, N, N)
        flat_idx = (idx_i * new_n + idx_j).reshape(B, N * N)

        z_m_flat = torch.zeros(B, new_n * new_n, D_z, device=device, dtype=torch.float32)
        z_m_flat.scatter_add_(1, flat_idx.unsqueeze(-1).expand(B, N * N, D_z),
                              z.reshape(B, N * N, D_z).float())
        counts_z = torch.zeros(B, new_n * new_n, 1, device=device, dtype=torch.float32)
        counts_z.scatter_add_(1, flat_idx.unsqueeze(-1),
                              torch.ones(B, N * N, 1, device=device, dtype=torch.float32))
        z_m = (z_m_flat / counts_z.clamp(min=1)).to(z.dtype).reshape(B, new_n, new_n, D_z)

        mask_m = torch.zeros(B, new_n, device=device, dtype=mask.dtype)
        mask_m.scatter_(1, orig_to_merged, mask)

        return s_m, z_m, mask_m

    def unmerge_sz(s_m, z_m, orig_to_merged, orig_n):
        B = s_m.shape[0]
        D_s = s_m.shape[-1]
        D_z = z_m.shape[-1]
        new_n = s_m.shape[1]

        s = torch.gather(s_m, 1, orig_to_merged.unsqueeze(-1).expand(B, orig_n, D_s))

        idx_i = orig_to_merged.unsqueeze(2).expand(B, orig_n, orig_n)
        idx_j = orig_to_merged.unsqueeze(1).expand(B, orig_n, orig_n)
        flat_idx = (idx_i * new_n + idx_j).reshape(B, orig_n * orig_n, 1).expand(B, orig_n * orig_n, D_z)
        z = torch.gather(z_m.reshape(B, new_n * new_n, D_z), 1, flat_idx).reshape(B, orig_n, orig_n, D_z)

        return s, z

    def tome_forward(self, s, z, mask, pair_mask, use_kernels=False):
        B, N, D_s = s.shape
        r = int(N * tome_ratio)
        num_layers = len(self.layers)
        merge_at = min(tome_merge_after_layer, num_layers)

        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        if r < 1 or N <= 4 or merge_at >= num_layers:
            for layer in self.layers:
                s, z = layer(s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
            return s, z

        # Phase 1: Full-resolution layers
        for i in range(merge_at):
            s, z = self.layers[i](s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)

        # Phase 2: Merge tokens
        orig_to_merged, _, new_n = bipartite_soft_matching(s, r, mask)
        if orig_to_merged is None:
            for i in range(merge_at, num_layers):
                s, z = self.layers[i](s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
            return s, z

        s_m, z_m, mask_m = merge_sz(s, z, mask, orig_to_merged, new_n)
        pair_mask_m = mask_m[:, :, None] * mask_m[:, None, :]

        if not self.training:
            if new_n > const.chunk_size_threshold:
                chunk_size_m = 128
            else:
                chunk_size_m = 512
        else:
            chunk_size_m = None

        # Phase 3: Run remaining layers at reduced resolution
        for i in range(merge_at, num_layers):
            s_m, z_m = self.layers[i](s_m, z_m, mask_m, pair_mask_m, chunk_size_m, use_kernels)

        # Phase 4: Unmerge
        s, z = unmerge_sz(s_m, z_m, orig_to_merged, N)

        return s, z

    PairformerModule.forward = tome_forward
    print(f"[deadend] Token Merging applied: ratio={tome_ratio:.2f}, merge_after_layer={tome_merge_after_layer}",
          file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                        choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    parser.add_argument("--cuda_warmup", action="store_true")
    parser.add_argument("--msa_directory", type=str, default=None)
    # Layer pruning
    parser.add_argument("--diff_transformer_k", type=int, default=None)
    parser.add_argument("--pairformer_k", type=int, default=None)
    # Token merging
    parser.add_argument("--tome_ratio", type=float, default=0.0)
    parser.add_argument("--tome_merge_after_layer", type=int, default=0)

    our_args, boltz_args = parser.parse_known_args()

    torch.set_float32_matmul_precision(our_args.matmul_precision)

    from dataclasses import dataclass
    import boltz.main as boltz_main
    from pytorch_lightning import Trainer as _OrigTrainer

    # ODE diffusion params
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

    # bf16 trunk
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Layer pruning (applied BEFORE token merging since they're mutually exclusive
    # for pairformer — you either prune OR merge, not both)
    if our_args.diff_transformer_k is not None or our_args.pairformer_k is not None:
        patch_layer_pruning(
            diff_transformer_k=our_args.diff_transformer_k,
            pairformer_k=our_args.pairformer_k,
        )

    # Token merging (only for pairformer, conflicts with pairformer pruning)
    if our_args.tome_ratio > 0:
        if our_args.pairformer_k is not None:
            print("[deadend] WARNING: Token merging AND pairformer pruning both set. "
                  "Token merging will override pairformer pruning.",
                  file=sys.stderr, flush=True)
        patch_token_merging(
            tome_ratio=our_args.tome_ratio,
            tome_merge_after_layer=our_args.tome_merge_after_layer,
        )

    # Kernel handling
    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
    else:
        try:
            import cuequivariance_torch
            if our_args.enable_kernels:
                print(f"[deadend] Kernels ENABLED ({cuequivariance_torch.__version__})",
                      file=sys.stderr, flush=True)
        except ImportError:
            boltz_args.append("--no_kernels")

    print(f"[deadend] gamma_0={our_args.gamma_0}, matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"diff_transformer_k={our_args.diff_transformer_k}, "
          f"pairformer_k={our_args.pairformer_k}, "
          f"tome_ratio={our_args.tome_ratio}",
          file=sys.stderr, flush=True)

    # Bypass Lightning Trainer
    _cuda_warmup = our_args.cuda_warmup

    def _bypass_predict(trainer_self, model, datamodule=None, return_predictions=False, **kwargs):
        writer = None
        for cb in trainer_self.callbacks:
            if hasattr(cb, 'write_on_batch_end'):
                writer = cb
                break

        if writer is None:
            print("[deadend] WARNING: No writer callback found, falling back to Lightning",
                  file=sys.stderr, flush=True)
            return _OrigTrainer.predict(trainer_self, model, datamodule=datamodule,
                                        return_predictions=return_predictions, **kwargs)

        if datamodule is not None:
            dataloader = datamodule.predict_dataloader()
        else:
            print("[deadend] ERROR: No datamodule provided", file=sys.stderr, flush=True)
            return None

        device = torch.device("cuda")
        model.eval()
        model.to(device)

        if _cuda_warmup:
            print("[deadend] Running CUDA warmup pass...", file=sys.stderr, flush=True)
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
                print("[deadend] CUDA warmup complete", file=sys.stderr, flush=True)
            except StopIteration:
                print("[deadend] WARNING: No data for warmup", file=sys.stderr, flush=True)
            dataloader = datamodule.predict_dataloader()

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
                print(f"[deadend] Batch {batch_idx} complete", file=sys.stderr, flush=True)

        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)

        if hasattr(writer, 'failed'):
            print(f"Number of failed examples: {writer.failed}", flush=True)

        return None

    _OrigTrainer.predict = _bypass_predict

    sys.argv = [sys.argv[0]] + boltz_args
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)
    boltz_main.predict()
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
