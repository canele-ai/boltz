"""Bypass-Lightning wrapper + 75% head pruning.

Zeros out 75% of Pairformer sequence attention heads (least important by
weight norm), then runs bypass inference.
"""
import gc
import sys
import time
import argparse

import torch


_CPU_ONLY_KEYS = frozenset([
    "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
    "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
    "record", "affinity_mw",
])


def _transfer_batch_to_device(batch: dict, device: torch.device) -> dict:
    for key in batch:
        if key not in _CPU_ONLY_KEYS:
            batch[key] = batch[key].to(device)
    return batch


def measure_head_importance_and_rank(model):
    """Return heads sorted by importance (least important first)."""
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
            proj_z_linear = attn.proj_z[1]
            proj_z_linear.weight[head_idx, :].zero_()


def apply_head_pruning_hook(prune_fraction):
    """Monkey-patch Boltz2.predict_step to prune on first call."""
    from boltz.model.models.boltz2 import Boltz2
    _original_predict_step = Boltz2.predict_step
    _pruned = [False]

    def patched_predict_step(self, batch, batch_idx):
        if not _pruned[0]:
            _pruned[0] = True
            ranked = measure_head_importance_and_rank(self)
            n_prune = int(len(ranked) * prune_fraction)
            heads_to_prune = [(b, h) for b, h, _ in ranked[:n_prune]]
            prune_heads_by_zeroing(self, heads_to_prune)
            print(f"[bypass-headprune] Pruned {n_prune}/{len(ranked)} heads",
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
    parser.add_argument("--msa_directory", type=str, default=None)
    parser.add_argument("--cuda_warmup", action="store_true")
    parser.add_argument("--prune_fraction", type=float, default=0.75)

    our_args, boltz_args = parser.parse_known_args()
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    from dataclasses import dataclass
    import boltz.main as boltz_main
    from pytorch_lightning import Trainer as _OrigTrainer

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

    if our_args.bf16_trunk:
        from boltz.model.layers.triangular_mult import (
            TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,
        )

        def forward_outgoing_bf16(self, x, mask, use_kernels=False):
            if use_kernels:
                from boltz.model.layers.triangular_mult import kernel_triangular_mult
                return kernel_triangular_mult(
                    x, direction="outgoing", mask=mask,
                    norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
                    p_in_weight=self.p_in.weight, g_in_weight=self.g_in.weight,
                    norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
                    p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight, eps=1e-5)
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
                    p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight, eps=1e-5)
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

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
    else:
        try:
            import cuequivariance_torch
            if our_args.enable_kernels:
                pass
        except ImportError:
            boltz_args.append("--no_kernels")

    # Apply head pruning hook
    if our_args.prune_fraction > 0:
        apply_head_pruning_hook(our_args.prune_fraction)
        print(f"[bypass-headprune] Head pruning hook installed: {our_args.prune_fraction*100:.0f}%",
              file=sys.stderr, flush=True)

    _cuda_warmup = our_args.cuda_warmup

    def _bypass_predict(trainer_self, model, datamodule=None, return_predictions=False, **kwargs):
        writer = None
        for cb in trainer_self.callbacks:
            if hasattr(cb, 'write_on_batch_end'):
                writer = cb
                break
        if writer is None:
            return _OrigTrainer.predict(trainer_self, model, datamodule=datamodule,
                                        return_predictions=return_predictions, **kwargs)
        if datamodule is not None:
            dataloader = datamodule.predict_dataloader()
        else:
            return None

        device = torch.device("cuda")
        model.eval()
        model.to(device)

        if _cuda_warmup:
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
            except StopIteration:
                pass
            dataloader = datamodule.predict_dataloader()

        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                batch = _transfer_batch_to_device(batch, device)
                output = model.predict_step(batch, batch_idx)
                writer.write_on_batch_end(
                    trainer=None, pl_module=model, prediction=output,
                    batch_indices=None, batch=batch, batch_idx=batch_idx, dataloader_idx=0,
                )

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
