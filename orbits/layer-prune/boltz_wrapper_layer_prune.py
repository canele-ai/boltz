"""Boltz wrapper with layer pruning on top of stacked optimizations.

Extends the stacked wrapper (ODE + TF32 + bf16) with:
4. DiffusionTransformer layer pruning — skip last N layers of the 24-layer token transformer
5. Pairformer layer pruning — skip last N blocks of the 48-block pairformer

The DiffusionTransformer monkey-patch truncates model.structure_module.score_model.token_transformer.layers
after model loading. The Pairformer monkey-patch truncates model.trunk.pairformer.layers.

Both patches are applied via a post-load hook that runs after boltz.main.predict() loads the model
but before inference begins. Since boltz.main.predict() is a monolithic function, we patch the
model class's forward method to intercept and prune on first call.
"""
import sys
import argparse
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
                x,
                direction="outgoing",
                mask=mask,
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
                x,
                direction="incoming",
                mask=mask,
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
    print("[layer-prune] bf16 trunk patch applied")


def patch_layer_pruning(diff_transformer_k=None, pairformer_k=None):
    """Monkey-patch DiffusionTransformer and PairformerModule to use fewer layers.

    This patches the forward() methods to only iterate over the first K layers,
    rather than modifying the module list (which would break checkpoint loading).

    Parameters
    ----------
    diff_transformer_k : int or None
        Number of DiffusionTransformer layers to keep (out of 24).
        None means keep all.
    pairformer_k : int or None
        Number of trunk Pairformer blocks to keep (out of 64 for Boltz2).
        None means keep all. Only the trunk pairformer is pruned;
        the confidence head pairformer is left intact.
    """
    if diff_transformer_k is not None:
        from boltz.model.modules.transformers import DiffusionTransformer
        original_forward = DiffusionTransformer.forward

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
        print(f"[layer-prune] DiffusionTransformer pruned to K={diff_transformer_k} layers (of 24)")

    if pairformer_k is not None:
        # Patch the Boltz2 model class to prune only the TRUNK pairformer,
        # not the confidence head pairformer.
        #
        # Boltz2 uses PairformerArgsV2 with 64 blocks. The confidence head
        # also has a PairformerModule with the same block count -- we must
        # NOT prune that one, as it directly computes pLDDT scores.
        #
        # Strategy: patch PairformerModule.forward with instance-level tagging.
        # Then patch boltz.main.predict to tag the trunk pairformer after
        # model loading.
        from boltz.model.layers.pairformer import PairformerModule
        original_pf_forward = PairformerModule.forward

        def pruned_pf_forward(self, s, z, mask, pair_mask, use_kernels=False):
            from boltz.data import const
            if not self.training:
                if z.shape[1] > const.chunk_size_threshold:
                    chunk_size_tri_attn = 128
                else:
                    chunk_size_tri_attn = 512
            else:
                chunk_size_tri_attn = None

            # Only prune if this instance has been tagged for pruning
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

        # Patch boltz.main.predict to tag the trunk pairformer after model loading.
        # We wrap the model_cls.load_from_checkpoint call by patching Boltz2.eval()
        # which is called right after load_from_checkpoint in predict().
        import boltz.model.models.boltz2 as boltz2_mod
        original_eval = boltz2_mod.Boltz2.eval

        def patched_eval(self_model):
            result = original_eval(self_model)
            # Tag ONLY the trunk pairformer for pruning (not confidence head's)
            if hasattr(self_model, 'pairformer_module') and not getattr(self_model, '_pf_tagged', False):
                pf = self_model.pairformer_module
                # Handle compiled modules
                if hasattr(pf, '_orig_mod'):
                    pf._orig_mod._prune_k = pairformer_k
                    total = len(pf._orig_mod.layers)
                else:
                    pf._prune_k = pairformer_k
                    total = len(pf.layers)
                self_model._pf_tagged = True
                print(f"[layer-prune] Tagged trunk pairformer: K={pairformer_k} of {total} blocks")
            return result

        boltz2_mod.Boltz2.eval = patched_eval
        print(f"[layer-prune] PairformerModule trunk-only pruning to K={pairformer_k} (of 64 for Boltz2)")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")
    # Layer pruning arguments
    parser.add_argument("--diff_transformer_k", type=int, default=None,
                       help="Keep first K of 24 DiffusionTransformer layers")
    parser.add_argument("--pairformer_k", type=int, default=None,
                       help="Keep first K of 48 Pairformer blocks")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz and monkey-patch
    import boltz.main as boltz_main
    from dataclasses import dataclass

    # Monkey-patch Boltz2DiffusionParams for ODE mode
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

    # Apply bf16 trunk patch if requested
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Apply layer pruning patches
    patch_layer_pruning(
        diff_transformer_k=our_args.diff_transformer_k,
        pairformer_k=our_args.pairformer_k,
    )

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[layer-prune] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False

    if our_args.no_kernels_flag:
        boltz_args.append("--no_kernels")
    elif not (our_args.enable_kernels and kernels_available):
        if not kernels_available:
            boltz_args.append("--no_kernels")

    print(f"[layer-prune] gamma_0={our_args.gamma_0}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}, "
          f"diff_transformer_k={our_args.diff_transformer_k}, "
          f"pairformer_k={our_args.pairformer_k}")

    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()


if __name__ == "__main__":
    main()
