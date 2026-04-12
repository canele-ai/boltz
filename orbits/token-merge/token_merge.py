"""Token Merging (ToMe) for Boltz-2 Pairformer speedup.

Implements bipartite soft matching from Bolya et al. 2023 adapted for
structure prediction.

Strategy: partial-layer merging. Run the first K layers at full resolution,
then merge tokens and run remaining layers at reduced resolution, then
unmerge. This preserves the geometric precision of early layers while
getting speedup from later, more abstract layers.

For z (pair) merging, we operate in float32 to avoid dtype issues under
autocast/bf16. The merge/unmerge overhead is one-time O(N^2) while the
pairformer savings are (64-K) * O(N^2 - N'^2).
"""

import torch
from torch import Tensor


def bipartite_soft_matching(
    s: Tensor,
    r: int,
    mask: Tensor,
) -> tuple[Tensor, Tensor, int]:
    """Bipartite soft matching — returns merge mapping.

    Parameters
    ----------
    s : (B, N, D) token representations
    r : number of tokens to merge away
    mask : (B, N) token validity mask

    Returns
    -------
    orig_to_merged : (B, N) mapping from original to merged position
    unmerge_indices : same as orig_to_merged (used for unmerge gather)
    new_n : reduced token count
    """
    B, N, D = s.shape
    device = s.device

    a_idx = torch.arange(0, N, 2, device=device)
    b_idx = torch.arange(1, N, 2, device=device)
    n_a = a_idx.shape[0]
    n_b = b_idx.shape[0]
    r = min(r, n_b)
    if r < 1:
        return None, None, N

    # Cosine similarity between set B and set A
    s_a_n = torch.nn.functional.normalize(s[:, a_idx].float(), dim=-1)
    s_b_n = torch.nn.functional.normalize(s[:, b_idx].float(), dim=-1)
    sim = torch.bmm(s_b_n, s_a_n.transpose(1, 2))  # (B, n_b, n_a)

    # Mask padding
    mask_a = mask[:, a_idx]
    mask_b = mask[:, b_idx]
    sim.masked_fill_(
        (mask_b[:, :, None] * mask_a[:, None, :]) == 0, -1e9
    )

    # Best A match for each B token, then take top-r
    best_sim, best_a_local = sim.max(dim=-1)  # (B, n_b)
    _, topk_b_local = best_sim.topk(r, dim=-1)  # (B, r)
    topk_a_local = torch.gather(best_a_local, 1, topk_b_local)  # (B, r)

    new_n = N - r

    # Build mapping: orig_pos -> merged_pos
    # Layout: [A tokens (0..n_a-1)] [kept B tokens (n_a..new_n-1)]
    orig_to_merged = torch.zeros(B, N, device=device, dtype=torch.long)

    # A tokens at positions 0..n_a-1
    orig_to_merged[:, a_idx] = torch.arange(n_a, device=device).unsqueeze(0)

    # Mark merged B tokens
    b_merged = torch.zeros(B, n_b, device=device, dtype=torch.bool)
    b_merged.scatter_(1, topk_b_local, True)

    # Kept B tokens get sequential positions starting at n_a
    kept_b_cumidx = (~b_merged).long().cumsum(dim=1) - 1
    kept_b_pos = kept_b_cumidx + n_a
    b_idx_exp = b_idx.unsqueeze(0).expand(B, n_b)
    orig_to_merged.scatter_(1, b_idx_exp, kept_b_pos)

    # Merged B tokens -> their destination A token's position
    topk_b_global = b_idx[topk_b_local]  # (B, r)
    orig_to_merged.scatter_(1, topk_b_global, topk_a_local)

    return orig_to_merged, orig_to_merged, new_n


def merge_sz(s, z, mask, orig_to_merged, new_n):
    """Merge s and z using scatter_add averaging. Returns merged tensors."""
    B, N, D_s = s.shape
    D_z = z.shape[-1]
    device = s.device

    # Merge s (in float32)
    idx_s = orig_to_merged.unsqueeze(-1).expand(B, N, D_s)
    s_m = torch.zeros(B, new_n, D_s, device=device, dtype=torch.float32)
    s_m.scatter_add_(1, idx_s, s.float())
    counts_s = torch.zeros(B, new_n, 1, device=device, dtype=torch.float32)
    counts_s.scatter_add_(1, orig_to_merged.unsqueeze(-1),
                          torch.ones(B, N, 1, device=device, dtype=torch.float32))
    s_m = (s_m / counts_s.clamp(min=1)).to(s.dtype)

    # Merge z (in float32, vectorized)
    idx_i = orig_to_merged.unsqueeze(2).expand(B, N, N)
    idx_j = orig_to_merged.unsqueeze(1).expand(B, N, N)
    flat_idx = (idx_i * new_n + idx_j).reshape(B, N * N)

    z_m_flat = torch.zeros(B, new_n * new_n, D_z, device=device, dtype=torch.float32)
    z_m_flat.scatter_add_(1, flat_idx.unsqueeze(-1).expand(B, N * N, D_z), z.reshape(B, N * N, D_z).float())
    counts_z = torch.zeros(B, new_n * new_n, 1, device=device, dtype=torch.float32)
    counts_z.scatter_add_(1, flat_idx.unsqueeze(-1),
                          torch.ones(B, N * N, 1, device=device, dtype=torch.float32))
    z_m = (z_m_flat / counts_z.clamp(min=1)).to(z.dtype).reshape(B, new_n, new_n, D_z)

    # Merge mask
    mask_m = torch.zeros(B, new_n, device=device, dtype=mask.dtype)
    mask_m.scatter_(1, orig_to_merged, mask)

    return s_m, z_m, mask_m


def unmerge_sz(s_m, z_m, orig_to_merged, orig_n):
    """Unmerge s and z back to original size via gather."""
    B = s_m.shape[0]
    D_s = s_m.shape[-1]
    D_z = z_m.shape[-1]
    new_n = s_m.shape[1]

    # Unmerge s
    s = torch.gather(s_m, 1, orig_to_merged.unsqueeze(-1).expand(B, orig_n, D_s))

    # Unmerge z
    idx_i = orig_to_merged.unsqueeze(2).expand(B, orig_n, orig_n)
    idx_j = orig_to_merged.unsqueeze(1).expand(B, orig_n, orig_n)
    flat_idx = (idx_i * new_n + idx_j).reshape(B, orig_n * orig_n, 1).expand(B, orig_n * orig_n, D_z)
    z = torch.gather(z_m.reshape(B, new_n * new_n, D_z), 1, flat_idx).reshape(B, orig_n, orig_n, D_z)

    return s, z


def patch_pairformer_with_tome(merge_ratio: float = 0.2, merge_after_layer: int = 0):
    """Monkey-patch PairformerModule.forward with partial-layer token merging.

    Parameters
    ----------
    merge_ratio : float
        Fraction of tokens to merge (0.0 = disabled, 0.5 = merge half).
    merge_after_layer : int
        Run this many layers at full resolution before merging.
        0 = merge before all layers (maximum speedup, minimum quality).
        32 = merge only for the last 32 layers (balanced).
    """
    from boltz.model.layers.pairformer import PairformerModule
    from boltz.data import const

    def tome_forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> tuple[Tensor, Tensor]:
        B, N, D_s = s.shape
        r = int(N * merge_ratio)
        num_layers = len(self.layers)
        merge_at = min(merge_after_layer, num_layers)

        if r < 1 or N <= 4 or merge_at >= num_layers:
            # Fall back to standard forward
            if not self.training:
                if z.shape[1] > const.chunk_size_threshold:
                    chunk_size_tri_attn = 128
                else:
                    chunk_size_tri_attn = 512
            else:
                chunk_size_tri_attn = None
            for layer in self.layers:
                s, z = layer(s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
            return s, z

        # Determine chunk size
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        # Phase 1: Full-resolution layers (0..merge_at-1)
        for i in range(merge_at):
            s, z = self.layers[i](s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)

        # Phase 2: Merge tokens
        orig_to_merged, _, new_n = bipartite_soft_matching(s, r, mask)
        if orig_to_merged is None:
            # Can't merge, run remaining layers at full resolution
            for i in range(merge_at, num_layers):
                s, z = self.layers[i](s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
            return s, z

        s_m, z_m, mask_m = merge_sz(s, z, mask, orig_to_merged, new_n)
        pair_mask_m = mask_m[:, :, None] * mask_m[:, None, :]

        # Update chunk size for reduced N
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
    print(f"[tome] Patch applied: merge_ratio={merge_ratio:.2f}, merge_after_layer={merge_after_layer}")
