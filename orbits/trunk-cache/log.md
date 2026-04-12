---
strategy: trunk-cache
type: experiment
status: negative
eval_version: eval-v4
metric: null
issue: 32
parents:
  - orbit/eval-v2-winner
---

# Trunk Cache — Analysis & Negative Result

## Hypothesis

In binder design, the target protein is fixed across many predictions. If the
Pairformer trunk output (s_trunk, z_trunk) for the target could be computed once
and reused, we would skip the MSA module (31% of GPU time) and Pairformer (26%)
for repeated predictions of the same target with different binders.

## Architecture Analysis

After reading the full forward pass in `src/boltz/model/models/boltz2.py` (lines
401-606) and the Pairformer in `src/boltz/model/layers/pairformer.py`, the trunk
caching hypothesis is **not viable** for the following reasons:

### 1. Joint tokenisation prevents dimension-level separation

The forward pass (boltz2.py line 414) calls `input_embedder(feats)` on the
*entire* complex — target + binder tokens concatenated. The resulting `s_inputs`
has shape `(B, N_total, D)` where `N_total = N_target + N_binder`. When the
binder changes, N_binder (and therefore N_total) can change, invalidating any
cached tensors.

### 2. Pair representation mixes all chains from the start

The pair representation `z` is initialised (lines 420-429) as:

```python
z_init = z_init_1(s_inputs)[:, :, None] + z_init_2(s_inputs)[:, None, :]
z_init += relative_position_encoding + token_bonds + contact_conditioning
```

This creates a `(B, N_total, N_total, D)` tensor where target-binder cross-terms
are present from the very first layer. There is no block-diagonal structure to
exploit.

### 3. Triangle operations contract over the full sequence dimension

Each Pairformer layer applies:
- `TriangleMultiplicationOutgoing`: `einsum("bikd,bjkd->bijd", a, b)` — contracts
  over k ∈ [0, N_total), mixing target and binder.
- `TriangleMultiplicationIncoming`: `einsum("bkid,bkjd->bijd", a, b)` — same.
- `TriangleAttention{Starting,Ending}Node`: attention over full rows/columns.
- `AttentionPairBias`: sequence attention biased by the full z matrix.

Even if we froze the target-target block of z, the target-binder and binder-target
blocks would change with each new binder, and these cross-blocks participate in
every triangle contraction. The cost of recomputing just the cross-blocks is
comparable to running the full Pairformer.

### 4. MSA module depends on binder features

The MSA module (trunkv2.py line 567) takes `z`, `emb` (s_inputs), and `feats` as
input. The MSA sequences include the binder's profile, so even the MSA output
changes when the binder changes. Pre-caching the MSA module output for the target
alone doesn't work because the outer-product-mean and pair-weighted-averaging
operations couple target and binder MSA rows.

### 5. Eval harness uses different targets

The eval-v4 test set contains 3 *different* complexes (small, medium, large) —
not the same target with different binders. To demonstrate trunk caching, we'd
need to run the same target with different binder sequences, which the current
eval harness doesn't support. Even if we added this, the architectural blockers
above prevent meaningful caching.

## What Could Work (Future Directions)

1. **Warm-start recycling**: For repeated predictions of similar complexes, use
   the previous s,z as the recycling initialisation (instead of zeros). This
   doesn't cache the trunk but reduces the number of recycling steps needed.
   However, eval-v2-winner already uses 0 recycling steps, making this moot.

2. **Block-sparse Pairformer**: If the Pairformer were redesigned with explicit
   intra-chain and inter-chain blocks (like some MSA Transformers), the
   intra-target block could be cached. This is a model architecture change, not
   an inference optimisation.

3. **Partial input embedding cache**: The atom encoder portion of
   `InputEmbedder.forward()` for target atoms is deterministic and could be
   cached. But this is <5% of total trunk time, so the speedup would be
   negligible.

## Conclusion

**Status: Negative.** The Boltz-2 architecture processes target and binder tokens
jointly through all trunk modules (input embedding, MSA, Pairformer). The pair
representation z is (N_total × N_total × D) with no block-diagonal structure,
and triangle operations contract over the full sequence dimension. Caching any
portion of the trunk for a fixed target requires recomputing all cross-chain terms,
which costs nearly as much as the full computation.

No eval was run because the hypothesis is architecturally blocked — there is no
config change or monkey-patch that implements trunk caching without a fundamental
redesign of the Pairformer's triangle operations.
