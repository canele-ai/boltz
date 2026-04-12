---
strategy: token-merge
type: experiment
status: negative
eval_version: eval-v3
metric: 0.87x (slower than baseline, fails quality gate)
issue: 27
parents:
  - orbit/eval-v2-winner
---

# Token Merging for Pairformer Speedup — NEGATIVE RESULT

## Hypothesis
Apply Token Merging (ToMe, Bolya et al. 2023) to reduce the Pairformer's
token count N, cutting its O(N^2) pair computation from O(N^2) to O((N*r)^2)
where r is the retention ratio.

## Implementation
- Bipartite soft matching on s (single repr) using cosine similarity
- Full merge of both s (single) and z (pair) representations via scatter_add
- Unmerge after pairformer via gather
- Partial-layer merging: run first K layers at full resolution, merge for rest
- All operations vectorized (no Python loops over batch/token dims)

## Results

### Experiment 1: 10% merge, all layers (merge_after_layer=0)
| Complex | Time (s) | pLDDT | Baseline pLDDT |
|---------|----------|-------|----------------|
| small   | 96.0*    | 0.607 | 0.834          |
| medium  | 43.7     | 0.452 | 0.509          |
| large   | 46.5     | 0.572 | 0.807          |

*includes cold-start model download

Speedup: 0.86x (SLOWER). pLDDT delta: -17.3pp. **FAIL**.

### Experiment 2: 30% merge, after layer 32
| Complex | Time (s) | pLDDT | Baseline pLDDT |
|---------|----------|-------|----------------|
| small   | 99.4*    | 0.369 | 0.834          |
| medium  | 41.5     | 0.363 | 0.509          |
| large   | 42.8     | 0.818 | 0.807          |

Speedup: 0.87x (SLOWER). pLDDT delta: -20.0pp. **FAIL**.

### Experiment 3: 50% merge, all layers (maximum theoretical savings)
| Complex | Time (s) | pLDDT | Baseline pLDDT |
|---------|----------|-------|----------------|
| small   | 98.4*    | 0.376 | 0.834          |
| medium  | 41.2     | 0.371 | 0.509          |
| large   | 45.7     | 0.645 | 0.807          |

Speedup: 0.87x (SLOWER). pLDDT delta: -25.3pp. **FAIL**.

### Baseline (no merging, same wrapper)
| Complex | Time (s) | pLDDT |
|---------|----------|-------|
| small   | 84.7*    | 0.878 |
| medium  | 36.5     | 0.485 |
| large   | 41.6     | 0.816 |

## Why Token Merging Fails for the Pairformer

### 1. Merge/unmerge overhead cancels out savings
The z (pair) representation has shape (B, N, N, D_z). Merging it requires
scatter_add over N^2 * D_z elements; unmerging requires gather over the same.
This is O(N^2) work — the same order as one pairformer layer. For the savings
to exceed this overhead, we'd need the pairformer to run many layers on the
reduced set. But even with all 64 layers merged (50% merge = 4x theoretical
reduction), the overhead still exceeded the savings by ~13%.

The scatter_add/gather pattern has poor GPU memory coalescing, making the
practical cost even higher than the theoretical O(N^2).

### 2. Quality degradation is catastrophic
Even at 10% merge (removing 1 in 10 tokens), pLDDT drops by 17 percentage
points. The Pairformer's pair representation encodes precise pairwise geometric
relationships between tokens. Averaging these relationships (as required by
merging) destroys spatial information that cannot be recovered by unmerging.

This is fundamentally different from Vision Transformers (where ToMe was
designed) because ViTs have no pair representation — only token embeddings.
The ViT's self-attention implicitly computes pairwise relationships at each
layer, so they're resilient to token reduction. The Pairformer's explicit
pair representation makes it brittly dependent on exact token count and
identity.

### 3. The pair representation IS the bottleneck AND the barrier
The pair representation (z) is simultaneously:
- Why the Pairformer is O(N^2) and slow (the reason to try merging)
- Why merging is expensive (O(N^2) merge/unmerge cost)
- Why merging destroys quality (pair info is precision-critical)

This creates a fundamental incompatibility between token merging and the
Pairformer architecture.

## Implications for Future Work
- **Token merging is not viable** for Pairformer-like architectures with
  explicit pair representations.
- The ~35s fixed-cost floor (model loading + Pairformer trunk + confidence)
  identified by previous orbits remains the binding constraint.
- Future speedup approaches should focus on **algorithmic changes to the
  Pairformer itself** (e.g., sparse attention, low-rank pair representations)
  rather than token-level tricks designed for simpler architectures.
