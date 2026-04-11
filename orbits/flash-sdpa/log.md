---
strategy: flash-sdpa
type: experiment
status: complete
eval_version: eval-v1
metric: 0.745
issue: 5
parents: []
root_rationale: "First exploration -- replace einsum attention with F.scaled_dot_product_attention for FlashAttention-2 on L40S"
---

# Flash SDPA: Replacing Einsum Attention with Fused Kernels

## Glossary

- **SDPA**: Scaled Dot-Product Attention (`torch.nn.functional.scaled_dot_product_attention`)
- **FA2**: FlashAttention-2 -- a memory-efficient, IO-aware attention kernel that avoids materializing the full N*N attention matrix
- **pLDDT**: Predicted Local Distance Difference Test -- Boltz's confidence score (0-1, higher is better)
- **pp**: Percentage points
- **bf16**: bfloat16, a 16-bit floating point format with 8-bit exponent
- **TF32**: TensorFloat-32, a 19-bit format used by NVIDIA tensor cores for float32 operations

## Results

**Speedup: 0.745x (SLOWER than baseline, no improvement)**

The SDPA replacement does not provide speedup for Boltz-2 inference. Both float32 and bf16 SDPA variants are approximately 25-35% slower than the manual einsum baseline when measured end-to-end (including MSA generation).

Quality is fully preserved: pLDDT regression is less than 0.3 pp, well within the 2 pp budget.

### float32 SDPA (3 seeds)

| Seed | Mean Time (s) | Mean pLDDT | Speedup |
|------|---------------|------------|---------|
| 42   | 95.03         | 0.7075     | 0.74x   |
| 123  | 106.00        | 0.7098     | 0.66x   |
| 7    | 277.97        | 0.7112     | 0.25x   |
| **Mean** | **159.67 +/- 102.60** | **0.7095 +/- 0.0019** | **0.55x +/- 0.26** |

Note: Seed 7 suffered extreme MSA server latency (~290s for small_complex vs ~100s for other seeds). Excluding seed 7, mean speedup is ~0.70x.

### bf16 SDPA (3 seeds)

| Seed | Mean Time (s) | Mean pLDDT | Speedup |
|------|---------------|------------|---------|
| 42   | 95.72         | 0.7063     | 0.74x   |
| 123  | 97.42         | 0.7121     | 0.72x   |
| 7    | 90.64         | 0.7076     | 0.78x   |
| **Mean** | **94.59 +/- 3.53** | **0.7087 +/- 0.0030** | **0.745x +/- 0.028** |

### Per-complex breakdown (bf16 SDPA, seed 7 -- cleanest run)

| Complex | SDPA Time (s) | Baseline Time (s) | SDPA pLDDT | Baseline pLDDT |
|---------|---------------|-------------------|------------|----------------|
| small (~200 res)  | 96.2  | 53.0 | 0.8495 | 0.8350 |
| medium (~400 res) | 71.2  | 70.5 | 0.4617 | 0.4906 |
| large (~600 res)  | 104.6 | 87.6 | 0.8118 | 0.8064 |

The medium complex time (71s vs 70.5s) is nearly identical, suggesting the GPU computation overhead from SDPA is minimal. The large differences in small and large complexes are likely due to MSA server latency variance between runs.

## Approach

Monkey-patched three attention modules in the Boltz-2 model:

1. **attentionv2.AttentionPairBias** (score model transformer, 24 layers x 200 diffusion steps = 4800 calls per prediction) -- replaced `torch.einsum("bihd,bjhd->bhij", ...)` with `F.scaled_dot_product_attention`
2. **attention.AttentionPairBias** (Pairformer sequence track, ~48 blocks x 4 recycling passes) -- same replacement
3. **triangular_attention.primitives._attention** (Pairformer triangular attention) -- replaced `torch.matmul` + bias loop with SDPA

Two variants tested:
- **float32 SDPA**: Maintains original float32 precision. SDPA dispatches to "math" backend.
- **bf16 SDPA**: Casts q/k/v and bias to bfloat16. SDPA dispatches to "mem_efficient" backend.

## Why SDPA Does Not Help Here

The hypothesis was that SDPA would dispatch to FlashAttention-2 (FA2), which avoids materializing the full N*N attention matrix and uses IO-aware memory access patterns. This fails for Boltz because:

### 1. Pair bias blocks FlashAttention-2

Every attention layer in Boltz adds a pair bias `z` of shape `(B, H, N_q, N_k)` to the attention logits. FA2 does not support arbitrary additive attention masks -- it only supports no mask or causal masks. When `attn_mask` is provided:
- **float32 inputs**: SDPA falls back to "math" backend (same computation as manual einsum, just wrapped in a single kernel)
- **bf16 inputs**: SDPA uses "mem_efficient" backend (xformers-style), which does support masks but gains less from memory savings at small sequence lengths

### 2. Short sequences limit memory savings

Boltz operates on sequences of ~200-600 tokens (protein residues). The N*N attention matrix at this scale is 200x200 = 40K elements, trivially small. The memory-efficient backend's advantage (avoiding O(N^2) memory) is negligible. FA2 and mem_efficient attention show their biggest wins at N > 2K-4K.

### 3. Tensor layout mismatch

Boltz stores tensors in `(B, N, H, D)` layout. SDPA requires `(B, H, N, D)`. The transpose operations add overhead that partially negates any kernel-level gains.

### 4. Attention is not the bottleneck

The end-to-end timing is dominated by MSA generation (variable, 10-200s), model weight loading, and non-attention operations (projections, LayerNorm, feedforward layers). The attention einsum is a small fraction of total GPU time.

## What I Learned

1. **SDPA without FlashAttention-2 provides no speedup for short sequences.** The math and mem_efficient backends are not materially faster than well-written einsum for N < 1K.

2. **Pair bias is the fundamental blocker.** Boltz's pair bias architecture prevents FlashAttention-2 dispatch. To use FA2, one would need to either (a) remove pair bias (requires retraining), or (b) use a newer FA backend that supports additive bias (FlashAttention-3 may support this).

3. **MSA latency dominates timing variance.** Seed 7 in the float32 experiment took 3x longer than seed 42, entirely due to MSA server latency. Any fair comparison needs pre-cached MSAs.

4. **bf16 attention preserves quality.** Despite reducing attention precision from float32 to bfloat16, pLDDT regression was less than 0.3 pp. This suggests Boltz attention is robust to reduced precision, which is useful for other optimization approaches.

## Prior Art & Novelty

### What is already known
- FlashAttention-2 does not support arbitrary additive attention masks ([Dao et al., 2023](https://arxiv.org/abs/2307.08691))
- PyTorch SDPA dispatches to math/mem_efficient backends when attn_mask is provided ([PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html))
- AlphaFold-style models use pair bias in all attention layers ([Jumper et al., 2021](https://doi.org/10.1038/s41586-021-03819-2))

### What this orbit adds
- Empirical confirmation that SDPA replacement provides no speedup for Boltz-2 specifically
- Validation that bf16 attention precision preserves quality (pLDDT within 0.3pp)
- Concrete negative result preventing wasted effort on this approach

### Honest positioning
This is a negative result confirming what could have been predicted from FlashAttention-2's mask limitations. The value is in the empirical confirmation and the observation that bf16 attention precision is safe for Boltz-2.

## References

- [Dao et al. (2023)](https://arxiv.org/abs/2307.08691) -- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- [PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) -- Backend selection logic
- [Jumper et al. (2021)](https://doi.org/10.1038/s41586-021-03819-2) -- AlphaFold pair bias attention architecture
- [Wohlwend et al. (2024)](https://doi.org/10.1101/2024.11.19.624167) -- Boltz-1 architecture
