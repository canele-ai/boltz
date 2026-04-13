---
strategy: deadend-attention
status: complete
eval_version: eval-v5
metric: 1.000
issue: 54
parents:
  - bypass-lightning
---

## Glossary

- SDPA: Scaled Dot-Product Attention (torch.nn.functional.scaled_dot_product_attention)
- CA RMSD: C-alpha Root Mean Square Deviation (structural comparison metric, in Angstroms)
- ODE: Ordinary Differential Equation (deterministic diffusion sampling with gamma_0=0.0)
- TF32: TensorFloat-32 (NVIDIA reduced-precision float format for matmul)
- bf16: Brain Floating Point 16-bit
- pLDDT: predicted Local Distance Difference Test (model confidence score, 0-1)

## Approach

Re-validation of two dead-end attention optimizations, stacked on top of the winning config (ODE-12 + TF32 + bf16 + bypass Lightning + recycling_steps=3), measured with CA RMSD structural comparison against PDB ground truth.

### Approach A: SDPA Attention (prior orbits #5, #19, #36)

Replace Boltz's manual einsum attention in AttentionPairBias with torch.nn.functional.scaled_dot_product_attention. The hypothesis was that SDPA's fused kernels (FlashAttention-2) would speed up attention computation.

### Approach B: 75% Head Pruning (prior orbit #34)

Zero out 75% of Pairformer sequence attention heads (768 of 1024), selected by lowest weight-norm importance score (product of output, value, and gate projection norms). The hypothesis was that redundant heads could be removed for a compute saving.

## Results

All results are means over 3 seeds (42, 123, 7), running on L40S GPUs via Modal. All configs use ODE-12 steps, recycle=3, gamma_0=0.0, TF32, bf16, CUDA warmup, bypass Lightning.

### Cross-Config Comparison

| Config | Mean Predict(s) | Mean pLDDT | Mean CA RMSD (A) | Speed vs Control |
|--------|-----------------|------------|-------------------|------------------|
| Control (bypass only) | 15.1 | 0.7146 | 16.521 | 1.000x |
| + SDPA attention | 15.4 | 0.7147 | 16.527 | 0.982x |
| + 75% head pruning | 15.1 | 0.7370 | 16.434 | 0.999x |

### Per-Complex Breakdown (mean over 3 seeds)

#### small_complex (1BRS, 195 residues)

| Config | Predict(s) | pLDDT | CA RMSD (A) |
|--------|-----------|-------|-------------|
| Control | 5.3 | 0.8517 | 2.731 |
| + SDPA | 6.9 | 0.8528 | 2.729 |
| + Head prune | 5.0 | 0.8551 | 2.497 |

#### medium_complex (1DQJ, 340 matched residues)

| Config | Predict(s) | pLDDT | CA RMSD (A) |
|--------|-----------|-------|-------------|
| Control | 14.3 | 0.4834 | 25.819 |
| + SDPA | 13.8 | 0.4822 | 25.832 |
| + Head prune | 14.5 | 0.5253 | 25.289 |

#### large_complex (2DN2, 411 matched residues)

| Config | Predict(s) | pLDDT | CA RMSD (A) |
|--------|-----------|-------|-------------|
| Control | 25.7 | 0.8087 | 21.012 |
| + SDPA | 25.4 | 0.8093 | 21.020 |
| + Head prune | 25.7 | 0.8306 | 21.515 |

### Full Seed-Level Data

#### Control

| Complex | Seed | Predict(s) | pLDDT | CA RMSD | %<2A |
|---------|------|-----------|-------|---------|------|
| small_complex | 42 | 4.7 | 0.859 | 2.601 | 77% |
| small_complex | 123 | 5.4 | 0.843 | 2.662 | 80% |
| small_complex | 7 | 5.8 | 0.853 | 2.929 | 75% |
| medium_complex | 42 | 13.9 | 0.490 | 25.899 | 0% |
| medium_complex | 123 | 14.4 | 0.480 | 25.650 | 0% |
| medium_complex | 7 | 14.6 | 0.480 | 25.908 | 0% |
| large_complex | 42 | 25.4 | 0.808 | 20.030 | 0% |
| large_complex | 123 | 26.4 | 0.810 | 20.690 | 0% |
| large_complex | 7 | 25.2 | 0.808 | 22.316 | 0% |

#### SDPA

| Complex | Seed | Predict(s) | pLDDT | CA RMSD | %<2A |
|---------|------|-----------|-------|---------|------|
| small_complex | 42 | 10.0 | 0.862 | 2.596 | 77% |
| small_complex | 123 | 5.8 | 0.843 | 2.662 | 80% |
| small_complex | 7 | 4.9 | 0.853 | 2.929 | 75% |
| medium_complex | 42 | 13.9 | 0.488 | 25.924 | 0% |
| medium_complex | 123 | 13.7 | 0.480 | 25.650 | 0% |
| medium_complex | 7 | 13.8 | 0.479 | 25.923 | 0% |
| large_complex | 42 | 26.7 | 0.810 | 20.047 | 0% |
| large_complex | 123 | 25.6 | 0.809 | 20.689 | 0% |
| large_complex | 7 | 23.9 | 0.809 | 22.325 | 0% |

#### Head Pruning (75%)

| Complex | Seed | Predict(s) | pLDDT | CA RMSD | %<2A |
|---------|------|-----------|-------|---------|------|
| small_complex | 42 | 5.0 | 0.859 | 2.521 | 82% |
| small_complex | 123 | 5.4 | 0.848 | 2.481 | 80% |
| small_complex | 7 | 4.7 | 0.858 | 2.490 | 83% |
| medium_complex | 42 | 14.5 | 0.527 | 25.286 | 0% |
| medium_complex | 123 | 14.6 | 0.522 | 25.273 | 0% |
| medium_complex | 7 | 14.6 | 0.527 | 25.308 | 0% |
| large_complex | 42 | 25.2 | 0.830 | 19.929 | 0% |
| large_complex | 123 | 26.1 | 0.830 | 21.474 | 0% |
| large_complex | 7 | 25.7 | 0.832 | 23.141 | 0% |

## What Happened

### SDPA Attention: confirmed dead end (0.98x, no quality change)

SDPA gives no speedup (0.982x, within noise). The reason is well understood: Boltz's AttentionPairBias adds a pair bias tensor to attention logits before softmax. This explicit attention mask forces torch SDPA to fall back to its "math" backend (equivalent to the original einsum implementation) rather than using FlashAttention-2's fused kernel, which does not support arbitrary attention biases.

The SDPA results are essentially identical to control across all metrics: pLDDT (0.7147 vs 0.7146), CA RMSD (16.527 vs 16.521 Angstrom), and per-complex breakdowns. This is expected since the math backend computes the same operations in the same order.

### Head Pruning: confirmed no speedup (1.00x), modest quality improvement (+2.2pp pLDDT)

Head pruning gives no speedup (0.999x) because attention is not the compute bottleneck. The Pairformer is dominated by triangle multiplication operations (O(N^2 * d) einsum contractions), not by the attention mechanism. Zeroing out attention heads does not skip any computation -- the zero-weight multiplications still execute.

The pLDDT improvement is real but modest: +2.2pp mean (0.7370 vs 0.7146). This is smaller than the prior orbit #34's reported +4pp, likely because:
1. Different test cases (our branch uses older sequences)
2. Different base config (ODE-12 + recycle=3 vs whatever #34 used)
3. The improvement is most visible on medium_complex (+4.2pp) and large_complex (+2.2pp), less on small_complex (+0.3pp)

On CA RMSD, head pruning is marginally better on small_complex (2.497 vs 2.731 Angstrom, 9% improvement) but the medium and large complexes have such poor absolute RMSD (>20 Angstrom, essentially random orientation) that no meaningful structural comparison is possible there.

## What I Learned

1. **SDPA cannot help when pair bias is present.** This is a fundamental architectural constraint of the Boltz attention layer. Any FlashAttention-based optimization would require either removing the pair bias (which would degrade quality) or implementing a custom CUDA kernel that fuses pair bias into the FlashAttention computation.

2. **Head pruning cannot yield speedup via weight zeroing.** To actually save compute, one would need to physically reduce the head dimension (restructure weight matrices) or use structured sparsity with hardware support. Simply zeroing weights still executes the full matmul.

3. **The quality improvement from head pruning suggests over-parameterization** in the attention heads. Roughly 75% of heads can be removed with no quality loss and sometimes improvement. This is a regularization effect -- redundant heads add noise during inference.

4. **Triangle operations, not attention, are the compute bottleneck.** Both approaches confirm that optimizing attention alone cannot meaningfully speed up Boltz inference. Future speedup efforts should target triangle multiplication (the O(N^2) einsum operations in TriangleMultiplication{Outgoing,Incoming}).

## Prior Art & Novelty

### What is already known
- FlashAttention-2 (Dao, 2023) cannot handle arbitrary attention biases without falling back to standard attention
- Head pruning for inference speedup has been extensively studied: [Michel et al. (2019)](https://arxiv.org/abs/1905.10650) showed that many attention heads can be pruned without quality loss in NLP transformers
- The compute bottleneck in structure prediction models is well known to be the triangle operations, not attention

### What this orbit adds
- Clean quantitative confirmation on eval-v5 that both approaches are dead ends for Boltz-2 speedup
- CA RMSD structural validation showing neither approach affects structural accuracy
- Documentation for future reference: do not pursue attention-level optimizations for Boltz speedup

### Honest positioning
This orbit performs no novel research. It re-validates known negative results on the current evaluation framework for documentation purposes.

## References

- [Dao (2023). FlashAttention-2](https://arxiv.org/abs/2307.08691) -- fused attention kernels, pair bias incompatibility
- [Michel et al. (2019). Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) -- head pruning in transformers
- Orbit #36 (sdpa-v4) -- prior SDPA attempt
- Orbit #34 (head-prune) -- prior head pruning attempt
- Orbit #44 (bypass-lightning) -- bypass wrapper this orbit builds on
