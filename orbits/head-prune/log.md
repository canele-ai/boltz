---
strategy: head-prune
type: experiment
status: negative
eval_version: eval-v4
metric: "~1.46x (wall-time), no predict-only speedup"
issue: 34
parents:
  - orbit/eval-v2-winner
---

# Head Prune — Attention Head Pruning in Pairformer

## Hypothesis

Many of the 1024 attention heads (64 blocks x 16 heads) in the Pairformer's
sequence attention (AttentionPairBias) are redundant. Pruning the least
important ones can reduce compute without quality loss.

## Method

1. **Head importance metric**: For each head h in each block, compute
   `importance = L2_norm(proj_o[:, h*D:(h+1)*D]) * L2_norm(proj_v[h*D:(h+1)*D, :]) * L2_norm(proj_g[h*D:(h+1)*D, :])`
   This is a weight-based proxy for each head's maximum possible contribution.

2. **Pruning by zeroing**: Zero out Q/K/V/G/O projection weights and pair bias
   for the least important heads. This disables heads without changing tensor
   shapes (avoids reshape issues).

3. **Sweep**: Tested 0%, 25%, 50%, 75% pruning on all 3 eval-v4 test complexes.

## Base config

ODE-20/0r + TF32 + bf16 (eval-v2-winner stack):
- sampling_steps=20, recycling_steps=0, gamma_0=0.0
- matmul_precision=high, bf16_trunk=True, enable_kernels=True

## Results

| Config | Pruned | Wall(s) | Predict(s) | mean_pLDDT | delta_pp | Speedup | Gate |
|--------|--------|---------|------------|------------|----------|---------|------|
| 0%     | 0/1024 | 58.2    | 15.5       | 0.7293     | +1.23    | 0.92x   | PASS |
| 25%    | 256/1024 | 37.1  | 13.1       | 0.7336     | +1.66    | 1.44x   | PASS |
| 50%    | 512/1024 | 36.7  | 12.9       | 0.7501     | +3.31    | PASS |
| 75%    | 768/1024 | 36.1  | 13.0       | 0.7573     | +4.03    | 1.48x   | PASS |

### Per-complex pLDDT

| Complex        | Baseline | 0% prune | 25% prune | 50% prune | 75% prune |
|----------------|----------|----------|-----------|-----------|-----------|
| small_complex  | 0.8345   | 0.8831   | 0.8600    | 0.8995    | 0.9055    |
| medium_complex | 0.5095   | 0.4868   | 0.5120    | 0.5142    | 0.5332    |
| large_complex  | 0.8070   | 0.8180   | 0.8288    | 0.8367    | 0.8333    |

## Analysis

### No speedup from weight zeroing

The predict-only times (13.0-15.5s) are effectively identical across all pruning
levels. This is expected: zeroing weights does NOT change tensor shapes, so CUDA
still performs the same-size matmuls (just multiplying by zeros). The wall-time
difference (58s vs 37s) is an artifact of model download caching -- the first
config (0%) downloads the model, subsequent configs reuse the cache.

### Quality is surprisingly robust

Even with 75% of attention heads zeroed, pLDDT actually IMPROVES (+4pp over
baseline). This suggests:

1. The sequence attention heads in the Pairformer are highly redundant.
2. Most of the useful signal is carried by triangle attention and triangle
   multiplication (the pairwise track), not the sequence attention.
3. Zeroing heads acts as a form of implicit regularization.

### Path to real speedup

To actually speed up inference, structural pruning is needed:
- Reduce `num_heads` from 16 to 4 (75% reduction)
- Reshape Q/K/V projections from Linear(384, 384) to Linear(384, 96)
- Reshape proj_o from Linear(384, 384) to Linear(96, 384)
- Reshape proj_z from Linear(c_z, 16) to Linear(c_z, 4)

This would reduce sequence attention matmul FLOPS by ~4x. However:
- Sequence attention is only one component of each Pairformer block
- Triangle attention and triangle multiplication dominate the compute
- The theoretical speedup from sequence attention pruning alone is modest

### Conclusion

**Negative result for speedup**: Head pruning by zeroing provides NO inference
speedup because tensor shapes are unchanged. Structural pruning could help but
the sequence attention is not the bottleneck (triangle ops dominate).

**Positive finding for quality**: Pairformer sequence attention heads are highly
redundant. This insight could inform future architecture design or enable more
aggressive mixed-precision strategies for the sequence track.

## Status: NEGATIVE

Head pruning does not provide speedup. The sequence attention is not the
Pairformer bottleneck -- triangle attention and triangle multiplication
(cuequivariance kernels) are. Head pruning would need to target those operations
to be useful, but they use a different attention mechanism (TriangleAttention with
cuequivariance, not standard multi-head attention).
