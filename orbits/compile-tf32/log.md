---
strategy: compile-tf32
type: experiment
status: in-progress
eval_version: eval-v1
metric: null
issue: 4
parents: []
root_rationale: "First exploration -- measure speedup from TF32 matmul precision and torch.compile, both already supported by the wrapper"
---

# Compile + TF32: Free Lunch Optimizations

**Hypothesis**: Enable TF32 matmul precision and torch.compile on the score model to speed up Boltz-2 inference.

**Metric**: speedup_at_iso_quality (MAXIMIZE) = T_baseline / T_optimized, subject to mean_pLDDT regression <= 2pp.

**Status**: in_progress

## Key Findings (Investigation)

### Compile flags during inference -- critical discovery

The Boltz2 model has compile flags in its constructor, but there is an important nuance:

1. **compile_pairformer** and **compile_msa**: The forward pass explicitly reverts to the uncompiled module when `not self.training` (i.e., during inference) by accessing `._orig_mod`. These flags have **zero effect** during inference. See `boltz2.py` lines 468-481.

2. **compile_structure** (score model): No revert logic in the diffusion module (`diffusionv2.py` lines 205-208). The score model stays compiled during inference. **Should work.**

3. **compile_confidence**: No revert logic. **Should work.**

### Wrapper gap

The `boltz_wrapper.py` captures compile flags but stores them in a local `_compile_flags` dict that is never used. The `boltz.main.predict()` function calls `model_cls.load_from_checkpoint()` without passing compile flags. So even if the wrapper properly intercepted these, the flags would not reach the model constructor.

**For compile to work, the wrapper needs to monkey-patch the `load_from_checkpoint` call** to inject compile kwargs.

### TF32 matmul precision

The wrapper correctly sets `torch.set_float32_matmul_precision()` before importing boltz, so TF32 should work as expected.

## Experiment Log

| # | Config | mean_wall_time_s | mean_pLDDT | speedup | pLDDT_delta_pp | quality_gate |
|---|--------|-----------------|------------|---------|----------------|--------------|
| baseline | 200 steps, highest precision | 70.37 | 0.7107 | 1.00x | 0.0 | PASS |

## Results

(pending)
