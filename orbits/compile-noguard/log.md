---
strategy: compile-noguard
type: experiment
status: complete
eval_version: eval-v2
metric: 0.88
issue: 16
parents:
  - orbit/eval-v2-winner
---

# Aggressive torch.compile Without Inference Guards

## Glossary

- **pLDDT**: predicted Local Distance Difference Test -- Boltz confidence proxy for structural accuracy (0--1 scale)
- **pp**: percentage points (absolute difference in pLDDT scaled to 0--100)
- **ODE**: Ordinary Differential Equation -- deterministic sampler with gamma_0=0 (no noise injection)
- **TF32**: TensorFloat-32 -- 19-bit format on Ada Lovelace/Ampere+ GPUs
- **bf16 trunk**: removing `.float()` upcast in triangular_mult.py so einsum stays in bf16
- **torch.compile**: PyTorch's JIT compilation via TorchDynamo + TorchInductor
- **_orig_mod**: the original uncompiled module stored by `torch.compile`; the Boltz2 forward method falls back to this during inference
- **MSA**: Multiple Sequence Alignment -- evolutionary sequence search that dominates end-to-end latency

## Results

**Negative result: torch.compile makes Boltz-2 inference 2.5x SLOWER, not faster.**

The hypothesis was that Boltz2's `_orig_mod` fallback was the reason prior compile attempts failed -- the model explicitly undoes compilation during inference. Fixing this fallback (so compiled modules actually run) reveals that the real bottleneck is compilation overhead: torch.compile spends minutes tracing and optimizing the large DiffusionModule score model, which is never amortized in a typical 3-complex, 20-step evaluation.

### Experiment 1: No compile (parent config baseline)

ODE-20/0r + TF32 + bf16, no torch.compile. 3 seeds (42, 123, 7), eval-v2 harness.

| Seed | Small(s) | Medium(s) | Large(s) | Mean(s) | pLDDT | Speedup | Gate |
|------|----------|-----------|----------|---------|-------|---------|------|
| 42   | 91.5     | 44.5      | 49.8     | 61.9    | 0.7293 | 0.87x  | PASS |
| 123  | 96.2     | 42.0      | 47.6     | 61.9    | 0.7193 | 0.87x  | PASS |
| 7    | 87.8     | 41.5      | 47.0     | 58.8    | 0.7371 | 0.91x  | PASS |
| **Mean** | **91.8** | **42.7** | **48.1** | **60.9** | **0.7286** | **0.88x +/- 0.03** | **PASS** |

Note: Speedup < 1.0x because the small_complex includes cold-start overhead (MSA download + model loading). Excluding cold start, medium and large complexes average ~45s, consistent with parent orbit.

### Experiment 2: Compile pairformer + structure (default mode)

ODE-20/0r + TF32 + bf16 + torch.compile(pairformer, structure, mode="default"). 3 seeds.

| Seed | Small(s) | Medium(s) | Large(s) | Mean(s) | pLDDT | Speedup | Gate |
|------|----------|-----------|----------|---------|-------|---------|------|
| 42   | 174.8    | 121.7     | 134.3    | 143.6   | 0.7289 | 0.37x  | PASS |
| 123  | 168.6    | 110.0     | 116.9    | 131.8   | 0.7206 | 0.41x  | PASS |
| 7    | 192.7    | 175.7     | 201.7    | 190.0   | 0.7378 | 0.28x  | PASS |
| **Mean** | **178.7** | **135.8** | **151.0** | **155.1** | **0.7291** | **0.35x +/- 0.06** | **PASS** |

**Compile makes inference 2.5x slower** (155.1s vs 60.9s mean). Quality is preserved (pLDDT identical within noise), confirming the compilation does not break correctness -- it is purely an overhead problem.

### Overhead breakdown

| Complex | No compile | With compile | Overhead |
|---------|-----------|-------------|----------|
| Small   | 91.8s     | 178.7s      | +86.9s (+95%) |
| Medium  | 42.7s     | 135.8s      | +93.1s (+218%) |
| Large   | 48.1s     | 151.0s      | +102.9s (+214%) |

The overhead is roughly constant (~90-100s) per complex, consistent with per-complex compilation cost. The first complex (small) has a smaller relative overhead because MSA cold-start is already ~50s; for medium and large complexes where actual GPU time is ~40-50s, compilation adds 2x the GPU time.

## Approach

The Boltz2 model has an anti-pattern where compiled modules are explicitly bypassed during inference:

```python
# boltz2.py:478-481
if self.is_pairformer_compiled and not self.training:
    pairformer_module = self.pairformer_module._orig_mod  # UNDOES compilation!
else:
    pairformer_module = self.pairformer_module
```

This pattern repeats for template_module, msa_module, and pairformer_module. The fix: apply torch.compile AFTER model loading without setting `is_*_compiled` flags. The compiled modules are then used directly because the guard condition is always False.

The wrapper monkey-patches `Boltz2.eval()` to intercept the first call after `load_from_checkpoint()`, applying torch.compile to the desired submodules.

## What I Learned

1. **The _orig_mod fallback was real but irrelevant.** The hypothesis was correct that Boltz2 explicitly undoes compilation during inference. The fix works -- compiled modules do run. But compilation overhead overwhelms any per-forward-pass speedup.

2. **torch.compile overhead is fatal for few-shot inference.** The DiffusionModule score model is large (24-layer token transformer with atom encoder/decoder). Tracing and optimizing this graph takes ~90-100s. With only 20 diffusion steps x 3 complexes = 60 forward passes, the overhead never amortizes. torch.compile is designed for training loops with thousands of iterations, not inference with tens of forward passes.

3. **Each complex recompiles.** Despite running 3 complexes sequentially in a single subprocess, the overhead is ~90-100s per complex (not just the first). This suggests TorchDynamo is retracing on different input shapes (each complex has different numbers of residues).

4. **The `@torch.compiler.disable` decorators on cuequivariance kernels create graph breaks** that force separate compiled subgraphs, reducing fusion opportunities and increasing tracing overhead.

5. **The parent orbit was right, for the wrong reason.** The eval-v2-winner log stated "torch.compile does not help at 20 steps." The reasoning was partially wrong (they thought compilation was running but not helping, when actually it was never running due to `_orig_mod`). But the conclusion was correct: compile is not viable at 20 steps.

## What Would Make torch.compile Work

For torch.compile to provide a speedup for Boltz-2 inference:
- **Persistent compilation cache** (`torch._inductor.config.force_disable_caches = False` + disk cache)
- **Fixed input shapes** across complexes (padding to maximum size) to avoid recompilation
- **Many more forward passes** per compiled graph (200+ steps or batch many complexes)
- **Remove `@torch.compiler.disable`** from cuequivariance kernels (upstream change)
- **`reduce-overhead` mode** with CUDA graphs (requires static shapes)

## Prior Art & Novelty

### What is already known
- torch.compile overhead for inference is a well-known limitation ([PyTorch docs](https://pytorch.org/docs/stable/generated/torch.compile.html))
- The prior orbit/compile-tf32 (#4) reported 0.97x with torch.compile on eval-v1
- The parent orbit/eval-v2-winner (#13) noted "torch.compile does not help at 20 steps"

### What this orbit adds
- Identified and fixed the `_orig_mod` fallback that prevented torch.compile from running during inference
- Demonstrated that even with the fix, torch.compile makes inference 2.5x slower due to compilation overhead (~90-100s per complex)
- Showed that compilation cost is per-complex (not amortized), suggesting shape-dependent retracing
- Closed the question definitively: torch.compile is not a viable speedup path for Boltz-2 with current architecture

### Honest positioning
This is a negative result that closes an investigation direction. The _orig_mod fallback fix is novel code, but the conclusion -- that compilation overhead dominates for few-shot inference -- is well-known. The contribution is empirical confirmation specific to Boltz-2 and quantification of the overhead (~90-100s per complex).

## References

- [PyTorch torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html)
- Parent orbit: orbit/eval-v2-winner (#13) -- baseline stacked optimizations
- Prior orbit: orbit/compile-tf32 (#4) -- earlier compile attempt on eval-v1
- [Karras et al. (2022)](https://arxiv.org/abs/2206.00364) -- EDM/Karras diffusion sampler
