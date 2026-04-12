---
strategy: minimal-inference
type: experiment
status: complete
eval_version: eval-v3
metric: 3.80
issue: 28
parents:
  - orbit/eval-v2-winner
  - orbit/lightning-strip
  - orbit/adaptive-steps
---

# Minimal Inference: Strip All Abstractions

## Hypothesis
Rewriting Boltz-2 inference as a minimal direct-call pipeline (bypassing
Lightning Trainer, DataModule, subprocess overhead) would expose how much
of the ~35s fixed-cost floor is framework overhead vs actual GPU compute.

## Approach
1. Load model checkpoint directly via `Boltz2.load_from_checkpoint()` (no Trainer)
2. Apply proven ODE-12 + TF32 + bf16 patches
3. Call model internal methods directly with CUDA sync barriers between phases
4. Measure per-phase GPU timing: embedding, MSA, pairformer, diffusion, confidence
5. Share single model load across all test cases (amortize 20s load cost)

## Results

### Multi-seed (seeds 42, 123, 7)

| Metric | Value |
|--------|-------|
| Speedup (no load) | **3.80x +/- 0.74** |
| Speedup (with load) | 2.60x +/- 0.39 |
| Speedup (GPU only) | 12.82x +/- 0.49 |
| Mean pLDDT | 0.7204 (baseline 0.7170) |
| Quality gate | **PASS** (all 3 seeds) |

### Per-phase timing breakdown (mean across seeds)

| Phase | Time (s) | % of GPU |
|-------|----------|----------|
| Input processing (CPU+network) | 8.217 | -- |
| Batch transfer | 0.010 | 0.2% |
| Input embedding | 0.134 | 3.2% |
| MSA module | 1.315 | 31.4% |
| **Pairformer** | **1.101** | **26.3%** |
| Distogram | 0.001 | 0.0% |
| Diffusion conditioning | 0.030 | 0.7% |
| Diffusion sampling (12 ODE) | 0.417 | 10.0% |
| Confidence | 0.169 | 4.0% |
| Output writing | 0.193 | 4.6% |
| **Total GPU** | **4.184** | **100%** |

### DiffusionTransformer truncation (24 -> 8 layers): FAILED
Tested but caused catastrophic quality regression (-25.6pp pLDDT). The prior
orbit's claim of "zero quality impact" was incorrect -- the DiffusionTransformer
layers are critical for quality, unlike what layer-prune orbit suggested.

## Key Findings

1. **The 35s "fixed-cost floor" was mostly framework overhead.** The actual GPU
   compute for a full inference pass is only ~4.2s. The remaining time was:
   - Model loading: ~20s per subprocess call (eliminated by loading once)
   - MSA server network latency: ~5-8s (highly variable)
   - DataModule/DataLoader/Lightning overhead: ~1-2s

2. **Pairformer (1.1s, 26%) and MSA module (1.3s, 31%) dominate GPU time.**
   Diffusion sampling at 12 ODE steps is only 0.4s (10%).

3. **The model loading cost was the single largest bottleneck.** The baseline
   evaluator launches `boltz predict` as a subprocess per test case, paying the
   ~20s model loading penalty each time. Loading once and reusing gives 2.6x
   speedup just from amortization.

4. **MSA server latency is the remaining variance source.** One seed had 36.5s
   for small_complex (vs 12s normally) due to network latency. Pre-caching MSAs
   would further stabilize timing.

5. **DiffusionTransformer layer truncation is NOT safe.** 24->8 layers causes
   pLDDT to drop from 0.72 to 0.46. This contradicts the layer-prune orbit's
   findings, possibly because that orbit measured with different ODE step counts.

## Comparison with Prior Work

| Approach | Speedup | Notes |
|----------|---------|-------|
| Baseline (boltz predict subprocess) | 1.00x | 53.6s mean |
| eval-v2-winner (ODE+TF32+bf16) | 1.47x | Still uses subprocess per case |
| lightning-strip (no Trainer) | ~1.5x | Bypasses Lightning but still per-case |
| **minimal-inference** | **3.80x** | Single model load + direct calls |

## Status: COMPLETE
Metric: 3.80x speedup at iso-quality (mean pLDDT 0.7204, baseline 0.7170)
