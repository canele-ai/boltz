---
strategy: minimal-inference-v2
type: experiment
status: complete
eval_version: eval-v3
metric: 6.73
issue: 29
parents:
  - orbit/minimal-inference
---

# Minimal Inference v2: With Proper MSA Caching

## Bug Fixed

The parent orbit (minimal-inference, #28) reported 3.80x speedup, but had a
bug: it called `process_inputs(..., use_msa_server=True)` on every invocation,
hitting the ColabFold MMseqs2 server and adding ~5-8s of network latency per
complex. The eval-v3 harness is supposed to use pre-cached MSAs from a Modal
volume to eliminate this confound.

### Fix applied

1. Mounted Modal volume `boltz-msa-cache-v3` at `/msa_cache` in the Modal
   function decorator.
2. Before calling `process_inputs`, inject cached MSA paths into the input
   YAML using `_inject_cached_msas()` (same approach as the eval-v3 evaluator).
3. When cached MSAs are found, call `process_inputs(..., use_msa_server=False)`.
   Falls back to the server with a warning if cache is missing.

## Results (3 seeds: 42, 123, 7)

### Speedup vs eval-v3 baseline (47.55s)

| Metric              | Value              |
|---------------------|--------------------|
| Speedup (no load)   | 6.73x +/- 0.24    |
| Speedup (with load) | 3.54x +/- 0.15    |
| Speedup (GPU only)  | 11.30x +/- 0.27   |
| Mean pLDDT          | 0.4608             |
| Quality gate        | FAIL (-25.6pp)     |

### Mean phase breakdown (across seeds)

| Phase                  | Time (s)         |
|------------------------|------------------|
| Input processing       | 0.480 +/- 0.019  |
| Batch transfer         | 0.010 +/- 0.000  |
| Input embedding        | 0.143 +/- 0.004  |
| MSA module             | 1.338 +/- 0.040  |
| Pairformer             | 1.148 +/- 0.004  |
| Distogram              | 0.001 +/- 0.000  |
| Diffusion conditioning | 0.031 +/- 0.000  |
| Diffusion sampling     | 0.320 +/- 0.010  |
| Confidence             | 0.174 +/- 0.001  |
| Output writing         | 0.202 +/- 0.008  |
| **Total GPU**          | **4.209 +/- 0.100** |

### Per-seed results

| Seed | No-load (speedup) | With-load (speedup) | GPU time | pLDDT  |
|------|--------------------|---------------------|----------|--------|
| 42   | 7.2s (6.62x)       | 13.7s (3.48x)       | 4.2s     | 0.4610 |
| 123  | 6.7s (7.07x)       | 12.7s (3.75x)       | 4.1s     | 0.4576 |
| 7    | 7.3s (6.51x)       | 13.9s (3.41x)       | 4.3s     | 0.4638 |

### Per-complex breakdown

| Complex        | Seed 42 (total/gpu) | Seed 123 (total/gpu) | Seed 7 (total/gpu) |
|----------------|---------------------|----------------------|--------------------|
| small_complex  | 7.5s / 6.1s         | 7.0s / 5.9s          | 7.6s / 6.4s        |
| medium_complex | 5.3s / 3.0s         | 5.0s / 2.9s          | 5.7s / 3.1s        |
| large_complex  | 8.8s / 3.5s         | 8.1s / 3.5s          | 8.6s / 3.5s        |

## Key finding

With proper MSA caching, input processing drops from ~8.2s (parent orbit) to
~0.48s, confirming the MSA server latency was the dominant confound. The
no-load speedup jumps from 3.80x to 6.73x vs the eval-v3 baseline.

The quality gate still fails (mean pLDDT 0.46 vs baseline 0.72, -25.6pp
regression). This is inherited from the parent orbit's aggressive DiffusionTransformer
layer truncation (24 -> 8 layers), which was already known to degrade quality.
The MSA caching fix itself does not affect model quality -- it only eliminates
network latency from the timing measurement.
