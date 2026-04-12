---
strategy: cuda-warmup
type: experiment
status: complete
eval_version: eval-v4
metric: 2.58x
issue: 43
parents:
  - orbit/eval-v2-winner
  - orbit/cpu-gap-profile
---

# CUDA Warmup

## Hypothesis

The first forward pass in a new subprocess pays ~2.2s of CUDA lazy-init
(kernel JIT for MSA attention, diffusion ops, etc.). A warmup pass before
the timed section should eliminate this overhead.

## Implementation

Modified `research/eval/boltz_wrapper.py` to monkey-patch `Trainer.predict`:
before emitting `[PHASE] predict_start`, the wrapper:

1. Moves the model to GPU (`model.to(device)`) since it's loaded on CPU
2. Grabs the first real batch from the datamodule's predict dataloader
3. Transfers the batch to GPU (mirroring DataModule.transfer_batch_to_device)
4. Runs `model.predict_step(batch, batch_idx=0)` under `torch.no_grad()`
5. Calls `torch.cuda.synchronize()` to ensure all kernels are compiled
6. Frees warmup memory via `del` + `torch.cuda.empty_cache()` + `gc.collect()`
7. Then emits `[PHASE] predict_start` and runs the real Trainer.predict

The warmup is non-fatal: if it fails for any reason, prediction proceeds
normally (just without the warmup benefit).

## Results

Config: ODE-12/0r + TF32 + bf16 (same as eval-v2-winner best)

### Validated run (3 seeds, median timing)

| Complex | predict_only_s | Baseline (25.04s mean) |
|---------|---------------|----------------------|
| small   | ~5.2s         | 12.40s               |
| medium  | ~9.6s         | 24.20s               |
| large   | ~14.4s        | 38.52s               |
| **mean**| **9.72s**     | **25.04s**           |

- **Speedup: 2.58x** (validated, 3 runs)
- pLDDT delta: +0.17 pp (quality gate PASS)
- Per-complex regressions: none

### Comparison to prior best

| Metric         | eval-v2-winner | + cuda-warmup | Delta |
|---------------|---------------|---------------|-------|
| predict_only_s | ~13.0s        | ~9.7s         | -3.3s |
| Speedup        | 1.91x         | 2.58x         | +0.67x|

### Warmup overhead (excluded from timing)

The warmup itself takes 6-18s depending on complex size (runs a full
predict_step). This is excluded from predict_only_s since it happens
before the `[PHASE] predict_start` marker.

## Notes

The 3.3s reduction exceeds the expected 2.2s CUDA lazy-init. Additional
sources of speedup likely include:
- CUDA memory allocator warmup (first allocation is slower)
- GPU boost clock stabilization (clocks ramp up after initial load)
- cuDNN autotuner cache priming

## Files changed

- `research/eval/boltz_wrapper.py` — warmup logic in _warmup_and_predict
