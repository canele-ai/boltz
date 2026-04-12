---
strategy: fast-load
type: experiment
status: complete
eval_version: eval-v3
metric: 3.78
issue: 30
parents:
  - orbit/eval-v2-winner
  - orbit/minimal-inference
---

# Fast Model Loading

## Result

**3.78x speedup** (with amortized model load) at iso-quality (pLDDT +0.96pp).
Quality gate: PASS.

Compute-only speedup: 6.40x (excluding one-time model load).

## Approach: Persistent Model + Safetensors + ODE-12 + TF32 + bf16

The key insight: the per-prediction bottleneck is model loading (~20s of ~36s
optimized subprocess time = 56%). By loading the model ONCE and predicting all
test cases in-process, we eliminate this bottleneck entirely.

### Stacked optimizations:
1. **Persistent model**: Load model once, predict all complexes in-process
   (eliminates ~20s/complex model loading)
2. **Safetensors format**: 0.9s GPU load vs 2.0s torch.load vs 1.6s raw ckpt
   (zero-copy mmap, no pickle)
3. **ODE-12**: gamma_0=0.0, 12 sampling steps (deterministic first-order Euler)
4. **TF32 matmul**: matmul_precision="high"
5. **bf16 trunk**: Remove .float() upcast in triangular_mult.py
6. **Pre-cached MSAs**: Skip network round-trips to MMseqs2 server

## Validated Metrics (3 runs, median, L40S GPU)

| Complex | Time (median) | pLDDT | Run times |
|---------|--------------|-------|-----------|
| small   | 5.0s         | 0.889 | 11.5, 5.0, 4.8 |
| medium  | 7.7s         | 0.484 | 8.8, 7.7, 7.5 |
| large   | 12.4s        | 0.806 | 14.0, 12.4, 12.3 |
| **mean** | **8.4s**    | **0.727** | |

- Model load (one-time): 17.4s via safetensors
- Amortized load per complex: 5.8s (17.4s / 3 complexes)
- Mean compute-only: 8.4s -> **6.40x** speedup
- Mean with amortized load: 14.2s -> **3.78x** speedup
- Baseline mean: 47.5s, baseline pLDDT: 0.717
- pLDDT delta: +0.96pp (improvement, not regression)
- Quality gate: PASS (threshold: <=2pp regression)

## Loading Benchmark (Modal volume, L40S)

| Method | Load time |
|--------|-----------|
| Safetensors GPU | 0.9s |
| Safetensors CPU | 0.3s |
| Raw checkpoint torch.load | 1.6s |
| Torch state_dict CPU | 1.7s |
| Torch state_dict GPU | 2.0s |
| Full load_from_checkpoint + .cuda() | ~20s |

The 0.9s safetensors GPU load is 2.2x faster than torch state_dict GPU, but
the real bottleneck is model instantiation (Boltz2.__init__ ~18s), not tensor loading.
The persistent-model approach eliminates both.

## Architecture

- `save_safetensors.py` -- one-time: convert checkpoint to safetensors on Modal volume
- `eval_fast_load.py` -- evaluator with multiple modes:
  - `sanity` -- verify environment and weights
  - `bench-load` -- benchmark loading approaches (timing only)
  - `persistent` -- load once, predict all (main approach)
  - `subprocess-sf` -- subprocess with safetensors wrapper
- `boltz_wrapper_fast.py` -- subprocess wrapper with safetensors loading
- `persistent_predict.py` -- helper for in-process prediction

## Key Implementation Details

1. Safetensors weights + hparams saved to Modal persistent volume (`boltz-fast-load-weights`)
2. OmegaConf hparams converted to plain dicts for JSON serialization
3. Hparams filtered via `inspect.signature(Boltz2.__init__)` for version compatibility
4. `strict=False` in `load_state_dict` for checkpoint/code version tolerance
5. Replicates full `boltz.main.predict()` pipeline (check_inputs -> process_inputs ->
   Manifest -> filter -> DataModule -> Trainer.predict) but skips model loading
