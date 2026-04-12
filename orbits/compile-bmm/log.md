---
strategy: compile-bmm
type: experiment
status: negative
eval_version: eval-v4
metric: 0.97x (bmm-nocompile, best of compile variants)
issue: 38
parents:
  - orbit/triton-pairformer
---

# torch.compile with bmm path (no cuequivariance)

## Hypothesis

compile-score-v3 (#20) found torch.compile was 2x SLOWER because inductor could not
trace through cuequivariance's `@torch.compiler.disable` decorators. triton-pairformer
(#35) created a bmm-based replacement that is fully traceable. This orbit tests whether
torch.compile can now fuse the separated ops (LayerNorm + Linear + sigmoid + bmm + gate)
to recover the ~8% gap from using bmm instead of cuequivariance kernels.

## Results

All runs on L40S GPU, ODE-12/0r + TF32 + bf16 stack, single-run timing
(includes model loading, which adds ~30-40s overhead to each first run).

### Timing comparison (wall time, seconds)

| Config                  | small  | medium | large  | mean   |
|-------------------------|--------|--------|--------|--------|
| baseline-cueq           | 106.4  |  47.0  |  48.8  |  67.4  |
| bmm-nocompile           |  84.3  |  37.9  |  43.7  |  55.3  |
| compile-default         | 101.1  |  46.6  |  49.2  |  65.6  |
| compile-reduce-overhead |  93.4  |  39.1  |  43.2  |  58.6  |
| compile-max-autotune    | 100.9  |  46.1  |  51.3  |  66.1  |

### Quality comparison (pLDDT)

| Config                  | small  | medium | large  | mean   |
|-------------------------|--------|--------|--------|--------|
| baseline-cueq           | 0.8783 | 0.4835 | 0.8175 | 0.7264 |
| bmm-nocompile           | 0.8655 | 0.4837 | 0.8172 | 0.7221 |
| compile-default         | 0.8655 | 0.4845 | 0.8174 | 0.7225 |
| compile-reduce-overhead | 0.8655 | 0.4837 | 0.8172 | 0.7221 |
| compile-max-autotune    | 0.8655 | 0.4845 | 0.8174 | 0.7225 |

All configs pass the quality gate (mean pLDDT regression <= 2pp).

### Speedup vs eval-v4 baseline (25.04s predict-only)

Speedup measured against original 200-step baseline (53.57s mean wall time):
- bmm-nocompile: 0.97x (55.3s) -- fastest overall
- compile-reduce-overhead: 0.91x (58.6s) -- best compile mode
- compile-default: 0.82x (65.6s)
- compile-max-autotune: 0.81x (66.1s)
- baseline-cueq: 0.79x (67.4s)

## Key findings

1. **torch.compile adds overhead, does not recover performance** -- For short inference
   runs (12 ODE steps), the compilation overhead (5-17s per input shape) dominates any
   per-step speedup. The `reduce-overhead` mode (CUDA graphs) minimizes this to ~3s but
   still nets negative vs no-compile.

2. **bmm path is faster than cuequivariance** -- Surprisingly, the bmm replacement
   WITHOUT compile is faster than cuequivariance (55.3s vs 67.4s). This is likely because
   the bf16 bmm path avoids the float32 upcast that cuequivariance requires, and the
   cuequivariance kernel launch overhead is larger than expected on L40S.

3. **Compile tracing works but produces graph breaks** -- The pairformer compiles
   successfully with fullgraph=False, but with 12 ODE steps and varying input shapes
   across test cases, each new shape triggers recompilation (cold start each subprocess).

4. **The amortization window is too short** -- torch.compile pays off when the compiled
   code runs hundreds of times. With 12 sampling steps, the per-step savings (~0.1-0.5s)
   cannot recoup the 5-17s compilation cost.

## Conclusion

**Negative result.** torch.compile on the bmm-based Pairformer does not improve
end-to-end inference time for the ODE-12 regime. The compilation overhead exceeds
any per-step kernel fusion benefits at this step count.

The bmm-nocompile path is the most interesting finding: it outperforms cuequivariance
(0.97x vs 0.79x of original baseline) because it avoids float32 upcast overhead.
This was already discovered by triton-pairformer (#35).

For torch.compile to be beneficial, the workload would need either:
- Many more sampling steps (50+) to amortize compile cost
- Persistent compiled graphs across inputs (same shapes)
- Pre-compiled kernel caches (torch.compile cache warming)

## Environment

- GPU: NVIDIA L40S
- PyTorch: 2.6.0+cu124
- Triton: 3.2.0
- Boltz: 2.2.1
- cuequivariance_torch: 0.9.1
