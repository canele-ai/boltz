---
strategy: tensorrt-export
type: experiment
status: negative
eval_version: eval-v3
metric: null
issue: 31
parents:
  - orbit/eval-v2-winner
---

# TensorRT Export for Optimized Inference

## Hypothesis

Export Boltz-2's compute-heavy modules (Pairformer 1.1s/26%, DiffusionTransformer
0.4s/10%) to TensorRT, which uses a fundamentally different optimization strategy
from torch.compile/inductor. TensorRT has its own kernel library, performs layer
fusion, constant folding, precision calibration, and pre-allocates memory.

## Negative Result

TensorRT export is **not feasible** for the Pairformer module. Four separate
export paths were tested, all failed:

### Probe Results (L40S, torch 2.8.0+cu128, TensorRT 10.12.0, N=200)

| Method | Status | Error |
|--------|--------|-------|
| torch.export | FAILED | `chunk_layer` in TriangleAttention uses `_fetch_dims` with unsupported types |
| torch.compile(backend='torch_tensorrt') | FAILED | Bool tensor from dropout mask incompatible with TRT elementwise ops |
| ONNX export + ORT | FAILED | ONNX exports but ORT can't load: invalid Transpose perm in tri_att_start |
| torch.jit.trace | SUCCESS (trace only) | Cannot connect to TRT compilation pipeline |

### Transition Block (Pure PyTorch, No cuequivariance)

The Transition block (LayerNorm + 2x Linear + SiLU) **does** compile to TRT:
- Baseline: 1.3ms/layer, TRT: 0.7ms/layer (1.79x speedup)
- But Transition is only ~2% of total Pairformer compute
- 48 layers x 0.6ms savings = 29ms total -- negligible vs 47.5s baseline

### Root Causes

1. **TriangleAttention chunk_layer**: Uses `_fetch_dims()` which recursively
   inspects tensor shapes at trace time -- incompatible with `torch.export`
   and produces invalid ONNX transpose attributes.

2. **cuequivariance CUDA kernels**: Decorated with `@torch.compiler.disable`,
   making them opaque to all tracing/export frameworks. These kernels (for
   TriangleMultiplication) are already highly optimized CUDA code -- TRT would
   not improve over them even if export were possible.

3. **Dropout mask control flow**: The `get_dropout_mask()` produces Bool tensors
   via `ge(rand, threshold)` which TRT's elementwise layer doesn't support.

4. **torch_tensorrt pulls torch 2.8.0**: The `torch_tensorrt` pip package
   requires torch >= 2.8.0, creating version conflicts with the boltz==2.2.1
   environment (tested on torch 2.6.0).

### DiffusionTransformer Not Worth Pursuing

Even if the DiffusionTransformer (simpler architecture) could be exported:
- Only 0.4s at 12 ODE steps
- A 2x speedup would save 0.2s out of 47.5s baseline
- The 12-step ODE solver already reduced this from 3.3s (200 steps)

## Conclusion

TensorRT export is a dead end for Boltz-2 inference optimization. The model's
architecture is fundamentally hostile to export-based optimization:

- TriangleAttention uses dynamic chunking with runtime shape inspection
- cuequivariance CUDA kernels are already optimized and opaque to tracers
- The biggest wins (ODE steps, TF32, bf16) are already captured by eval-v2-winner

The current best of 1.47x (ODE-12, recycling=0, TF32, bf16) likely represents
close to the ceiling for optimizations that don't modify the model architecture.
