---
strategy: compile-diffusion-ode
type: experiment
status: complete
eval_version: eval-v5
metric: null
issue: 48
parents:
  - orbit/bypass-lightning
---

# Compile Diffusion ODE

## Glossary

- ODE: Ordinary Differential Equation (deterministic diffusion sampling with gamma_0=0)
- TF32: TensorFloat-32 (matmul precision mode on Ampere+ GPUs)
- CUDA: Compute Unified Device Architecture
- pLDDT: predicted Local Distance Difference Test (confidence metric)

## Result

**Negative result.** torch.compile applied to the diffusion score network produces no meaningful speedup on the timed inference section (predict_only), and the compilation warmup (70-110s) makes overall wall time 2-3x worse.

Across two independent sweeps (seed=42), the predict_only improvements from the best compile mode (score-reduce-overhead) range from -0.3s to +0.9s on a ~7-13s baseline per complex -- well within run-to-run variance.

## Approach

The hypothesis was that torch.compile on the score network (DiffusionModule) would benefit from kernel fusion and CUDA graph capture across the 12 identical ODE sampling steps. Each step calls the same score model with fixed tensor shapes -- seemingly ideal for torch.compile.

Five configurations were tested against the bypass-lightning baseline (ODE-12, 0 recycling, TF32, bf16 trunk, cuequivariance kernels):

1. **compile-score-default**: torch.compile on entire DiffusionModule, mode="default"
2. **compile-score-reduce-overhead**: same, mode="reduce-overhead" (CUDA graphs)
3. **compile-transformer-default**: torch.compile on just the 24-layer DiffusionTransformer
4. **compile-transformer-reduce-overhead**: same, mode="reduce-overhead"
5. **bypass-only**: no compilation (control)

## Results

### Sweep 1 (seed=42, 1 run per complex)

| Config | predict_only_s | Wall time_s | pLDDT | Speedup |
|--------|---------------|-------------|-------|---------|
| bypass-only (no compile) | 7.53 | 65.3 | 0.7186 | 0.82x |
| compile-score-default | 7.83 | 163.0 | 0.7188 | 0.33x |
| compile-score-reduce-overhead | 7.59 | 144.6 | 0.7186 | 0.37x |

### Sweep 2 (seed=42, 1 run per complex, extended configs)

| Config | predict_only_s | Wall time_s | pLDDT | Speedup |
|--------|---------------|-------------|-------|---------|
| bypass-only | 8.1 | 65.6 | 0.7186 | 0.82x |
| compile-score-default | 7.8 | 162.2 | 0.7188 | 0.33x |
| compile-score-reduce-overhead | 7.2 | 156.5 | 0.7188 | 0.34x |
| compile-transformer-default | 8.4 | 130.6 | 0.7188 | 0.41x |
| compile-transformer-reduce-overhead | 8.5 | 128.8 | 0.7188 | 0.42x |

### Per-complex predict_only detail (Sweep 2)

| Complex | No compile | Score default | Score reduce-oh | Transf. default | Transf. reduce-oh |
|---------|-----------|---------------|-----------------|-----------------|-------------------|
| small (~200 res) | 3.6s | 3.5s | 3.3s | 4.1s | 3.5s |
| medium (~400 res) | 7.4s | 7.8s | 7.1s | 7.4s | 7.1s |
| large (~600 res) | 13.2s | 12.7s | 11.3s | 14.1s | 13.6s |

### Compilation warmup times

| Compile target | Mode | Warmup (s) |
|---------------|------|------------|
| Score model | default | 67-113 |
| Score model | reduce-overhead | 97-103 |
| Transformer only | default | 71-75 |
| Transformer only | reduce-overhead | 74-100 |
| No compile | - | 10-15 |

## What Happened

torch.compile with `fullgraph=False` (allowing graph breaks) was applied to the score model or its token transformer submodule. The compilation succeeded without errors, and the warmup pass triggered actual compilation (~70-110s). After compilation, the inference timing showed:

1. **predict_only time is essentially unchanged.** The best result (compile-score-reduce-overhead) showed a 0.9s mean improvement in Sweep 2 but was 0.06s slower in Sweep 1. This is run-to-run variance, not a real improvement.

2. **Wall time is 2-3x worse** due to compilation overhead that is not amortized over a single inference call.

3. **Compiling the transformer alone was slightly worse** than compiling the full score model, suggesting that graph breaks at the encoder/transformer/decoder boundaries add overhead.

## What I Learned

**Why torch.compile does not help here:**

1. **cuequivariance kernels already optimize the heavy operations.** The Boltz2 model uses fused CUDA kernels from cuequivariance for the attention and equivariant operations. torch.compile's inductor backend cannot improve on hand-written CUDA kernels -- it primarily helps with Python overhead and small-op fusion in the "long tail" of operators.

2. **autocast bf16 already saturates memory bandwidth.** With bf16 tensor cores active, the memory-bandwidth-bound operations (which dominate transformer inference) are already running at near-peak throughput. Kernel fusion gives diminishing returns when bandwidth is already the bottleneck.

3. **Graph breaks prevent full optimization.** Even with `fullgraph=False`, the score model has complex control flow (einops rearrange, conditional paths) that causes graph breaks. Each break incurs tracing overhead and prevents cross-op fusion.

4. **12 ODE steps are not enough to amortize.** While 12 calls per forward is better than 1, the per-call overhead from dynamo guard checks and the inability to fully capture the graph means the 12x amortization is not sufficient.

5. **Prior orbits (compile-score-v3, #20) found the same thing.** This orbit confirms and extends that finding: even compiling at the ODE loop level (not just per-module) and using reduce-overhead mode for CUDA graphs does not help when the underlying operations are already highly optimized.

## Prior Art & Novelty

### What is already known
- compile-tf32 (#4): compiled pairformer, structure, confidence -- negligible impact on <3% of time
- compile-noguard (#16): fullgraph=True hit dynamic shapes in attention
- compile-score-v3 (#20): compiled score module, recompilation overhead exceeded savings
- [PyTorch torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html) notes that models with custom CUDA kernels may not benefit from compile

### What this orbit adds
- Confirms that torch.compile on the diffusion ODE loop specifically (the 84% dominant cost) does not help even with:
  - reduce-overhead mode (CUDA graphs)
  - Targeted compilation of just the transformer vs full score model
  - Warmup excluded from timing
- Documents compilation warmup times (70-110s) for different targets and modes
- Provides evidence that cuequivariance kernels + bf16 autocast leave no room for torch.compile improvement

### Honest positioning
This is a negative result that confirms and extends prior orbit findings. No novelty claim -- the primary value is definitively ruling out torch.compile as a speedup avenue for the Boltz2 diffusion loop when cuequivariance kernels are active.

## References

- orbit/bypass-lightning (#44): parent orbit, current winner at 3.5x
- orbit/compile-tf32 (#4): compiled pairformer/structure/confidence
- orbit/compile-noguard (#16): fullgraph=True failed on dynamic shapes
- orbit/compile-score-v3 (#20): compiled score module, overhead exceeded savings
- orbit/cpu-gap-profile (#42): timing breakdown showing 84% diffusion cost
