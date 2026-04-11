# Research Problem: Inference Speedups for Boltz Biomolecular Structure Prediction

## Research Question

How much can we speed up Boltz-2 GPU inference while keeping per-complex lDDT within 2 percentage points of the baseline (200-step EDM/Karras diffusion, 3 recycling steps), measured on the CASP15 / held-out PDB test set? Any measurable speedup is valuable; 2-3x would meaningfully accelerate production pipelines; 5x+ would be transformative.

---

## Background and Context

### What Boltz Is

Boltz is a family of open-source biomolecular structure prediction models (Boltz-1, Boltz-2) that predict protein, ligand, and nucleic-acid 3D structures — comparable in scope to AlphaFold3. The architecture combines:

- A **trunk** consisting of an MSA module followed by a Pairformer (triangular attention, 48–64 blocks). The trunk runs `recycling_steps + 1` times (default: 4×).
- A **diffusion module** for structure generation: 200 denoising steps, each passing atoms through a full 24-layer token transformer (atom encoder → transformer → atom decoder).

### Inference Pipeline (in order)

| Stage | Description | Approx. relative cost |
|---|---|---|
| 1 | Input parsing + MSA generation | Low (I/O-bound) |
| 2 | Data featurization | Low |
| 3 | Trunk (MSA + Pairformer) | Moderate (4× recycling) |
| 4 | Diffusion conditioning (Boltz-2: once, outside loop) | Low |
| 5 | Diffusion sampling (200 steps × 24-layer transformer) | **Dominant** |
| 6 | Confidence estimation (pLDDT, PAE, PTM) | Low |
| 7 | Output writing | Negligible |

Stage 5 accounts for the large majority of wall-clock inference time on a typical GPU. Any approach that meaningfully reduces the number of diffusion steps, reduces cost per step, or eliminates redundant computation in the trunk will have the greatest practical impact.

### Why This Matters

Boltz is used in production binder design pipelines (e.g., CD19 binder design) where each designed sequence requires a separate Boltz prediction with MSA generation. A typical campaign generates dozens to hundreds of candidate binders, each requiring 10-40 minutes of Boltz time (MSA: 5-30 min, GPU inference: 5-10 min). Inputs are protein-protein complexes (~300-400 residues total) with soft PDB templates. Both predicted structures and confidence scores (iPTM, pDockQ) are used downstream for filtering. A 5-10x end-to-end speedup would cut campaign turnaround from days to hours.

---

## Known Results (Existing Speedup Mechanisms in the Codebase)

The following mechanisms are already present or partially implemented but are **not enabled by default**:

1. **`torch.compile` on the score model and Pairformer** — Code exists to wrap these modules with `torch.compile`. Expected speedup: 1.3–2×. Not enabled by default due to compilation overhead on first call and potential compatibility issues.

2. **Matmul precision flag** — PyTorch's default `torch.set_float32_matmul_precision("highest")` disables TF32 on Ampere+ GPUs. Switching to `"high"` enables TF32 at negligible quality cost. Expected speedup: ~10% (free).

3. **Diffusion step count as a CLI parameter** — The number of diffusion sampling steps is already configurable. The default of 200 can be reduced; the quality vs. step-count trade-off has not been formally characterized for Boltz.

No Flash Attention implementation exists in the current codebase (attention is computed via manual einsum). INT8/FP8 quantization is not implemented. DDIM or other deterministic samplers are not implemented.

---

## Proposed Success Criteria

### Primary Metric

**Speedup factor at iso-quality**: the ratio of baseline GPU inference time to optimized GPU inference time, measured under the constraint that per-complex lDDT on the evaluation set does not decrease by more than 2 percentage points relative to the baseline.

**Timing scope**: End-to-end prediction time, from input YAML to output structure + confidence scores. This includes MSA generation (via `--use_msa_server`), featurization, trunk, diffusion, and confidence scoring. MSA is included because it is the dominant wallclock cost in production binder design pipelines (5-30 min per call). GPU-only time is reported as a secondary breakdown.

```
speedup = T_baseline / T_optimized
subject to: mean_pLDDT(optimized) >= mean_pLDDT(baseline) - 0.02
```

**Quality proxy**: We use pLDDT (Boltz's predicted lDDT from the confidence head) rather than true lDDT (which requires reference structures and OpenStructure). pLDDT correlates well with structural accuracy and is available for any prediction. True lDDT will be validated against reference structures for a representative subset to confirm the proxy holds.

- **Direction**: MAXIMIZE speedup
- **Practical value threshold**: 2-3× speedup at iso-quality
- **Aspirational target**: 5×+ speedup at iso-quality

### Secondary Metrics

- **Per-modality lDDT**: protein, ligand, and nucleic acid lDDT reported separately (prevents hiding ligand regression behind protein quality).
- **Per-step quality degradation**: lDDT as a function of diffusion step count, to characterize the quality–speed Pareto frontier.
- **TM-score**: global topology agreement, as a complement to lDDT.
- **Ligand RMSD** (for protein–ligand complexes): relevant for drug-discovery use cases.
- **DockQ** (for complexes): interface quality metric.
- **GPU memory peak**: optimized inference must fit within 40 GB VRAM (A100/A6000 class).

### Evaluation Protocol

- **Model**: Boltz-2 (primary). Boltz-1 results reported separately where applicable — the two models differ in precision (Boltz-2 uses bf16-mixed by default; Boltz-1 uses fp32), trunk depth (64 vs 48 pairformer blocks), and diffusion conditioning architecture.
- **Test set**: CASP15 targets + a representative held-out PDB set (release date after Boltz training cutoff). Exact target list to be defined in `research/eval/config.yaml`.
- **Hardware**: single A100-80GB or equivalent.
- **Baseline**: Boltz-2, 200-step EDM/Karras-style diffusion (stochastic sampler with gamma noise injection), 3 recycling steps, `float32_matmul_precision="highest"`, `torch.compile` disabled, `diffusion_samples=1`.
- **Timing**: median of 3 runs per complex, excluding model load and `torch.compile` warmup. Seeds fixed for reproducibility.
- **Quality**: lDDT computed via OpenStructure `compare-structures` (same tooling as `scripts/eval/run_evals.py`).

---

## Search Space

Approaches to explore, roughly ordered by expected impact:

### High impact — no training required

1. **Reduce diffusion steps (200 → 20–50)**: Directly cuts Stage 5 cost by 4–10×. The AlphaFold3 paper reports acceptable quality at 20–50 steps. Characterize the quality–step-count curve for Boltz.

2. **Deterministic ODE sampling**: The current sampler uses stochastic noise injection (gamma parameter) at each step. Setting `gamma_0=0` and `noise_scale=1.0` gives a deterministic first-order Euler ODE solver, which converges in fewer steps than the stochastic variant. No retraining needed — just parameter tuning.

3. **Flash Attention via `F.scaled_dot_product_attention`**: Replace manual einsum-based attention in the Pairformer and score-model transformer with PyTorch's fused SDPA kernel (backed by FlashAttention-2 on supported hardware). Expected 1.5–3× speedup on attention layers, plus memory reduction enabling larger batches.

4. **Enable TF32 (`torch.set_float32_matmul_precision("high")`)**: Single-line change. Verify no quality regression, then enable as default.

5. **Enable `torch.compile`**: Profile compilation overhead and steady-state speedup for the score model and Pairformer. Determine whether per-request compilation amortizes over a typical inference run.

6. **Reduce recycling steps (3 → 1)**: Cuts trunk cost by ~2×. The quality impact of fewer recycling iterations has not been characterized for Boltz; benchmark lDDT vs. recycling count.

### Medium impact — no training required

7. **Mixed-precision inference (BF16)**: Run trunk and score model in BF16 rather than FP32. Expected 1.5–2× memory bandwidth improvement and potential throughput gain on Ampere+.

8. **Batched diffusion steps**: Where memory permits, batch multiple denoising steps or samples through the score model simultaneously.

9. **Early exit / adaptive recycling**: Terminate trunk recycling early when the pair representation has converged (e.g., measured by cosine similarity between successive iterations).

### Medium impact — kernel-level optimizations

10. **Custom Triton kernels for triangular attention**: The Pairformer uses triangular attention (unique to AlphaFold-style models) implemented via generic einsum. A fused Triton kernel could eliminate memory round-trips and exploit the triangular structure.

11. **Fused attention + LayerNorm kernels**: Combine multi-head attention with post-attention LayerNorm into a single kernel pass, reducing memory bandwidth.

12. **Operator fusion via torch.compile inductor**: Let the inductor backend auto-fuse operations in the score model. Complements explicit Triton kernels — inductor handles the long tail of small ops.

### Medium-lower impact

13. **INT8 / FP8 post-training quantization**: Quantize score model and Pairformer weights without retraining. Requires calibration dataset. Expected 1.5–3× throughput on H100.

14. **Batched multi-sample inference**: When generating multiple diffusion samples, maximize `max_parallel_samples` to batch all samples in a single forward pass, removing chunking overhead.

---

## Constraints

1. **Quality floor**: Mean lDDT on the evaluation set must remain within 2 percentage points of the 200-step DDPM baseline. This is non-negotiable — a speedup that breaks predictions is not a speedup.

2. **Standard GPU hardware**: Primary evaluation on A100-80GB (for published comparability). Secondary validation on L4-24GB (production deployment target for binder design pipelines — cheaper, lower VRAM, no BF16 tensor cores, no FP8). Approaches should ideally work on both, but A100 is the gating benchmark.

3. **No retraining**: All approaches must work with existing Boltz checkpoints. No fine-tuning, distillation, or retraining of any kind. This constrains us to inference-time optimizations only.

4. **Upstream API compatibility**: The public `boltz predict` CLI interface and the Python API must remain unchanged. Speedup flags should be opt-in (e.g., `--fast` or `--diffusion_steps N`).

5. **Reproducibility**: Optimized inference must be deterministic (or have a fixed-seed mode) to allow result comparison across runs and hardware.

6. **No external service dependencies**: Inference must run fully offline. Approaches relying on external APIs or proprietary libraries not already in the dependency tree are out of scope.
