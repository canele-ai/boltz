---
strategy: bypass-lightning
type: experiment
status: in-progress
eval_version: eval-v4
metric: null
issue: 44
parents:
  - orbit/eval-v2-winner
  - orbit/cpu-gap-profile
---

# Bypass Lightning

## Hypothesis

The cpu-gap-profile (#42) found that ~6.7s of the ~13s predict_only_s budget
is non-GPU overhead:
- In-loop overhead: 4.7s (36%) -- Lightning's Trainer.predict() adding batch
  collation, per-batch hooks/callbacks, and tensor transfer management
- Trainer setup: 2.0s (15%) -- strategy initialization, callback registration

Replacing Lightning's Trainer.predict() loop with direct model.predict_step()
calls should eliminate most of this overhead.

## Approach

Monkey-patch `pytorch_lightning.Trainer.predict` inside the wrapper subprocess
so that when boltz.main.predict() calls `trainer.predict(model, datamodule=dm)`,
our replacement function runs instead. This replacement:

1. Gets the DataLoader from the DataModule directly
2. Moves model to GPU manually
3. Optional CUDA warmup: runs one forward pass to trigger kernel JIT (~2.2s)
4. Iterates batches with `torch.no_grad()` + `torch.autocast("cuda", dtype=torch.bfloat16)`
5. Calls `model.predict_step(batch, batch_idx)` directly
6. Writes outputs via BoltzWriter.write_on_batch_end (trainer=None is fine, arg unused)

All eval-v2-winner optimizations are also applied:
- ODE sampling (gamma_0=0.0) with 12 steps, 0 recycling
- TF32 matmul precision
- bf16 trunk (no .float() upcast in triangular_mult)
- cuequivariance CUDA kernels

## Key Design Decisions

- **Monkey-patch Trainer.predict** rather than reimplementing input processing:
  This lets boltz.main.predict() handle all the complex input processing
  (YAML parsing, MSA fetching, featurization, manifest creation) unchanged.
  We only replace the final prediction loop.

- **transfer_batch_to_device**: Replicated the exact CPU-only key list from
  Boltz2InferenceDataModule.transfer_batch_to_device to ensure correct tensor
  placement.

- **BoltzWriter compatibility**: The writer's write_on_batch_end accepts
  trainer=None because the arg is annotated `noqa: ARG002` (unused).

## Files

- `boltz_bypass_wrapper.py` -- The bypass wrapper (monkey-patches Trainer.predict)
- `eval_bypass.py` -- Modal evaluator with sweep (Trainer vs bypass vs bypass+warmup)

## Results

(Pending evaluation on Modal L40S)
