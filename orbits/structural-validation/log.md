# Structural Validation: Baseline vs Optimized Predictions

## Goal

Validate that the optimized inference config (ODE-12, TF32+bf16, bypass+warmup) produces structures that are geometrically consistent with the baseline (200-step, fp32). The existing eval harness checks pLDDT (model confidence), but pLDDT could be preserved while the actual 3D coordinates diverge significantly. This orbit adds an all-atom RMSD comparison to catch that case.

## Method

1. Run **baseline** config: 200 sampling steps, 3 recycles, fp32, seed=42
2. Run **optimized** config: 12 ODE steps (gamma_0=0), 3 recycles, TF32+bf16, seed=42
3. Parse both mmCIF outputs with BioPython
4. Compute CA RMSD and all-atom RMSD after optimal superposition (Kabsch algorithm via `Bio.PDB.Superimposer`)
5. Compute per-residue CA deviation after alignment

## Thresholds

| Metric | Negligible | Marginal | Divergent |
|--------|-----------|----------|-----------|
| CA RMSD | < 0.5 A | 0.5 -- 1.0 A | > 1.0 A |
| All-atom RMSD | < 1.0 A | 1.0 -- 2.0 A | > 2.0 A |
| Per-residue max | < 2.0 A | 2.0 -- 5.0 A | > 5.0 A |

"Negligible" means the difference is within floating-point precision changes (fp32 vs TF32/bf16). "Divergent" means the ODE sampling trajectory (gamma_0=0) produces fundamentally different conformations -- which would be a real finding about the optimization.

## Results

**Status: PENDING** -- run with:
```bash
cd /home/liambai/code/boltz/.worktrees/structural-validation
modal run orbits/structural-validation/validate_structures.py
```

Results table will be filled in after execution.

### Expected RMSD Table

| Complex | CA RMSD (A) | AA RMSD (A) | Max CA Dev (A) | % < 2A | BL pLDDT | Opt pLDDT |
|---------|-------------|-------------|----------------|--------|----------|-----------|
| small_complex | -- | -- | -- | -- | -- | -- |
| medium_complex | -- | -- | -- | -- | -- | -- |
| large_complex | -- | -- | -- | -- | -- | -- |

### Verdict

Pending execution.

## Script

`validate_structures.py` -- Modal script that runs both configs on the test set and computes RMSD.

## Notes

- Same seed (42) is used for both configs so diffusion noise initialization is identical
- The key variable is `gamma_0=0` (ODE mode) vs `gamma_0=0.8` (SDE mode with stochastic noise) and step count (12 vs 200)
- Even with the same seed, ODE sampling with 12 steps will follow a different numerical trajectory than SDE with 200 steps, so some structural difference is expected
- The question is whether the difference is within the noise floor of the model or represents a meaningful structural divergence
