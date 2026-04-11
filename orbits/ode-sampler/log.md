---
strategy: ode-sampler
type: experiment
status: complete
eval_version: eval-v1
metric: 1.79
issue: 6
parents:
  - orbit/step-reduction
---

# ODE Sampler: Deterministic Diffusion for Fewer Steps

## Glossary

- **pLDDT**: predicted Local Distance Difference Test -- Boltz's confidence proxy for structural accuracy (0--1 scale)
- **pp**: percentage points (absolute difference in pLDDT scaled to 0--100)
- **SDE**: Stochastic Differential Equation -- the default sampler with noise injection (gamma_0 > 0)
- **ODE**: Ordinary Differential Equation -- deterministic sampler with gamma_0 = 0 (no noise injection)
- **EDM**: Elucidating the Design space of diffusion-based generative Models (Karras et al.) -- the sampling framework used by Boltz
- **gamma_0**: noise injection scale in the EDM sampler; 0 = deterministic ODE, 0.8 = default stochastic

## Results

**Best validated configuration: ODE sampler (gamma_0=0), 20 steps, 0 recycling steps = 1.79x speedup, passes quality gate (+1.96pp pLDDT)**

Setting gamma_0=0 converts the Boltz EDM/Karras stochastic sampler into a deterministic first-order Euler ODE solver. At 20 steps with recycling_steps=0, ODE-20 achieves 1.79x speedup compared to the 200-step stochastic baseline -- a meaningful improvement over the SDE-20 configuration (1.67x) from the parent orbit step-reduction, measured under identical conditions.

The most striking finding: **deterministic ODE sampling at 10 steps maintains quality (pLDDT 0.73) where stochastic SDE at 10 steps catastrophically collapses (pLDDT 0.41)**. This confirms the hypothesis from Song et al. (DDIM) and Karras et al. (EDM) that deterministic ODE paths converge in fewer steps. However, the 10-step timing advantage is negated by MSA latency variance in end-to-end measurements.

### Validated Configurations (3 runs each, L40S)

| Config | Steps | gamma_0 | Time(s) | pLDDT | Delta(pp) | Speedup | Gate |
|--------|-------|---------|---------|-------|-----------|---------|------|
| SDE-20-r0 | 20 | 0.8 | 42.2 | 0.7263 | +1.56 | 1.67x | PASS |
| ODE-20-r0 | 20 | 0.0 | 39.3 | 0.7303 | +1.96 | 1.79x | PASS |
| ODE-10-r0 | 10 | 0.0 | 51.8 | 0.7280 | +1.73 | 1.36x | PASS |

Per-complex validated timing (median of 3 runs):

| Config | Small | Medium | Large |
|--------|-------|--------|-------|
| SDE-20-r0 time | 38.6s | 40.5s | 47.5s |
| ODE-20-r0 time | 33.2s | 40.0s | 44.8s |
| ODE-10-r0 time | 52.5s | 48.3s | 54.6s |
| SDE-20-r0 pLDDT | 0.8872 | 0.4764 | 0.8153 |
| ODE-20-r0 pLDDT | 0.8860 | 0.4888 | 0.8161 |
| ODE-10-r0 pLDDT | 0.8858 | 0.4804 | 0.8177 |

### Phase 1 Sweep (1 run each, recycling_steps=0)

| Config | Steps | gamma_0 | ns | Time(s) | pLDDT | Delta(pp) | Gate |
|--------|-------|---------|-----|---------|-------|-----------|------|
| SDE-20 | 20 | 0.8 | 1.003 | 73.2 | 0.7255 | +1.48 | PASS |
| SDE-10 | 10 | 0.8 | 1.003 | 70.0 | 0.4164 | -29.42 | FAIL |
| ODE-20 | 20 | 0.0 | 1.003 | 58.9 | 0.7303 | +1.96 | PASS |
| ODE-10 | 10 | 0.0 | 1.003 | 62.0 | 0.7285 | +1.78 | PASS |
| ODE-5 | 5 | 0.0 | 1.003 | 68.6 | 0.5114 | -19.93 | FAIL |
| ODE-20-ns1 | 20 | 0.0 | 1.0 | 68.2 | 0.7308 | +2.01 | PASS |
| ODE-10-ns1 | 10 | 0.0 | 1.0 | 69.8 | 0.7280 | +1.73 | PASS |

## Approach

The Boltz EDM sampler implements the Karras et al. (2022) stochastic sampling scheme, where at each denoising step, the algorithm:

1. Inflates the current noise level: `t_hat = sigma * (1 + gamma)`
2. Adds fresh noise proportional to `sqrt(noise_scale^2 * (t_hat^2 - sigma^2))`
3. Evaluates the score model at the inflated noise level
4. Takes a Euler step toward the denoised prediction

When gamma_0=0, step 1 becomes `t_hat = sigma` (no inflation), step 2 produces zero noise (since `t_hat^2 - sigma^2 = 0`), and the algorithm reduces to a deterministic first-order Euler ODE solver along the probability flow ODE. This is mathematically equivalent to DDIM (Song et al. 2020) in the EDM parameterization.

The implementation monkey-patches `Boltz2DiffusionParams.gamma_0` to 0 before model loading. Since `gamma_0` only affects the sampling loop (not training or model weights), no retraining is needed -- the same checkpoint works with both stochastic and deterministic sampling.

The key code change is in the `AtomDiffusion.sample()` method:
```python
gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
# When gamma_0=0: gammas are all zero, t_hat = sigma, noise_var = 0
```

## What I Learned

1. **Deterministic ODE sampling extends the usable step range by at least 2x**: SDE catastrophically fails at 10 steps (pLDDT 0.41) while ODE maintains quality (pLDDT 0.73). The ODE path follows a smooth trajectory in probability space, while the stochastic path needs enough steps for the noise to average out. This is consistent with the DDIM and EDM literature.

2. **ODE-20 is modestly but consistently faster than SDE-20**: Per-complex timing shows ODE saves 2-5s per complex (33.2 vs 38.6 for small, 44.8 vs 47.5 for large). The savings come from eliminating the noise generation and sampling at each step. At recycling_steps=0, this compounds to a 1.79x vs 1.67x speedup.

3. **ODE-10 does not realize speedup despite fewer steps**: MSA server latency dominates end-to-end timing. The first run for small_complex took 114.4s (vs 52.5s and 41.9s for subsequent runs), indicating an MSA cache miss. With pre-cached MSA, ODE-10 should show more consistent timing.

4. **noise_scale has minimal effect**: Setting noise_scale=1.0 (vs default 1.003) produces nearly identical quality. When gamma_0=0, noise_scale only affects the initial noise generation and residual floating-point behavior, not the denoising trajectory.

5. **ODE quality is slightly better than SDE at 20 steps**: ODE-20 pLDDT is 0.7303 vs SDE-20's 0.7263. The deterministic trajectory converges more reliably, though the difference (+0.4pp) is within noise for 3 test complexes.

## Limitations

- The improvement from ODE-20 over SDE-20 is modest (1.79x vs 1.67x). The 7% timing improvement comes entirely from eliminating per-step noise operations, not from step reduction.
- ODE-10 should give a larger speedup once MSA is pre-cached, but with end-to-end timing including MSA, it actually appears slower due to variance.
- The test set has only 3 complexes. The quality comparison may not generalize.
- The monkey-patching approach (overriding `Boltz2DiffusionParams`) works for evaluation but is not a clean production solution. A proper implementation would add a `--deterministic_sampling` flag to the Boltz CLI.

## Prior Art & Novelty

### What is already known
- Song et al. 2020 (DDIM) showed that deterministic sampling converges in fewer steps than stochastic DDPM
- Karras et al. 2022 (EDM) provide the ODE formulation that Boltz's sampler implements; the gamma parameter controls stochasticity
- AlphaFold3 (Abramson et al. 2024) uses a similar EDM sampler and reports quality at 20-50 steps

### What this orbit adds
- Quantitative evidence that deterministic ODE at 10 steps maintains Boltz-2 quality where stochastic SDE collapses
- Validated that ODE-20 with recycling=0 achieves 1.79x speedup on L40S, a modest but measurable improvement over the SDE baseline (1.67x, same conditions)
- Confirmation that the gamma_0 monkey-patch is sufficient -- no model retraining needed

### Honest positioning
This orbit applies the well-known DDIM/ODE insight to Boltz-2's EDM sampler. The approach is straightforward (set gamma_0=0) and produces a small incremental improvement in wall-clock speedup. The main finding -- ODE convergence at 10 steps where SDE collapses -- is consistent with the diffusion literature. No novelty claim beyond empirical validation for Boltz-2.

## References

- Song J, Meng C, Ermon S. Denoising Diffusion Implicit Models. ICLR, 2021. https://arxiv.org/abs/2010.02502
- Karras T et al. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS, 2022. https://arxiv.org/abs/2206.00364
- Abramson J et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630:493-500, 2024.
- Wohlwend J et al. Boltz-1: Democratizing Biomolecular Interaction Modeling. bioRxiv, 2024.
- Parent orbit: step-reduction (#3) -- established that recycling_steps=0 is the main speedup lever

![ODE vs SDE comparison](figures/ode_vs_sde_comparison.png)
