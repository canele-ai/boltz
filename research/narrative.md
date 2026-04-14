**Thesis:** Boltz-2 end-to-end inference latency is dominated by trunk recycling (MSA + Pairformer, 4 passes) and MSA server latency, not by the diffusion sampling loop. The highest-leverage inference-time optimization is eliminating trunk recycling (recycling_steps=0) combined with deterministic ODE sampling at 20 steps, which together yield 1.79x speedup at +1.96pp quality. The config-space ceiling with current evaluator architecture is ~1.8x; reaching 2x+ requires structural changes (MSA pre-caching, torch upgrade, bf16 trunk, or persistent model serving).

**Key evidence:**
- orbit/step-reduction (1.73x): Recycling reduction yields 1.7x; step reduction alone yields 0.95x. Trunk is the bottleneck.
- orbit/ode-sampler (1.79x): ODE sampling adds 7% improvement on top; extends viable step count to 10 steps where SDE collapses.
- orbit/compile-tf32 (0.97x, dead-end): TF32 is irrelevant for bf16-mixed models; torch.compile harmful in subprocess architecture.
- orbit/flash-sdpa (0.745x, dead-end): FA2 blocked by pair bias; SDPA slower than einsum at Boltz sequence lengths.
- orbit/early-exit-recycling (1.69x, dead-end): Adaptive exit requires 2 trunk passes minimum; dominated by recycling=0 (1 pass).
- orbit/l40s-kernels (diagnostic): cuequivariance kernels blocked by cublas version conflict; bf16 triangular multiply shows 1.94x potential on isolated op; TF32 invisible in e2e.
- orbit/combined-fast (1.66x, dead-end): Exhaustive config sweep confirms ceiling at 1.7-1.85x; identifies 4 architectural paths to 2x+.

**CASP15 validation (2026-04-13):**
- 35 CASP15 protein targets (T* + H*), 44–1,439 residues, predict-only timing with pre-cached MSAs on L40S.
- Predict-only speedup: **1.61x** (40.5s → 25.1s mean), consistent with 1.69x on 3-complex dev set.
- pLDDT: +0.09pp (0.8133 → 0.8143), quality preserved across diverse targets.
- CA RMSD: −0.7Å (20.9Å → 20.2Å on 30 targets with valid chain matching), slightly better.
- 3 targets >2,800 residues OOM'd on L40S; 5 targets have N/A RMSD due to non-standard chain IDs.
- Only 1/35 targets regressed >5pp pLDDT (T1123, −12.3pp); 2/35 regressed >2pp.
- Scripts: research/eval/prepare_casp15.py, generate_casp15_msas.py, casp15_eval.py
- Results: research/eval/casp15/results/baseline.json, optimized.json

**Remaining open questions:**
- Can removing x.float() from triangular_mult.py:116 unlock bf16 throughout the trunk without quality regression? The isolated op shows 1.94x; the end-to-end gain is unknown.
- Does torch 2.6+ with cuequivariance kernels provide measurable e2e speedup once MSA latency is controlled for?
- Would persistent model serving (eliminate per-complex model load) add 5-10s of savings, pushing the ceiling to ~2x even without MSA pre-caching?

**Story arc (if writing a paper today):**
1. Introduction: Boltz-2 pipeline latency bottleneck is misidentified in the literature as the diffusion loop; we show it is the trunk recycling + MSA combination.
2. Method: Systematic ablation of all inference-time optimizations: step reduction, ODE sampling, recycling elimination, precision/compile, attention kernels.
3. Results: 1.79x speedup at +1.96pp quality. Qualitative shift: ODE extends viable step range from 20 to 10.
4. Analysis: Negative controls (TF32, compile, SDPA) as ablation ruling out diffusion-loop approaches. Time breakdown showing diffusion is 7-12% of optimized runtime.
5. CASP15 validation: 1.61x speedup at +0.09pp pLDDT across 35 targets. Missing: per-modality lDDT, clean implementation.

**Shift from Milestone 1:** Previous thesis was "trunk recycling is the bottleneck, ODE and kernel optimizations remain open." New thesis: "ODE-20+recycling=0 is the Pareto-optimal config-space solution at 1.79x; the 2x+ question is now an architectural/systems question, not a configuration question."
