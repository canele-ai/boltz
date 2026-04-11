# Metric Provenance: speedup_at_iso_quality (lDDT-gated)

## Overview

This document traces the provenance of the two metrics that compose our primary evaluation measure:

1. **lDDT** (local Distance Difference Test) -- the quality gate metric
2. **speedup_at_iso_quality** -- the composite speedup metric, defined as wall-clock ratio subject to an lDDT regression bound

---

## 1. lDDT: Formal Definition

### Primary Reference

Mariani V, Biasini M, Barbato A, Schwede T. "lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests." *Bioinformatics*, 29(21):2722-2728, 2013.
DOI: [10.1093/bioinformatics/btt473](https://doi.org/10.1093/bioinformatics/btt473)

### Mathematical Definition

Given a reference structure R and a model structure M:

1. **Distance inclusion radius**: For each atom *i* in R, collect all atoms *j* in R within a cutoff distance R_0 (default: 15 A) to form the set of reference distance pairs L_i.

2. **Distance preservation test**: For each pair (i, j) in L_i, compute the absolute difference between the inter-atomic distance in M and the corresponding distance in R:

   delta_ij = |d_M(i,j) - d_R(i,j)|

3. **Thresholded fraction**: For each of four tolerance thresholds t in {0.5, 1.0, 2.0, 4.0} Angstroms, compute the fraction of preserved distances:

   f_t = (number of pairs where delta_ij < t) / (total number of pairs)

4. **lDDT score**: The average of the four fractions:

   lDDT = (f_0.5 + f_1.0 + f_2.0 + f_4.0) / 4

### Key Properties

- **Range**: [0, 1] (sometimes reported as percentage [0, 100])
- **Superposition-free**: Does not require structural alignment; compares local distance patterns directly
- **Atom scope**: Default is all-atom (including side chains). The variant **lDDT-Ca** restricts to C-alpha atoms only. Some papers use backbone atoms.
- **Sequence separation**: Default is zero (all pairs considered). Some evaluations use a minimum sequence separation to focus on non-trivial contacts.
- **Stereochemical filter**: The original definition includes optional exclusion of stereochemically implausible models (bond length/angle violations), though this is not universally applied.

### Variants in the Literature

| Variant | Atoms | Used by |
|---------|-------|---------|
| lDDT (all-atom) | All heavy atoms | Mariani et al. 2013 (default), CASP assessors, Boltz-1 |
| lDDT-Ca | C-alpha only | AlphaFold2 (pLDDT trains against this), ESMFold |
| lDDT-PLI | Protein-ligand interface atoms | CASP15 ligand assessment, Boltz-1, AlphaFold3 |
| pLDDT | Predicted lDDT (confidence score) | AlphaFold2, AlphaFold3, ESMFold, Boltz-1, Boltz-2 |

**pLDDT** is a model-internal confidence estimate that is trained to predict the true lDDT-Ca of the model's own output. It is not lDDT itself, but a predictor of it. In AlphaFold2, the relationship is approximately linear: lDDT-Ca = 0.997 * pLDDT - 1.17 (Pearson r = 0.76 across 10,795 chains). Our evaluator uses pLDDT as a proxy, with true lDDT validated on a subset via OpenStructure.

---

## 2. Published Uses of lDDT (>= 3 required)

### Use 1: AlphaFold2 (Jumper et al. 2021)

- **Citation**: Jumper J, Evans R, Pritzel A, Green T, Figurnov M, Ronneberger O, Tunyasuvunakool K, Bates R, Zidek A, Potapenko A, Bridgland A, Meyer C, Kohl SAA, Ballard AJ, Cowie A, Romera-Paredes B, Nikolov S, Jain R, Adler J, Back T, Petersen S, Reiman D, Clancy E, Zielinski M, Steinegger M, Pacholska M, Berghammer T, Bodenstein S, Silver D, Vinyals O, Senior AW, Kavukcuoglu K, Kohli P, Hassabis D. "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596(7873):583-589, 2021.
- **URL**: [10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- **lDDT definition used**: lDDT-Ca (C-alpha only), following Mariani et al. 2013. The model produces pLDDT as a per-residue confidence score trained to regress the true lDDT-Ca.
- **Dataset**: CASP14 (87 protein domains from 67 targets)
- **Reported values**: Median domain GDT_TS = 92.4. The paper reports pLDDT as a confidence measure and demonstrates its strong correlation with true lDDT-Ca. AlphaFold2 achieved near-experimental accuracy on the majority of CASP14 targets, with 58 out of 92 domains achieving GDT_TS > 90.
- **Normalization**: lDDT-Ca in [0, 1]; pLDDT in [0, 1] (sometimes displayed as 0-100).

### Use 2: AlphaFold3 (Abramson et al. 2024)

- **Citation**: Abramson J, Adler J, Dunger J, Evans R, Green T, Pritzel A, Ronneberger O, Willmore L, Ballard AJ, Bambrick J, Bodenstein SW, Evans DA, Hung CC, O'Neill M, Reiman D, Tunyasuvunakool K, Wu Z, Zemgulyte A, Arvaniti E, Beattie C, Bertolli O, Bridgland A, Cherepanov A, Congreve M, Cowen-Rivers AI, Cowie A, Figurnov M, Fuchs FB, Gladman H, Jain R, Khan YA, Low CMR, Perlin K, Potapenko A, Savy P, Singh S, Stecula A, Thillaisundaram A, Tong C, Yakneen S, Zhong ED, Zielinski M, Zidek A, Bapst V, Kohli P, Jaderberg M, Hassabis D, Jumper JM. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature*, 630(8016):493-500, 2024.
- **URL**: [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)
- **lDDT definition used**: Full-complex lDDT (all-atom), lDDT-PLI for protein-ligand interfaces, following Mariani et al. 2013. Also uses pLDDT as confidence.
- **Dataset**: PDB evaluation set (post-training-cutoff structures), CASP15 RNA targets, PoseBusters benchmark
- **Reported values**: Full-complex lDDT of 82.8 (PDB 7PZB, bacterial transcriptional regulator + DNA + cGMP) and 83.0 (human coronavirus OC43 spike protein). The paper uses lDDT as a training metric and reports it on the evaluation set as a function of training steps. Uses 200 diffusion sampling steps with a noise schedule based on the EDM/Karras framework.
- **Normalization**: lDDT in [0, 100] for reporting; [0, 1] internally.

### Use 3: Boltz-1 (Wohlwend et al. 2024)

- **Citation**: Wohlwend J, Corso G, Passaro S, Getz N, Reveiz M, Leidal K, Swiderski W, Atkinson L, Portnoi T, Chinn I, Silterra J, Jaakkola T, Barzilay R. "Boltz-1: Democratizing Biomolecular Interaction Modeling." *bioRxiv*, 2024.
- **URL**: [10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167)
- **lDDT definition used**: All-atom lDDT and lDDT-PLI, computed via OpenStructure version 2.8.0, following Mariani et al. 2013.
- **Dataset**: CASP15 targets, PDB test set (post-training-cutoff)
- **Reported values**: Results are "comparable" to AlphaFold3 across CASP15, with AlphaFold3 having "a slight edge" on mean lDDT. On protein-ligand metrics, AlphaFold3 and Boltz-1 obtain "slightly better mean lDDT-PLI" than Chai-1, with differences "within the confidence intervals." RNA lDDT median: 0.54 (vs Chai-1: 0.41). All models run with 200 sampling steps and 10 recycling rounds for the comparison.
- **Normalization**: lDDT in [0, 1].

### Use 4: ESMFold (Lin et al. 2023)

- **Citation**: Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, Smetanin N, Verkuil R, Kabeli O, Shmueli Y, dos Santos Costa A, Fazel-Zarandi M, Sercu T, Candido S, Rives A. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 379(6637):1123-1130, 2023.
- **URL**: [10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
- **lDDT definition used**: lDDT-Ca (C-alpha), following Mariani et al. 2013. pLDDT as confidence score.
- **Dataset**: CASP14 targets, CAMEO weekly benchmark
- **Reported values**: Mean lDDT approximately 0.68 on CASP14 (vs AlphaFold2 approximately 0.85, RoseTTAFold approximately 0.81). ESMFold is approximately 6x faster than AlphaFold2 (single-sequence inference, no MSA), representing a direct speedup-at-quality-cost trade-off study.
- **Normalization**: lDDT in [0, 1].

### Use 5: Boltz-2 (Passaro, Corso, Wohlwend et al. 2025)

- **Citation**: Passaro S, Corso G, Wohlwend J, Reveiz M, Thaler S, Somnath VR, Getz N, Portnoi T, Roy J, Stark H, Kwabi-Addo D, Beaini D, Jaakkola T, Barzilay R. "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction." *bioRxiv*, 2025.
- **URL**: [10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)
- **lDDT definition used**: All-atom lDDT, lDDT-PLI, recall lDDT (for conformational ensemble evaluation), following Mariani et al. 2013.
- **Dataset**: PDB evaluation set, CASP15 targets, mdCATH and ATLAS (molecular dynamics benchmarks)
- **Reported values**: Boltz-2 "modestly outperforms Boltz-1" on recall lDDT. On the PDB test set, average lDDT approximately 0.82. Improvements particularly notable for RNA chains and DNA-protein complexes. Matches or moderately improves over Boltz-1 across modalities.
- **Normalization**: lDDT in [0, 1].

---

## 3. Consistency Check

### Core Definition Consistency

All five papers reference lDDT as defined by Mariani et al. 2013. The core formula (four thresholds at 0.5/1/2/4 A, distance inclusion radius 15 A, averaged fraction of preserved distances) is consistent across all uses.

### Variant Divergence

| Aspect | AlphaFold2 | AlphaFold3 | Boltz-1 | ESMFold | Boltz-2 |
|--------|-----------|-----------|---------|---------|---------|
| Atom scope | Ca only | All-atom + PLI | All-atom + PLI | Ca only | All-atom + PLI |
| pLDDT target | lDDT-Ca | lDDT (all-atom) | lDDT (all-atom) | lDDT-Ca | lDDT (all-atom) |
| Evaluation tool | CASP official | CASP official | OpenStructure 2.8 | CASP official | OpenStructure |
| Scale | [0, 1] | [0, 100] reporting | [0, 1] | [0, 1] | [0, 1] |

### Key Inconsistencies to Note

1. **Ca vs all-atom**: AlphaFold2 and ESMFold use lDDT-Ca; later models (AF3, Boltz-1, Boltz-2) use all-atom lDDT. All-atom lDDT is strictly more informative (includes side-chain accuracy) and is typically lower than lDDT-Ca for the same structure.

2. **Scale convention**: AlphaFold3 reports lDDT as a percentage (0-100) in figures, while other papers use the 0-1 convention. Our evaluator uses [0, 1].

3. **pLDDT as proxy**: Our evaluator uses pLDDT (predicted lDDT from the Boltz confidence head) rather than computing true lDDT against reference structures. This is standard practice for inference optimization research where reference structures are not available for every prediction. The correlation between pLDDT and true lDDT is well-established (Pearson r approximately 0.76 in AlphaFold2).

### Implications for Our Evaluation

Our quality gate uses **pLDDT** (which in Boltz-2 is trained against all-atom lDDT), with a 2 percentage point maximum regression. This is consistent with the Boltz evaluation convention. True lDDT (via OpenStructure) is validated on a subset to confirm the proxy holds.

---

## 4. speedup_at_iso_quality: Provenance

### Definition

```
speedup = T_baseline / T_optimized
subject to: mean_pLDDT(optimized) >= mean_pLDDT(baseline) - 0.02
```

This is a standard engineering metric (wall-clock speedup ratio) with a quality constraint. The components are:

- **Speedup ratio**: Standard in systems and ML efficiency literature.
- **Quality gate (lDDT)**: The specific use of lDDT as the quality constraint for structure prediction speedup is our construction, but the pattern of "speedup at iso-quality" is well-established.

### Precedent: Speedup at Iso-Quality in Diffusion Model Acceleration

The pattern of measuring speedup while constraining quality degradation is standard in diffusion model acceleration research:

1. **DDIM** (Song, Meng, Ermon, ICLR 2021): Demonstrated 10-50x speedup over DDPM by reducing sampling steps, with FID as the quality constraint. The paper explicitly characterizes the quality-speed trade-off curve.
   - Citation: Song J, Meng C, Ermon S. "Denoising Diffusion Implicit Models." *ICLR*, 2021.
   - URL: [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

2. **Consistency Models** (Song et al., ICML 2023): Achieve single-step generation with FID constraints (3.55 on CIFAR-10, 6.20 on ImageNet 64x64 for one-step). The framework explicitly trades number of sampling steps for quality.
   - Citation: Song Y, Dhariwal P, Chen M, Sutskever I. "Consistency Models." *ICML*, 2023.
   - URL: [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)

3. **ESMFold vs AlphaFold2** (Lin et al. 2023): While not framed as "speedup at iso-quality," ESMFold demonstrates a approximately 6x speed improvement over AlphaFold2 with reduced lDDT (0.68 vs 0.85 on CASP14), providing a direct precedent for speed-quality trade-off analysis in protein structure prediction using lDDT as the quality metric.

### Novelty Assessment

The speedup_at_iso_quality metric with an lDDT quality gate is a straightforward composition of established components:
- The speedup ratio is a universal engineering metric
- lDDT is the canonical protein structure quality metric (Mariani et al. 2013)
- The iso-quality constraint pattern is standard in diffusion acceleration (DDIM, consistency models)

We do not claim novelty in the metric design. The contribution is applying it systematically to biomolecular structure prediction inference optimization.

---

## 5. Implementation Notes

### Evaluator Implementation

- **Quality proxy**: pLDDT from Boltz confidence head (complex_plddt field in confidence JSON)
- **True lDDT validation**: OpenStructure `compare-structures` (same tooling as `scripts/eval/run_evals.py`)
- **Timing**: Wall-clock time via `time.perf_counter()` bracketing a subprocess call. Includes all stages from input parsing through output writing.
- **Quality gate**: Mean pLDDT regression <= 2 percentage points (0.02 in [0,1] scale). Additionally, a per-complex floor of 5 pp prevents strong easy-case performance from masking catastrophic failures.
- **Baseline**: 200-step EDM/Karras-style diffusion, 3 recycling steps, `float32_matmul_precision="highest"`, `torch.compile` disabled, `diffusion_samples=1`.

### Reproducibility

- Seeds pinned for all evaluations
- Modal L40S-48GB for consistent hardware (production deployment target)
- OpenStructure version 2.8.0 for true lDDT computation (matching Boltz-1 evaluation)
- Median of 3 runs per complex for timing stability

---

## References

1. Mariani V, Biasini M, Barbato A, Schwede T. "lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests." *Bioinformatics*, 29(21):2722-2728, 2013. [DOI: 10.1093/bioinformatics/btt473](https://doi.org/10.1093/bioinformatics/btt473)

2. Jumper J et al. "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596(7873):583-589, 2021. [DOI: 10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

3. Abramson J et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature*, 630(8016):493-500, 2024. [DOI: 10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)

4. Wohlwend J, Corso G, Passaro S et al. "Boltz-1: Democratizing Biomolecular Interaction Modeling." *bioRxiv*, 2024. [DOI: 10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167)

5. Lin Z et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 379(6637):1123-1130, 2023. [DOI: 10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

6. Passaro S, Corso G, Wohlwend J et al. "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction." *bioRxiv*, 2025. [DOI: 10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)

7. Song J, Meng C, Ermon S. "Denoising Diffusion Implicit Models." *ICLR*, 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

8. Song Y, Dhariwal P, Chen M, Sutskever I. "Consistency Models." *ICML*, 2023. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
