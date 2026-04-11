# Evaluation Datasets Catalog

## Overview

This document catalogs evaluation datasets and benchmarks used across the published literature for biomolecular structure prediction, with a focus on those relevant to our Boltz-2 inference speedup research. For each dataset, we assess whether to ADOPT (use as-is), ADAPT (modify for our use), or CONSTRUCT (build a new dataset).

---

## 1. Dataset Catalog

### 1.1 CASP15 Targets

| Property | Value |
|----------|-------|
| **Name** | CASP15 (Critical Assessment of Structure Prediction, Round 15) |
| **Year** | 2022 |
| **Source** | [https://predictioncenter.org/casp15/](https://predictioncenter.org/casp15/) |
| **Domain list** | [https://predictioncenter.org/casp15/domains_summary.cgi](https://predictioncenter.org/casp15/domains_summary.cgi) |
| **GitHub mirror** | [Bhattacharya-Lab/CASP15](https://github.com/Bhattacharya-Lab/CASP15) |
| **Size** | 68 full-length tertiary targets, 94 assessment domains (47 TBM, 47 FM); 40 protein assembly targets |
| **Modalities** | Proteins, protein-protein complexes, protein-ligand complexes, RNA, protein-nucleic acid |
| **Used by** | AlphaFold3 (Abramson et al. 2024), Boltz-1 (Wohlwend et al. 2024), Boltz-2 (Passaro et al. 2025), Chai-1 |
| **Canonical** | Yes (>=4 papers) |
| **Metrics reported** | lDDT, lDDT-PLI, GDT_TS, TM-score, DockQ |
| **Cost to adopt** | Low. Targets and experimental structures are publicly available. Requires downloading PDB structures and assembling input YAML files. |
| **Notes** | The standard benchmark for comparing biomolecular structure prediction methods. All Boltz papers evaluate on CASP15. The Boltz-1 paper ran all models with 200 sampling steps and 10 recycling rounds on this set. |

**Decision: ADOPT**

Rationale: CASP15 is the canonical benchmark used by all relevant papers (AlphaFold3, Boltz-1, Boltz-2, Chai-1). Results are directly comparable to published values. The dataset covers multiple modalities (protein, ligand, RNA) matching our evaluation needs.

---

### 1.2 CASP14 Targets

| Property | Value |
|----------|-------|
| **Name** | CASP14 (Critical Assessment of Structure Prediction, Round 14) |
| **Year** | 2020 |
| **Source** | [https://predictioncenter.org/casp14/](https://predictioncenter.org/casp14/) |
| **Size** | 67 targets, 87 protein domains for tertiary structure assessment |
| **Modalities** | Proteins (single-chain only, no complexes) |
| **Used by** | AlphaFold2 (Jumper et al. 2021), ESMFold (Lin et al. 2023), RoseTTAFold |
| **Canonical** | Yes (>=3 papers), but superseded by CASP15 |
| **Metrics reported** | lDDT-Ca, GDT_TS, TM-score |
| **Cost to adopt** | Low. Publicly available. |
| **Notes** | CASP14 was the breakout competition for AlphaFold2 (median GDT_TS = 92.4). However, it predates the multi-modal structure prediction era and does not include protein-ligand or protein-nucleic acid complexes. |

**Decision: DO NOT ADOPT**

Rationale: CASP14 is a protein-only benchmark. Our research targets Boltz-2, which handles proteins, ligands, and nucleic acids. CASP15 is the direct successor and is used by all Boltz papers. Including CASP14 would add cost without providing additional insight for our speedup research.

---

### 1.3 PDB Test Set (Post-Training-Cutoff)

| Property | Value |
|----------|-------|
| **Name** | Held-out PDB test set |
| **Source** | [https://www.rcsb.org/](https://www.rcsb.org/) |
| **Size** | Variable; typically hundreds of structures released after the model's training data cutoff |
| **Modalities** | All (proteins, nucleic acids, ligands, ions, modified residues) |
| **Used by** | AlphaFold3 (Abramson et al. 2024), Boltz-1 (Wohlwend et al. 2024), Boltz-2 (Passaro et al. 2025) |
| **Canonical** | Partially (>=3 papers use PDB test sets, but cutoff dates and filtering criteria differ) |
| **Metrics reported** | lDDT, lDDT-PLI, DockQ, RMSD |
| **Cost to adopt** | Medium. Requires defining the exact cutoff date, applying quality filters (resolution, R-free), removing redundancy, and assembling input files. Each group defines their own filtering, so results are not directly comparable unless the exact set is matched. |
| **Notes** | Boltz-1 and AlphaFold3 both use PDB test sets but with different cutoff dates and filtering criteria. The Boltz-1 paper documents their exact filtering procedure. |

**Decision: ADAPT**

Rationale: A held-out PDB test set provides broader coverage than CASP15 alone and is useful for stress-testing speedup approaches on diverse complexes (varying sizes, modalities). However, the exact set must be defined carefully to match or extend the Boltz evaluation protocol. We should use the same cutoff date and filtering criteria as the Boltz-1/Boltz-2 papers to ensure comparable baselines.

---

### 1.4 CAMEO (Continuous Automated Model Evaluation)

| Property | Value |
|----------|-------|
| **Name** | CAMEO-3D |
| **Source** | [https://cameo3d.org/](https://cameo3d.org/) |
| **Size** | Approximately 100-200 new targets per quarter (rolling weekly releases) |
| **Modalities** | Primarily single-chain proteins |
| **Used by** | ESMFold (Lin et al. 2023), various CASP participants |
| **Canonical** | Partially (used for continuous server evaluation, but not the primary benchmark for Boltz papers) |
| **Metrics reported** | lDDT, TM-score, GDT_TS |
| **Cost to adopt** | Medium. Requires registering a server for automated evaluation, or manually downloading recent targets. |
| **Notes** | CAMEO provides a rolling benchmark that avoids the biennial CASP cycle. Useful for continuous monitoring but not for point-in-time comparisons. No Boltz paper reports CAMEO results. |

**Decision: DO NOT ADOPT**

Rationale: CAMEO is protein-only and not used by any Boltz paper. It is designed for continuous server evaluation, not one-time benchmarking. Adding it would not improve comparability with published results and would add operational complexity.

---

### 1.5 PoseBusters Benchmark

| Property | Value |
|----------|-------|
| **Name** | PoseBusters |
| **Source** | Published alongside the PoseBusters validity checks for molecular docking |
| **Size** | 428 complexes (protein-ligand) |
| **Modalities** | Protein-ligand complexes |
| **Used by** | AlphaFold3 (Abramson et al. 2024) |
| **Canonical** | No (1 paper in our set) |
| **Metrics reported** | Ligand RMSD, validity rates |
| **Cost to adopt** | Medium. |
| **Notes** | Focused specifically on protein-ligand docking quality. AlphaFold3 uses it to benchmark ligand prediction accuracy. |

**Decision: DO NOT ADOPT**

Rationale: PoseBusters is ligand-focused and used by only one paper in our reference set. Our primary metric is overall structural quality (pLDDT), not ligand pose accuracy specifically. If ligand quality emerges as a concern, we can revisit.

---

### 1.6 Our Current Test Set (config.yaml)

| Property | Value |
|----------|-------|
| **Name** | Boltz-2 inference speedup test cases |
| **Source** | `research/eval/test_cases/` |
| **Size** | 3 complexes: small (~200 residues), medium (~400 residues), large (~600 residues) |
| **Modalities** | Protein-protein complexes |
| **Used by** | This research campaign only |
| **Canonical** | No |
| **Metrics reported** | pLDDT (predicted lDDT), iPTM, wall-clock time |
| **Cost to adopt** | Zero (already implemented) |
| **Notes** | Designed for fast iteration during speedup research. Covers the size range relevant to binder design pipelines. Uses pLDDT as quality proxy rather than true lDDT. |

**Decision: ADOPT (as primary iteration set)**

Rationale: This test set is already integrated into the evaluator and covers the size range most relevant to the binder design use case (the motivating application from problem.md). It enables fast iteration during the optimization loop. However, final results should be validated on CASP15 or a PDB test set for publishability.

---

## 2. Summary and Recommendation

### Evaluation Tiers

| Tier | Dataset | Purpose | When to Use |
|------|---------|---------|-------------|
| **Tier 0: Iteration** | Current test cases (3 complexes) | Fast feedback during optimization | Every experiment |
| **Tier 1: Validation** | CASP15 targets (subset) | Confirm quality at iso-quality constraint | Promising configurations |
| **Tier 2: Publication** | Full CASP15 + PDB test set | Publishable comparison with literature | Final results |

### Primary Dataset Decision

**ADOPT: CASP15** as the canonical benchmark for publishable results. This is the dataset used by AlphaFold3, Boltz-1, and Boltz-2. Results on CASP15 are directly comparable to published values.

**ADAPT: PDB test set** as a supplementary validation set, using the Boltz-1/Boltz-2 filtering criteria. This provides broader coverage and stress-tests speedup approaches on diverse complex types and sizes.

**RETAIN: Current 3-complex test set** for fast iteration during the optimization loop. This is not publishable but is essential for rapid development.

### Dataset Adoption Plan

1. **Immediate**: Continue using the 3-complex test set in `research/eval/test_cases/` for all optimization iterations.
2. **Validation**: When a configuration achieves >= 5x speedup on the iteration set, validate on a CASP15 subset (protein targets + protein-ligand targets).
3. **Publication**: For final results, run the full CASP15 set and a held-out PDB test set matching the Boltz-1/2 evaluation protocol.

---

## 3. References

1. CASP15: [https://predictioncenter.org/casp15/](https://predictioncenter.org/casp15/)
2. CASP14: [https://predictioncenter.org/casp14/](https://predictioncenter.org/casp14/)
3. RCSB PDB: [https://www.rcsb.org/](https://www.rcsb.org/)
4. CAMEO: [https://cameo3d.org/](https://cameo3d.org/)
5. Kryshtafovych A et al. "Critical Assessment of Methods of Protein Structure Prediction (CASP)--Round XV." *Proteins*, 91(12):1541-1557, 2023. [DOI: 10.1002/prot.26617](https://doi.org/10.1002/prot.26617)
6. Haas J et al. "Continuous Automated Model EvaluatiOn (CAMEO)--Perspectives on the future of fully automated evaluation of structure prediction methods." *Proteins*, 89(12):1977-1986, 2021. [DOI: 10.1002/prot.26213](https://doi.org/10.1002/prot.26213)
