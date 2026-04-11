# boltz

Can we achieve a 5x or greater GPU inference speedup for Boltz-2 structure prediction while keeping per-complex lDDT within 2 percentage points of the baseline (200-step EDM/Karras diffusion, 3 recycling steps), measured on the CASP15 / held-out PDB test set?

## Problem

See [`research/problem.md`](research/problem.md) for the full problem definition and [`research/background.md`](research/background.md) for motivation and related work.

## Metric

- **Name:** `name: speedup_at_iso_quality`
- **Direction:** <tbd>
- **Provenance:** [`research/eval/metric_provenance.md`](research/eval/metric_provenance.md) — literature review with ≥3 published references (Step 3.0 of `/launch`).
- **Dataset:** [`research/eval/datasets.md`](research/eval/datasets.md) — adoption decision (ADOPT / ADAPT / CONSTRUCT).
- **Baseline statistics & significance threshold:** [`research/eval/config.yaml`](research/eval/config.yaml) under `significance:`.

## Quick Start

```bash
uv sync
python research/eval/evaluator.py --sanity-check
```

## Structure

```
research/
  problem.md             # Problem definition
  background.md          # Motivation + related work (Phase 2.5)
  style.md               # Plotting conventions (read by every orbit)
  eval/
    evaluator.py         # Frozen evaluation harness (eval-v1)
    config.yaml          # Metric, baselines, sanity checks
    metric_provenance.md # Literature backing for the metric (Step 3.0)
    datasets.md          # Dataset adoption decision (Step 3.0)
  figures/               # Teaser, baseline results, eval pipeline
orbits/                  # Agent experiment branches
docs/                    # GitHub Pages visualization
```

## Campaign

Tracked on GitHub Issues: [Campaign issue](see GitHub Issues).

Run `/research` to start the autonomous search loop.
