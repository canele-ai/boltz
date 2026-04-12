#!/bin/bash
# Reproduce the recycle-warmstart experiment.
# Compares recycling_steps=0 vs 1 with ODE-12 + bypass wrapper on Modal.
set -e
cd "$(dirname "$0")/../.."
modal run orbits/recycle-warmstart/eval_recycle.py --mode compare
