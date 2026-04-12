#!/bin/bash
# Reproduce compile-diffusion-ode experiment
# Run from repo root: bash orbits/compile-diffusion-ode/run.sh

set -e

# Sanity check
echo "=== Sanity Check ==="
modal run orbits/compile-diffusion-ode/eval_compile.py --mode sanity

# Sweep: compare no-compile vs compile modes (single seed for fast iteration)
echo "=== Sweep (single seed) ==="
modal run orbits/compile-diffusion-ode/eval_compile.py --mode sweep --seed 42

# Validation: sweep across 3 seeds
echo "=== Sweep-seeds (3 seeds, parallel) ==="
modal run orbits/compile-diffusion-ode/eval_compile.py --mode sweep-seeds
