#!/bin/bash
# Reproduce bf16-trunk experiment from scratch
# Run from repository root

set -e

echo "=== BF16 Trunk: Removing FP32 Upcast in Triangular Multiply ==="
echo ""

# Phase 1: Quality check - bf16 tri_mult + ODE-20/0r (1 run)
echo "--- Phase 1: Quality check (1 run) ---"
modal run orbits/bf16-trunk/eval_bf16.py \
    --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "bf16_trunk": true}'

echo ""
echo "--- Phase 2: bf16 tri_mult + OPM (1 run) ---"
modal run orbits/bf16-trunk/eval_bf16.py \
    --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "bf16_trunk": true, "bf16_opm": true}'

echo ""
echo "--- Phase 3: Control - fp32 trunk + ODE-20/0r (1 run) ---"
modal run orbits/bf16-trunk/eval_bf16.py \
    --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0}'

echo ""
echo "--- Phase 4: Validate best config (3 runs) ---"
modal run orbits/bf16-trunk/eval_bf16.py \
    --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "bf16_trunk": true}' \
    --validate

echo ""
echo "=== Done ==="
