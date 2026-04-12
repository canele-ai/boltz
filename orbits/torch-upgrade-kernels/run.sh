#!/bin/bash
# Reproduce torch-upgrade-kernels experiment
# Requires: modal CLI authenticated, ~30 min GPU time total

set -e

MODAL=/home/liambai/code/boltz/.venv/bin/modal
EVAL=orbits/torch-upgrade-kernels/eval_kernels.py

echo "=== Step 1: Sanity check (verify image + kernel imports) ==="
$MODAL run $EVAL --mode sanity

echo ""
echo "=== Step 2: ODE-20/0r + kernels ON (validated, 3 runs) ==="
$MODAL run $EVAL --mode eval \
  --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "enable_kernels": true}' \
  --validate

echo ""
echo "=== Step 3: ODE-20/0r + kernels OFF (validated, 3 runs) ==="
$MODAL run $EVAL --mode eval \
  --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "enable_kernels": false}' \
  --validate

echo ""
echo "=== Step 4: ODE-20/0r + kernels ON + TF32 (validated, 3 runs) ==="
$MODAL run $EVAL --mode eval \
  --config '{"sampling_steps": 20, "recycling_steps": 0, "gamma_0": 0.0, "enable_kernels": true, "matmul_precision": "high"}' \
  --validate

echo ""
echo "=== Step 5: 200s/3r + kernels ON (validated, 3 runs) ==="
$MODAL run $EVAL --mode eval \
  --config '{"sampling_steps": 200, "recycling_steps": 3, "gamma_0": 0.8, "enable_kernels": true}' \
  --validate

echo ""
echo "=== Step 6: 200s/3r + kernels OFF (validated, 3 runs) ==="
$MODAL run $EVAL --mode eval \
  --config '{"sampling_steps": 200, "recycling_steps": 3, "gamma_0": 0.8, "enable_kernels": false}' \
  --validate

echo ""
echo "=== Step 7: Generate figure ==="
/home/liambai/code/boltz/.venv/bin/python3 orbits/torch-upgrade-kernels/make_figure.py

echo ""
echo "=== Done ==="
