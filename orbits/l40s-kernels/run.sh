#!/bin/bash
# Reproduce the L40S kernel investigation experiments.
#
# Prerequisites: modal CLI authenticated, boltz venv available
#
# This script runs three experiments:
# 1. GPU profiling (op-level timing of TF32/bf16 on triangular ops)
# 2. Standard evaluator with 20s/0r + TF32 (3 runs)
# 3. Standard evaluator with 20s/0r + highest precision (3 runs, for comparison)

set -euo pipefail

MODAL="${MODAL:-/home/liambai/code/boltz/.venv/bin/modal}"
WORKTREE="/home/liambai/code/boltz/.worktrees/l40s-kernels"

echo "=== Step 1: GPU profiling ==="
cd "$WORKTREE"
$MODAL run orbits/l40s-kernels/profile_evaluator.py

echo ""
echo "=== Step 2: Evaluator with TF32 (high precision) ==="
$MODAL run research/eval/evaluator.py \
  --config '{"sampling_steps": 20, "recycling_steps": 0, "matmul_precision": "high"}' \
  --validate

echo ""
echo "=== Step 3: Evaluator with highest precision ==="
$MODAL run research/eval/evaluator.py \
  --config '{"sampling_steps": 20, "recycling_steps": 0, "matmul_precision": "highest"}' \
  --validate

echo ""
echo "=== Step 4: Generate figure ==="
/home/liambai/code/boltz/.venv/bin/python orbits/l40s-kernels/make_figure.py

echo ""
echo "Done. Results in orbits/l40s-kernels/"
