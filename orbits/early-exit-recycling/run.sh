#!/bin/bash
# Reproduce early-exit recycling experiments
# Must be run from the worktree root: /home/liambai/code/boltz/.worktrees/early-exit-recycling

set -e

MODAL=/home/liambai/code/boltz/.venv/bin/modal
SOLUTION=orbits/early-exit-recycling/solution.py

# Phase 1: Profile convergence (how fast does z converge?)
echo "=== Phase 1: Convergence profiling ==="
$MODAL run $SOLUTION --mode profile --config '{"sampling_steps": 20, "seed": 42}'

# Phase 2: Sweep thresholds (single seed, exploratory)
echo "=== Phase 2: Threshold sweep ==="
$MODAL run $SOLUTION --mode sweep --config '{"sampling_steps": 20, "seed": 42}'

# Phase 3: Validate best config with 3 seeds
# (Uncomment after identifying best threshold)
# echo "=== Phase 3: Validate ==="
# for seed in 42 123 7; do
#     $MODAL run $SOLUTION --mode evaluate \
#         --config "{\"sampling_steps\": 20, \"recycling_steps\": 3, \"early_exit_threshold\": 0.95, \"seed\": $seed}"
# done
