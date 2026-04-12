#!/bin/bash
# Reproduce compile-noguard experiments
# Run from repo root: bash orbits/compile-noguard/run.sh

set -e

echo "=== compile-noguard: torch.compile without inference guards ==="

# Experiment 1: Baseline comparison (parent orbit config, no compile)
echo ""
echo "--- Experiment 1: Parent baseline (ODE+TF32+bf16, no compile) ---"
modal run orbits/compile-noguard/run_eval.py \
    --seeds '42,123,7' \
    --compile-targets '' \
    --sampling-steps 20 \
    --recycling-steps 0 \
    --gamma-0 0.0

# Experiment 2: Compile pairformer + structure (default mode)
echo ""
echo "--- Experiment 2: Compile pairformer+structure (default) ---"
modal run orbits/compile-noguard/run_eval.py \
    --seeds '42,123,7' \
    --compile-targets 'pairformer,structure' \
    --compile-mode default \
    --sampling-steps 20 \
    --recycling-steps 0 \
    --gamma-0 0.0

# Experiment 3: Compile all modules (default mode)
echo ""
echo "--- Experiment 3: Compile all modules (default) ---"
modal run orbits/compile-noguard/run_eval.py \
    --seeds '42,123,7' \
    --compile-targets 'pairformer,structure,msa,confidence' \
    --compile-mode default \
    --sampling-steps 20 \
    --recycling-steps 0 \
    --gamma-0 0.0

# Experiment 4: Compile with reduce-overhead mode (CUDA graphs)
echo ""
echo "--- Experiment 4: Compile pairformer+structure (reduce-overhead) ---"
modal run orbits/compile-noguard/run_eval.py \
    --seeds '42,123,7' \
    --compile-targets 'pairformer,structure' \
    --compile-mode reduce-overhead \
    --sampling-steps 20 \
    --recycling-steps 0 \
    --gamma-0 0.0

echo ""
echo "=== Done ==="
