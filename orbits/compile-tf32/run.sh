#!/usr/bin/env bash
# Reproduce compile-tf32 orbit experiments.
# Requires: modal (pip install modal), authenticated to Modal.
# Run from repo root.

set -euo pipefail

MODAL="/home/liambai/code/boltz/.venv/bin/modal"
EVAL="research/eval/evaluator.py"

echo "=== Experiment 1: TF32 alone (matmul_precision=high) ==="
$MODAL run $EVAL --config '{"matmul_precision": "high"}' --validate

echo ""
echo "=== Experiment 2: TF32 + compile_structure ==="
$MODAL run $EVAL --config '{"matmul_precision": "high", "compile_structure": true}' --validate

echo ""
echo "=== Experiment 3: TF32 + compile_confidence ==="
$MODAL run $EVAL --config '{"matmul_precision": "high", "compile_confidence": true}'

echo ""
echo "=== Experiment 4: TF32 + compile_structure + compile_confidence ==="
$MODAL run $EVAL --config '{"matmul_precision": "high", "compile_structure": true, "compile_confidence": true}'
