#!/bin/bash
# Recycling RMSD study: measure CA RMSD vs PDB ground truth across recycling_steps=0,1,2,3
# All use ODE-12 optimizations (sampling_steps=12, gamma_0=0, matmul_precision=high, bf16_trunk=true)
# --validate runs 3 seeds for timing stability
#
# Run from repo root:
#   cd /home/liambai/code/boltz/.worktrees/recycling-rmsd-v5
#   bash orbits/recycling-rmsd-v5/run.sh

set -e
ORBIT_DIR="orbits/recycling-rmsd-v5"
RESULTS_DIR="${ORBIT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

echo "=== Recycling RMSD Study: 4 configs x 3 seeds each ==="
echo "Running all 4 recycling configs in parallel on Modal..."

# recycling=0
modal run research/eval/evaluator.py \
  --config '{"sampling_steps": 12, "recycling_steps": 0, "matmul_precision": "high", "gamma_0": 0.0, "bf16_trunk": true}' \
  --validate > "${RESULTS_DIR}/recycle_0.json" 2>&1 &
PID0=$!

# recycling=1
modal run research/eval/evaluator.py \
  --config '{"sampling_steps": 12, "recycling_steps": 1, "matmul_precision": "high", "gamma_0": 0.0, "bf16_trunk": true}' \
  --validate > "${RESULTS_DIR}/recycle_1.json" 2>&1 &
PID1=$!

# recycling=2
modal run research/eval/evaluator.py \
  --config '{"sampling_steps": 12, "recycling_steps": 2, "matmul_precision": "high", "gamma_0": 0.0, "bf16_trunk": true}' \
  --validate > "${RESULTS_DIR}/recycle_2.json" 2>&1 &
PID2=$!

# recycling=3
modal run research/eval/evaluator.py \
  --config '{"sampling_steps": 12, "recycling_steps": 3, "matmul_precision": "high", "gamma_0": 0.0, "bf16_trunk": true}' \
  --validate > "${RESULTS_DIR}/recycle_3.json" 2>&1 &
PID3=$!

echo "Waiting for all 4 jobs..."
wait $PID0 && echo "recycle=0 done" || echo "recycle=0 FAILED"
wait $PID1 && echo "recycle=1 done" || echo "recycle=1 FAILED"
wait $PID2 && echo "recycle=2 done" || echo "recycle=2 FAILED"
wait $PID3 && echo "recycle=3 done" || echo "recycle=3 FAILED"

echo "=== All done. Results in ${RESULTS_DIR}/ ==="
