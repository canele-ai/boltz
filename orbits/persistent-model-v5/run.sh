#!/usr/bin/env bash
# Reproduce persistent-model-v5 evaluation
# Prerequisites: Modal authenticated, boltz-msa-cache-v3 and boltz-model-cache volumes populated
set -euo pipefail

cd "$(dirname "$0")/../.."

# Step 1: Prepare ground truth volume (idempotent)
echo "[run.sh] Step 1: Preparing ground truth volume..."
modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode prep-ground-truth

# Step 2: Run full validation (3 seeds in parallel)
echo "[run.sh] Step 2: Running evaluation (3 seeds)..."
modal run orbits/persistent-model-v5/eval_persistent_v5.py --mode eval --validate
