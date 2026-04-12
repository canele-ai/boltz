#!/bin/bash
# Fast Model Load evaluation
# Runs persistent model loading + bypass-lightning optimizations
# 3 seeds in parallel on Modal L40S GPUs

set -euo pipefail

cd "$(dirname "$0")/../.."

# Single seed (quick iteration)
# modal run orbits/fast-model-load/eval_persistent.py --mode eval

# Multi-seed validation (3 seeds)
modal run orbits/fast-model-load/eval_persistent.py --mode eval --validate
