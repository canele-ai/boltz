#!/bin/bash
# Reproduce dead-end architecture evaluation
# Tests layer pruning (DT-8, PF-48) and token merging (10%)
# stacked on winning config (ODE-12 + TF32 + bf16 + bypass + recycle=3)
#
# Requirements: Modal auth, boltz-msa-cache-v5 and boltz-ground-truth-v1 volumes
#
# Usage:
#   cd /path/to/boltz
#   modal run orbits/deadend-architecture/eval_deadend.py
set -euo pipefail

cd "$(dirname "$0")/../.."
modal run orbits/deadend-architecture/eval_deadend.py
