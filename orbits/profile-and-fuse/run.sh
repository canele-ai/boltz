#!/bin/bash
# Reproduce the profile-and-fuse orbit experiments.
# All experiments run on Modal with L40S GPU.

set -e

cd "$(dirname "$0")/../.."

echo "=== Phase 1: Sanity check ==="
modal run orbits/profile-and-fuse/eval_sdpa.py --mode sanity

echo ""
echo "=== Phase 2: Single-run sweep (SDPA vs baseline-stacked) ==="
modal run orbits/profile-and-fuse/eval_sdpa.py --mode sweep

echo ""
echo "=== Phase 3: Validated sweep (3 runs per config) ==="
modal run orbits/profile-and-fuse/eval_sdpa.py --mode sweep --validate
