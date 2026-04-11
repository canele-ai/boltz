#!/usr/bin/env bash
# Reproduce the flash-sdpa orbit experiment.
#
# Prerequisites:
#   - Modal CLI installed and authenticated
#   - boltz repo with research/eval/ directory
#
# Usage:
#   cd <repo_root>
#   bash orbits/flash-sdpa/run.sh
#
# This runs the SDPA (float32) evaluation with 3 seeds in parallel on L40S GPUs.

set -euo pipefail

MODAL=$(command -v modal || echo "/home/liambai/code/boltz/.venv/bin/modal")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "=== Flash SDPA: float32 SDPA evaluation (3 seeds) ==="
"$MODAL" run orbits/flash-sdpa/evaluate_sdpa.py --seeds '42,123,7'

echo ""
echo "=== Flash SDPA: bf16 SDPA evaluation (3 seeds) ==="
"$MODAL" run orbits/flash-sdpa/evaluate_sdpa.py --sdpa-bf16 --seeds '42,123,7'
