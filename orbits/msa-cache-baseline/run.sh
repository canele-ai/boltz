#!/bin/bash
# Reproduce MSA caching baseline experiment
# Requires: Modal CLI authenticated, boltz-msa-cache volume exists
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODAL="${REPO_ROOT}/.venv/bin/modal"

cd "$REPO_ROOT"

echo "=== Phase 1: Cache MSAs (skip if volume already populated) ==="
$MODAL run orbits/msa-cache-baseline/eval_cached.py --phase cache-msas

echo ""
echo "=== Phase 2: Evaluate baseline + winner with cached MSAs (3 runs each) ==="
$MODAL run orbits/msa-cache-baseline/eval_cached.py --phase eval-both --num-runs 3

echo ""
echo "=== Phase 3: Generate comparison figure ==="
cd "$SCRIPT_DIR"
python3 make_figure.py

echo ""
echo "Done. Results in orbits/msa-cache-baseline/"
