#!/usr/bin/env bash
# Reproduce lightning-strip experiments
# Run from repo root

set -euo pipefail

echo "=== Sanity check ==="
modal run orbits/lightning-strip/eval_nolightning.py --mode sanity

echo ""
echo "=== Single eval (nolightning, best config) ==="
modal run orbits/lightning-strip/eval_nolightning.py --mode eval

echo ""
echo "=== Multi-seed eval (3 seeds in parallel) ==="
modal run orbits/lightning-strip/eval_nolightning.py --mode multi

echo ""
echo "=== Head-to-head comparison (same GPU) ==="
modal run orbits/lightning-strip/eval_nolightning.py --mode compare --validate
