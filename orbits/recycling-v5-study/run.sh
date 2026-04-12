#!/bin/bash
# Reproduce the recycling sweep experiment
# Requires: modal auth configured, boltz-msa-cache-v3 volume populated

set -e

cd "$(dirname "$0")/../.."

echo "=== Sanity check ==="
modal run orbits/recycling-v5-study/eval_recycling.py --mode sanity

echo ""
echo "=== Recycling sweep (0,1,2,3) x 3 seeds x 3 test cases ==="
modal run orbits/recycling-v5-study/eval_recycling.py --mode sweep
