#!/bin/bash
# Reproduce persistent model + recycling_steps=3 evaluation (Config F)
# Runs 3 seeds in parallel via Modal
set -euo pipefail
cd "$(dirname "$0")/../.."
modal run orbits/persistent-recycle3/eval_recycle3.py --mode eval --validate
