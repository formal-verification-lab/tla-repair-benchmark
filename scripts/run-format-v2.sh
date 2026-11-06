#!/bin/sh
# Run a fresh compatible pilot, then the separately recorded 36-branch series.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
pilot_output="results/pilot-format-v2-$(date -u +%Y%m%dT%H%M%SZ)-$$"
.venv/bin/python -m benchmark.experiment \
  --config configs/experiment-format-v2.json \
  --phase pilot --task medium_02 --max-attempts 3 --output "$pilot_output"
exec .venv/bin/python -m benchmark.experiment \
  --config configs/experiment-format-v2.json --phase main --pilot "$pilot_output"
