#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
study="results/eight-task-study-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$study"
for block in original extension; do
  if [ "$block" = original ]; then
    selection=configs/tasks.json
    dataset=data/tasks
    pilot_task=medium_02
  else
    selection=configs/tasks-extension.json
    dataset=data/tasks-extension
    pilot_task=medium_04
  fi
  .venv/bin/python -m benchmark.experiment --config configs/experiment-format-v2.json \
    --selection "$selection" --dataset "$dataset" --phase pilot --task "$pilot_task" \
    --max-attempts 3 --output "$study/$block-pilot" | tee "$study/$block-pilot.log"
  .venv/bin/python -m benchmark.experiment --config configs/experiment-format-v2.json \
    --selection "$selection" --dataset "$dataset" --phase main \
    --pilot "$study/$block-pilot" --output "$study/$block-main" | tee "$study/$block-main.log"
done
printf 'Study complete: %s\n' "$study"
