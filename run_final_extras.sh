#!/usr/bin/env bash
# Extra experiments requested at the final review (run from the project root):
#   1. patch placement sweep (evaluation only, ~20 min)
#   2. total-variation weight ablation: retrain the patch with lambda_TV = 0, 1, 5, 10 (~1.1 h each),
#      then test each patch under the detail-destroying conditions (resolution, JPEG).
# Safe to re-run: finished steps are skipped.
set -u
cd "$(dirname "$0")"
PY=/c/Python314/python.exe
mkdir -p runs/extra_logs

if [ ! -f runs/placement/yolov8s/placement_metrics.json ] || ! grep -q '"peak"' runs/placement/yolov8s/placement_metrics.json; then
  "$PY" src/placement_study.py > runs/extra_logs/placement.log 2>&1
fi

for tv in 0 1 5 10; do
  if [ ! -f runs/attack/tv_$tv/attack_metrics.json ]; then
    if [ -f runs/attack/tv_$tv/train_log.json ]; then RES=--resume; else RES=; fi
    "$PY" src/attack_patch.py --tv-weight $tv --name tv_$tv $RES >> runs/extra_logs/tv_$tv.log 2>&1
  fi
  if [ ! -f runs/robustness/tv_$tv/robustness_metrics.json ]; then
    "$PY" src/robustness.py --patch runs/attack/tv_$tv/patch.png --name tv_$tv \
      --factors resolution jpeg_quality >> runs/extra_logs/rob_tv_$tv.log 2>&1
  fi
done
echo ALL_DONE > runs/extra_logs/done.txt
