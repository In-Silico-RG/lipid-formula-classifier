#!/bin/bash
# Script 30 on the workstation, one model after another (GUANE not used; AC 2026-11-17).
# Order: fastest first, so partial results arrive early.
cd "$(dirname "$0")/.."
PY=${PY:-python}
J=${1:-4}
for args in "rf" "knn" "xgb --task=category" "mlp --task=category" "xgb --task=main_class" "mlp --task=main_class"; do
  tag=$(echo "$args" | tr -c 'A-Za-z0-9' _)
  echo "[$(date +%T)] start $args"
  nice -n 10 $PY scripts/30_feature_selection.py $args $J > results/30_${tag}.stdout 2>&1 \
    && echo "[$(date +%T)] OK $args" || echo "[$(date +%T)] FAILED $args (see results/30_${tag}.stdout)"
done
$PY scripts/30_summary.py > /dev/null 2>&1 && echo "[$(date +%T)] summary written: results/30_summary.log"
echo "[$(date +%T)] ALL DONE"
