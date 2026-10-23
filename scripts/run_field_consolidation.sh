#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_field"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  history="$run_root/forge2d_baseline/history.csv"
  if [[ -f "$history" ]]; then
    lines=$(wc -l < "$history")
  else
    lines=0
  fi
  if (( lines >= 81 )) && [[ -f "$run_root/forge2d_prior/applied.ok" ]]; then
    break
  fi
  sleep 60
done

mkdir -p "$run_root/forge2d_consolidated"
cd "$root"
$python scripts/train_seispace2d.py \
  --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
  --data-dir data/native_forge2d \
  --output-dir "$run_root/forge2d_consolidated" \
  --input-mode noisy \
  --init-checkpoint "$run_root/forge2d_baseline/best.pt" \
  --epochs 80 --batch-size 16 --lr 0.001 \
  --seed 42 --device cuda:2 > "$run_root/forge2d_consolidated/train.log" 2>&1

echo "FIELD_CONSOLIDATION_DONE"
