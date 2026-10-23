#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_field"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  baseline_lines=0
  if [[ -f "$run_root/forge2d_baseline/history.csv" ]]; then
    baseline_lines=$(wc -l < "$run_root/forge2d_baseline/history.csv")
  fi
  if (( baseline_lines >= 81 )) && [[ -f "$run_root/forge2d_prior/applied.ok" ]]; then
    break
  fi
  sleep 60
done

mkdir -p "$run_root/forge2d_no_consolidation"
cd "$root"
$python scripts/train_seispace2d.py \
  --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
  --data-dir data/native_forge2d \
  --output-dir "$run_root/forge2d_no_consolidation" \
  --input-mode noisy_prior \
  --init-checkpoint "$run_root/forge2d_baseline/best.pt" \
  --anchor-checkpoint "$run_root/forge2d_baseline/best.pt" \
  --anchor-weight 0.5 \
  --epochs 20 --batch-size 16 --lr 0.0001 \
  --seed 42 --device cuda:3 > "$run_root/forge2d_no_consolidation/train.log" 2>&1

echo "FORGE_NO_CONSOLIDATION_DONE"
