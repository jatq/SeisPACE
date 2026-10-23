#!/usr/bin/env bash
set -euo pipefail

seed=$1
gpu=$2
root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_ablation/seeds/2d_seed${seed}"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  history="$run_root/baseline/history.csv"
  if [[ -f "$history" ]]; then
    lines=$(wc -l < "$history")
  else
    lines=0
  fi
  if (( lines >= 81 )); then
    break
  fi
  sleep 60
done

mkdir -p "$run_root/consolidated" "$run_root/main"
cd "$root"
$python scripts/train_seispace2d.py \
  --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
  --data-dir data/seispace2d \
  --output-dir "$run_root/consolidated" \
  --input-mode noisy \
  --init-checkpoint "$run_root/baseline/best.pt" \
  --epochs 80 --batch-size 16 --lr 0.001 \
  --seed "$seed" --device "$gpu" > "$run_root/consolidated/train.log" 2>&1

$python scripts/train_seispace2d.py \
  --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
  --data-dir data/seispace2d \
  --output-dir "$run_root/main" \
  --init-checkpoint "$run_root/consolidated/best.pt" \
  --anchor-checkpoint "$run_root/consolidated/best.pt" \
  --anchor-weight 0.5 \
  --epochs 20 --batch-size 16 --lr 0.0001 \
  --seed "$seed" --device "$gpu" > "$run_root/main/train.log" 2>&1

echo "SEED_${seed}_TAIL_DONE"
