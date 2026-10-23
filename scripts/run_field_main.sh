#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_field"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  history="$run_root/forge2d_consolidated/history.csv"
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

cd "$root"
pids=()
for spec in "0.5 a05 cuda:0" "1.0 a10 cuda:1" "2.0 a20 cuda:4"; do
  read -r weight tag device <<< "$spec"
  output="$run_root/forge2d_main_${tag}"
  mkdir -p "$output"
  $python scripts/train_seispace2d.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_forge2d \
    --output-dir "$output" \
    --input-mode noisy_prior \
    --init-checkpoint "$run_root/forge2d_consolidated/best.pt" \
    --anchor-checkpoint "$run_root/forge2d_consolidated/best.pt" \
    --anchor-weight "$weight" \
    --epochs 20 --batch-size 16 --lr 0.0001 \
    --seed 42 --device "$device" > "$output/train.log" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

$python scripts/select_seispace2d_anchor.py \
  --root "$run_root" \
  --candidates 0.5=forge2d_main_a05 1.0=forge2d_main_a10 \
  2.0=forge2d_main_a20 \
  --output "$run_root/anchor_selection.json" \
  --destination "$run_root/forge2d_seispace/best.pt"

echo "FIELD_MAIN_DONE"
