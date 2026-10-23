#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_marmousi2"
data_dir="$root/data/native_marmousi2"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  baseline_lines=0
  if [[ -f "$run_root/baseline/history.csv" ]]; then
    baseline_lines=$(wc -l < "$run_root/baseline/history.csv")
  fi
  if (( baseline_lines >= 81 )) && [[ -f "$run_root/prior/applied.ok" ]]; then
    break
  fi
  sleep 60
done

history_has() {
  local path=$1
  local minimum=$2
  if [[ -f "$path" ]]; then
    (( $(wc -l < "$path") >= minimum ))
  else
    return 1
  fi
}

mkdir -p "$run_root/consolidated" "$run_root/no_consolidation"
cd "$root"
pids=()
if ! history_has "$run_root/consolidated/history.csv" 81 \
  && [[ ! -f "$run_root/consolidated/selected.ok" ]]; then
  $python scripts/train_seispace2d.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir "$data_dir" \
    --output-dir "$run_root/consolidated" \
    --input-mode noisy \
    --init-checkpoint "$run_root/baseline/best.pt" \
    --epochs 80 --batch-size 16 --lr 0.001 \
    --seed 42 --device cuda:4 > "$run_root/consolidated/train.log" 2>&1 &
  pids+=($!)
fi

if ! history_has "$run_root/no_consolidation/history.csv" 21; then
  $python scripts/train_seispace2d.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir "$data_dir" \
    --output-dir "$run_root/no_consolidation" \
    --input-mode noisy_prior \
    --init-checkpoint "$run_root/baseline/best.pt" \
    --anchor-checkpoint "$run_root/baseline/best.pt" \
    --anchor-weight 0.5 \
    --epochs 20 --batch-size 16 --lr 0.0001 \
    --seed 42 --device cuda:0 > "$run_root/no_consolidation/train.log" 2>&1 &
  pids+=($!)
fi

for pid in "${pids[@]}"; do
  wait "$pid"
done

pids=()
for spec in "0.5 a05 cuda:0" "1.0 a10 cuda:1" "2.0 a20 cuda:4"; do
  read -r weight tag device <<< "$spec"
  output="$run_root/main_${tag}"
  mkdir -p "$output"
  if ! history_has "$output/history.csv" 21; then
    $python scripts/train_seispace2d.py \
      --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
      --data-dir "$data_dir" \
      --output-dir "$output" \
      --input-mode noisy_prior \
      --init-checkpoint "$run_root/consolidated/best.pt" \
      --anchor-checkpoint "$run_root/consolidated/best.pt" \
      --anchor-weight "$weight" \
      --epochs 20 --batch-size 16 --lr 0.0001 \
      --seed 42 --device "$device" > "$output/train.log" 2>&1 &
    pids+=($!)
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

$python scripts/select_seispace2d_anchor.py \
  --root "$run_root" \
  --candidates 0.5=main_a05 1.0=main_a10 2.0=main_a20 \
  --output "$run_root/anchor_selection.json" \
  --destination "$run_root/main_selected/best.pt"

echo "MARMOUSI2_TAIL_DONE"
