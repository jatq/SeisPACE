#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_field"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  no_consolidation_history="$run_root/forge2d_no_consolidation/history.csv"
  if [[ -f "$no_consolidation_history" ]]; then
    no_consolidation_lines=$(wc -l < "$no_consolidation_history")
  else
    no_consolidation_lines=0
  fi
  if [[ -f "$run_root/forge2d_seispace/best.pt" ]] \
    && (( no_consolidation_lines >= 21 )); then
    break
  fi
  sleep 60
done

mkdir -p "$run_root/eval_forge2d_baseline" \
  "$run_root/eval_forge2d_consolidated" \
  "$run_root/eval_forge2d_no_consolidation" \
  "$run_root/eval_forge2d_seispace" \
  "$run_root/eval_prior"
cd "$root"

evaluate_level() {
  local level=$1
  local prefix=$2
  local device=$3
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_forge2d \
    --prefix "test_${level}" \
    --refiner-checkpoint "$run_root/forge2d_baseline/best.pt" \
    --input-mode noisy \
    --output "$run_root/eval_forge2d_baseline/${level}.json" \
    --device "$device"
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_forge2d \
    --prefix "test_${level}" \
    --refiner-checkpoint "$run_root/forge2d_consolidated/best.pt" \
    --input-mode noisy \
    --output "$run_root/eval_forge2d_consolidated/${level}.json" \
    --device "$device"
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_forge2d \
    --prefix "test_${level}" \
    --refiner-checkpoint "$run_root/forge2d_no_consolidation/best.pt" \
    --input-mode noisy_prior \
    --output "$run_root/eval_forge2d_no_consolidation/${level}.json" \
    --device "$device"
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_forge2d \
    --prefix "test_${level}" \
    --refiner-checkpoint "$run_root/forge2d_seispace/best.pt" \
    --input-mode noisy_prior \
    --output "$run_root/eval_forge2d_seispace/${level}.json" \
    --save-example "$run_root/eval_forge2d_seispace/example_${level}.npz" \
    --device "$device"
  $python scripts/evaluate_cached_prior2d.py \
    --data-dir data/native_forge2d \
    --prefix "test_${level}" \
    --output "$run_root/eval_prior/${level}.json"
}

evaluate_level m10 test_m10 cuda:0 &
pids=($!)
evaluate_level 0 test_0 cuda:1 &
pids+=($!)
evaluate_level p10 test_p10 cuda:2 &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

$python scripts/summarize_native2d_field.py \
  --root "$run_root" \
  --prior-dir eval_prior \
  --output-prefix forge2d_summary
echo "FIELD_EVAL_DONE"
