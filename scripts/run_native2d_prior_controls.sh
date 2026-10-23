#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

run_control() {
  local data_dir=$1
  local checkpoint=$2
  local output_root=$3
  local prefix=$4
  local level=$5
  local mode=$6
  local seed=$7
  local device=$8
  mkdir -p "$output_root"
  $python "$root/scripts/evaluate_seispace2d_cached.py" \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir "$data_dir" \
    --prefix "$prefix" \
    --refiner-checkpoint "$checkpoint" \
    --input-mode noisy_prior \
    --prior-mode "$mode" \
    --prior-seed "$seed" \
    --output "$output_root/${level}.json" \
    --device "$device"
}

cd "$root"
pids=()
run_control data/native_forge2d \
  runs/seispace_field/forge2d_seispace/best.pt \
  runs/seispace_field/eval_zero_prior test_m10 m10 zero 2026 cuda:0 &
pids+=($!)
run_control data/native_forge2d \
  runs/seispace_field/forge2d_seispace/best.pt \
  runs/seispace_field/eval_zero_prior test_0 0 zero 2026 cuda:1 &
pids+=($!)
run_control data/native_forge2d \
  runs/seispace_field/forge2d_seispace/best.pt \
  runs/seispace_field/eval_zero_prior test_p10 p10 zero 2026 cuda:2 &
pids+=($!)
run_control data/native_forge2d \
  runs/seispace_field/forge2d_seispace/best.pt \
  runs/seispace_field/eval_shuffled_prior test_m10 m10 shuffled 2026 cuda:3 &
pids+=($!)
run_control data/native_forge2d \
  runs/seispace_field/forge2d_seispace/best.pt \
  runs/seispace_field/eval_shuffled_prior test_0 0 shuffled 2026 cuda:4 &
pids+=($!)
run_control data/native_forge2d \
  runs/seispace_field/forge2d_seispace/best.pt \
  runs/seispace_field/eval_shuffled_prior test_p10 p10 shuffled 2026 cuda:0 &
pids+=($!)

run_control data/native_marmousi2 \
  runs/seispace_marmousi2/main_selected/best.pt \
  runs/seispace_marmousi2/eval_zero_prior test_m10 m10 zero 2026 cuda:1 &
pids+=($!)
run_control data/native_marmousi2 \
  runs/seispace_marmousi2/main_selected/best.pt \
  runs/seispace_marmousi2/eval_zero_prior test_0 0 zero 2026 cuda:2 &
pids+=($!)
run_control data/native_marmousi2 \
  runs/seispace_marmousi2/main_selected/best.pt \
  runs/seispace_marmousi2/eval_zero_prior test_p10 p10 zero 2026 cuda:3 &
pids+=($!)
run_control data/native_marmousi2 \
  runs/seispace_marmousi2/main_selected/best.pt \
  runs/seispace_marmousi2/eval_shuffled_prior test_m10 m10 shuffled 2026 cuda:4 &
pids+=($!)
run_control data/native_marmousi2 \
  runs/seispace_marmousi2/main_selected/best.pt \
  runs/seispace_marmousi2/eval_shuffled_prior test_0 0 shuffled 2026 cuda:0 &
pids+=($!)
run_control data/native_marmousi2 \
  runs/seispace_marmousi2/main_selected/best.pt \
  runs/seispace_marmousi2/eval_shuffled_prior test_p10 p10 shuffled 2026 cuda:1 &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "NATIVE2D_PRIOR_CONTROLS_DONE"
