#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

run_level() {
  local data_dir=$1
  local output_dir=$2
  local prefix=$3
  local level=$4
  mkdir -p "$output_dir"
  $python "$root/scripts/evaluate_native2d_classical.py" \
    --data-dir "$data_dir" \
    --prefix "$prefix" \
    --output "$output_dir/${level}.json"
}

cd "$root"
pids=()
run_level data/native_forge2d runs/seispace_field/eval_classical test_m10 m10 &
pids+=($!)
run_level data/native_forge2d runs/seispace_field/eval_classical test_0 0 &
pids+=($!)
run_level data/native_forge2d runs/seispace_field/eval_classical test_p10 p10 &
pids+=($!)
run_level data/native_marmousi2 runs/seispace_marmousi2/eval_classical test_m10 m10 &
pids+=($!)
run_level data/native_marmousi2 runs/seispace_marmousi2/eval_classical test_0 0 &
pids+=($!)
run_level data/native_marmousi2 runs/seispace_marmousi2/eval_classical test_p10 p10 &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "NATIVE2D_CLASSICAL_DONE"
