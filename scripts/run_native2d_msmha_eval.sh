#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

evaluate_source() {
  local run_dir=$1
  local data_dir=$2
  local result_dir=$3
  while [[ ! -f "$run_dir/final.pt" ]]; do
    sleep 60
  done
  mkdir -p "$result_dir"
  for pair in m10:test_m10 0:test_0 p10:test_p10; do
    level=${pair%%:*}
    prefix=${pair##*:}
    $python "$root/scripts/evaluate_msmha_wavelet2d.py" \
      --checkpoint "$run_dir/best.pt" \
      --data-dir "$data_dir" \
      --prefix "$prefix" \
      --num-samples 200 \
      --patch-size 48 --stride 24 \
      --output "$result_dir/${level}.json" \
      --device cuda:4
  done
}

cd "$root"
evaluate_source runs/msmha_native_forge2d data/native_forge2d \
  runs/seispace_field/eval_msmha
evaluate_source runs/msmha_native_marmousi2 data/native_marmousi2 \
  runs/seispace_marmousi2/eval_msmha
echo "NATIVE2D_MSMHA_EVAL_DONE"
