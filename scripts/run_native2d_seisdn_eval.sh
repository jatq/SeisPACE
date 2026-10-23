#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

evaluate_source() {
  local config=$1
  local checkpoint=$2
  local data_dir=$3
  local result_dir=$4
  local device=$5
  while [[ ! -f "$checkpoint" ]]; do
    sleep 60
  done
  mkdir -p "$result_dir"
  for pair in m10:test_m10 0:test_0 p10:test_p10; do
    level=${pair%%:*}
    prefix=${pair##*:}
    $python "$root/scripts/evaluate_seisdn_cached2d.py" \
      --seisdn-root /data/jatq/jatq_papers/seisDN_check \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --data-dir "$data_dir" \
      --prefix "$prefix" \
      --num-samples 200 \
      --output "$result_dir/${level}.json" \
      --device "$device"
  done
}

cd "$root"
evaluate_source configs/seisdn_forge2d.yaml \
  runs/seisdn_native_forge2d/forge2d_final_model.pth \
  data/native_forge2d runs/seispace_field/eval_seisdn cuda:2
evaluate_source configs/seisdn_marmousi2.yaml \
  runs/seisdn_native_marmousi2/marmousi2_final_model.pth \
  data/native_marmousi2 runs/seispace_marmousi2/eval_seisdn cuda:3
echo "NATIVE2D_SEISDN_EVAL_DONE"
