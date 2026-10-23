#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

select_for_source() {
  local tag=$1
  local data_dir=$2
  local run_dir=$3
  local result_dir=$4
  mkdir -p "$result_dir/validation"
  local candidates=()
  for iteration in 2500 5000 7500 10000; do
    local checkpoint="$run_dir/checkpoint_${iteration}.pt"
    while [[ ! -f "$checkpoint" ]]; do
      sleep 60
    done
    $python "$root/scripts/evaluate_ps_sgms_wavelet2d.py" \
      --ps-root /data/jatq/jatq_papers/PS-SGMs_check \
      --checkpoint "$checkpoint" \
      --config "$root/configs/ps_sgms_native2d.yaml" \
      --data-dir "$data_dir" \
      --prefix val \
      --num-samples 40 \
      --clip-range 0 1 \
      --output "$result_dir/validation/${iteration}.json" \
      --device cuda:4
    candidates+=("${iteration}=$result_dir/validation/${iteration}.json")
  done
  selected_iteration=$($python "$root/scripts/select_json_candidate.py" \
    --candidates "${candidates[@]}" \
    --output "$result_dir/selection.json")
  selected_checkpoint="$run_dir/checkpoint_${selected_iteration}.pt"
  for pair in m10:test_m10 0:test_0 p10:test_p10; do
    level=${pair%%:*}
    prefix=${pair##*:}
    $python "$root/scripts/evaluate_ps_sgms_wavelet2d.py" \
      --ps-root /data/jatq/jatq_papers/PS-SGMs_check \
      --checkpoint "$selected_checkpoint" \
      --config "$root/configs/ps_sgms_native2d.yaml" \
      --data-dir "$data_dir" \
      --prefix "$prefix" \
      --num-samples 200 \
      --clip-range 0 1 \
      --output "$result_dir/${level}.json" \
      --device cuda:4
  done
  echo "${tag}_PS_SGMS_EVAL_DONE"
}

cd "$root"
select_for_source forge2d \
  data/native_forge2d runs/ps_sgms_native_forge2d \
  runs/seispace_field/eval_ps_sgms
select_for_source marmousi2 \
  data/native_marmousi2 runs/ps_sgms_native_marmousi2 \
  runs/seispace_marmousi2/eval_ps_sgms

echo "NATIVE2D_PS_SGMS_EVAL_DONE"
