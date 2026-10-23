#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python
devices=(cuda:0 cuda:1 cuda:2 cuda:3 cuda:4)

declare -A data_dirs=(
  [forge2d]=data/native_forge2d
  [marmousi2]=data/native_marmousi2
)
declare -A run_dirs=(
  [forge2d]=runs/ps_sgms_native_forge2d
  [marmousi2]=runs/ps_sgms_native_marmousi2
)
declare -A result_dirs=(
  [forge2d]=runs/seispace_field/eval_ps_sgms
  [marmousi2]=runs/seispace_marmousi2/eval_ps_sgms
)

cd "$root"
pids=()
job=0
for tag in forge2d marmousi2; do
  result_dir=${result_dirs[$tag]}
  mkdir -p "$result_dir/validation"
  for iteration in 2500 5000 7500 10000; do
    output="$result_dir/validation/${iteration}.json"
    if [[ -f "$output" ]]; then
      continue
    fi
    device=${devices[$((job % ${#devices[@]}))]}
    $python scripts/evaluate_ps_sgms_wavelet2d.py \
      --ps-root /data/jatq/jatq_papers/PS-SGMs_check \
      --checkpoint "${run_dirs[$tag]}/checkpoint_${iteration}.pt" \
      --config configs/ps_sgms_native2d.yaml \
      --data-dir "${data_dirs[$tag]}" \
      --prefix val \
      --num-samples 40 \
      --clip-range 0 1 \
      --output "$output" \
      --device "$device" > "$result_dir/validation/${iteration}.log" 2>&1 &
    pids+=($!)
    job=$((job + 1))
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

declare -A selected=()
for tag in forge2d marmousi2; do
  result_dir=${result_dirs[$tag]}
  selected[$tag]=$($python scripts/select_json_candidate.py \
    --candidates \
      2500="$result_dir/validation/2500.json" \
      5000="$result_dir/validation/5000.json" \
      7500="$result_dir/validation/7500.json" \
      10000="$result_dir/validation/10000.json" \
    --output "$result_dir/selection.json")
done

pids=()
job=0
for tag in forge2d marmousi2; do
  result_dir=${result_dirs[$tag]}
  checkpoint="${run_dirs[$tag]}/checkpoint_${selected[$tag]}.pt"
  for pair in m10:test_m10 0:test_0 p10:test_p10; do
    level=${pair%%:*}
    prefix=${pair##*:}
    device=${devices[$((job % ${#devices[@]}))]}
    $python scripts/evaluate_ps_sgms_wavelet2d.py \
      --ps-root /data/jatq/jatq_papers/PS-SGMs_check \
      --checkpoint "$checkpoint" \
      --config configs/ps_sgms_native2d.yaml \
      --data-dir "${data_dirs[$tag]}" \
      --prefix "$prefix" \
      --num-samples 200 \
      --clip-range 0 1 \
      --output "$result_dir/${level}.json" \
      --device "$device" > "$result_dir/${level}.log" 2>&1 &
    pids+=($!)
    job=$((job + 1))
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "NATIVE2D_PS_SGMS_EVAL_PARALLEL_DONE"
