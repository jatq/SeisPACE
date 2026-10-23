#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_marmousi2"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  no_consolidation_lines=0
  if [[ -f "$run_root/no_consolidation/history.csv" ]]; then
    no_consolidation_lines=$(wc -l < "$run_root/no_consolidation/history.csv")
  fi
  if [[ -f "$run_root/main_selected/best.pt" ]] \
    && (( no_consolidation_lines >= 21 )); then
    break
  fi
  sleep 60
done

mkdir -p "$run_root/eval_baseline" "$run_root/eval_consolidated" \
  "$run_root/eval_no_consolidation" "$run_root/eval_seispace" \
  "$run_root/eval_prior"
cd "$root"
for pair in m10:test_m10 0:test_0 p10:test_p10; do
  level=${pair%%:*}
  prefix=${pair##*:}
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_marmousi2 \
    --prefix "$prefix" \
    --refiner-checkpoint "$run_root/baseline/best.pt" \
    --input-mode noisy \
    --output "$run_root/eval_baseline/${level}.json" \
    --device cuda:4
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_marmousi2 \
    --prefix "$prefix" \
    --refiner-checkpoint "$run_root/consolidated/best.pt" \
    --input-mode noisy \
    --output "$run_root/eval_consolidated/${level}.json" \
    --device cuda:4
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_marmousi2 \
    --prefix "$prefix" \
    --refiner-checkpoint "$run_root/no_consolidation/best.pt" \
    --input-mode noisy_prior \
    --output "$run_root/eval_no_consolidation/${level}.json" \
    --device cuda:4
  $python scripts/evaluate_seispace2d_cached.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --data-dir data/native_marmousi2 \
    --prefix "$prefix" \
    --refiner-checkpoint "$run_root/main_selected/best.pt" \
    --input-mode noisy_prior \
    --output "$run_root/eval_seispace/${level}.json" \
    --save-example "$run_root/eval_seispace/example_${level}.npz" \
    --device cuda:4
  $python scripts/evaluate_cached_prior2d.py \
    --data-dir data/native_marmousi2 \
    --prefix "$prefix" \
    --output "$run_root/eval_prior/${level}.json"
done

$python scripts/summarize_native2d_field.py \
  --root "$run_root" \
  --baseline-dir eval_baseline \
  --no-prior-dir eval_consolidated \
  --no-consolidation-dir eval_no_consolidation \
  --seispace-dir eval_seispace \
  --prior-dir eval_prior \
  --output-prefix marmousi2_summary
echo "MARMOUSI2_EVAL_DONE"
