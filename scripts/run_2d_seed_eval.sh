#!/usr/bin/env bash
set -euo pipefail

seed=$1
gpu=$2
root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_ablation/seeds/2d_seed${seed}"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  history="$run_root/main/history.csv"
  if [[ -f "$history" ]]; then
    lines=$(wc -l < "$history")
  else
    lines=0
  fi
  if (( lines >= 21 )); then
    break
  fi
  sleep 60
done

mkdir -p "$run_root/eval_baseline" "$run_root/eval_main"
cd "$root"
for pair in -10:m10 0:0 10:p10; do
  snr=${pair%%:*}
  level=${pair##*:}
  $python scripts/evaluate_seispace2d.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --prior-config configs/prior2d_wavelet.json \
    --prior-checkpoint runs/prior2d_wavelet/best.pt \
    --refiner-checkpoint "$run_root/baseline/best.pt" \
    --test-mat ${SEISPACE_DATA_ROOT:-$root/data}/simulate_wavelet/testing_data.mat \
    --output "$run_root/eval_baseline/${level}.json" \
    --input-mode noisy --num-samples 100 --snr-db "$snr" \
    --device "$gpu" >> "$run_root/eval_baseline/evaluate.log" 2>&1
  $python scripts/evaluate_seispace2d.py \
    --scrn-root ${SEISPACE_SCRN_ROOT:-$root/third_party/SCRN} \
    --prior-config configs/prior2d_wavelet.json \
    --prior-checkpoint runs/prior2d_wavelet/best.pt \
    --refiner-checkpoint "$run_root/main/best.pt" \
    --test-mat ${SEISPACE_DATA_ROOT:-$root/data}/simulate_wavelet/testing_data.mat \
    --output "$run_root/eval_main/${level}.json" \
    --input-mode noisy_prior --num-samples 100 --snr-db "$snr" \
    --device "$gpu" >> "$run_root/eval_main/evaluate.log" 2>&1
done

echo "SEED_${seed}_EVAL_DONE"
