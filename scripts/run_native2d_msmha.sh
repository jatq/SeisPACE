#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

cd "$root"
for pair in forge2d:native_forge2d marmousi2:native_marmousi2; do
  tag=${pair%%:*}
  data_name=${pair##*:}
  output="runs/msmha_native_${tag}"
  mkdir -p "$output"
  $python scripts/train_msmha_wavelet2d.py \
    --data-dir "data/${data_name}" \
    --output-dir "$output" \
    --epochs 100 --batch-size 128 --num-sections 2000 \
    --patches-per-section 8 --patch-size 48 --lr 0.0001 \
    --seed 42 --device cuda:4 > "$output/train.log" 2>&1
done
echo "NATIVE2D_MSMHA_TRAINED"
