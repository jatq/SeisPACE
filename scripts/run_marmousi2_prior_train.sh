#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_marmousi2"
field_marker="$root/runs/seispace_field/forge2d_prior/applied.ok"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while [[ ! -f "$field_marker" ]]; do
  sleep 60
done

mkdir -p "$run_root/prior"
cd "$root"
$python scripts/train_prior2d.py \
  --config configs/prior2d_wavelet.json \
  --data-dir data/native_marmousi2 \
  --output-dir "$run_root/prior" \
  --epochs 80 --batch-size 8 --device cuda:4 \
  > "$run_root/prior/train.log" 2>&1

echo "MARMOUSI2_PRIOR_TRAINED"
