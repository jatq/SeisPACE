#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python
config="$root/configs/ps_sgms_native2d.yaml"

cd "$root"
mkdir -p runs/ps_sgms_native_forge2d runs/ps_sgms_native_marmousi2
$python scripts/train_ps_sgms_wavelet2d.py \
  --ps-root /data/jatq/jatq_papers/PS-SGMs_check \
  --config "$config" \
  --data-dir data/native_forge2d \
  --output-dir runs/ps_sgms_native_forge2d \
  --iterations 10000 --batch-size 8 --sections 2000 \
  --device cuda:0 > runs/ps_sgms_native_forge2d/train.log 2>&1 &
forge_pid=$!

$python scripts/train_ps_sgms_wavelet2d.py \
  --ps-root /data/jatq/jatq_papers/PS-SGMs_check \
  --config "$config" \
  --data-dir data/native_marmousi2 \
  --output-dir runs/ps_sgms_native_marmousi2 \
  --iterations 10000 --batch-size 8 --sections 2000 \
  --device cuda:1 > runs/ps_sgms_native_marmousi2/train.log 2>&1 &
marmousi_pid=$!

wait "$forge_pid"
wait "$marmousi_pid"
echo "NATIVE2D_PS_SGMS_TRAINED"
