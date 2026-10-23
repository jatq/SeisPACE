#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

cd "$root"
CUDA_VISIBLE_DEVICES=2 $python scripts/prepare_seisdn_hdf5.py \
  --seisdn-root /data/jatq/jatq_papers/seisDN_check \
  --data-dir data/native_forge2d \
  --output-dir data/seisdn_native_forge2d \
  --num-train 2000 --num-val 200 --seed 42 \
  > runs/native2d_seisdn_prepare_forge2d.log 2>&1 &
forge_pid=$!

CUDA_VISIBLE_DEVICES=3 $python scripts/prepare_seisdn_hdf5.py \
  --seisdn-root /data/jatq/jatq_papers/seisDN_check \
  --data-dir data/native_marmousi2 \
  --output-dir data/seisdn_native_marmousi2 \
  --num-train 2000 --num-val 200 --seed 42 \
  > runs/native2d_seisdn_prepare_marmousi2.log 2>&1 &
marmousi_pid=$!

wait "$forge_pid"
wait "$marmousi_pid"
echo "NATIVE2D_SEISDN_DATA_READY"
