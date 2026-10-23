#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
repo=/data/jatq/jatq_papers/seisDN_check
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while [[ ! -f "$root/data/seisdn_native_forge2d/train/train.h5" ]] \
  || [[ ! -f "$root/data/seisdn_native_marmousi2/train/train.h5" ]]; do
  sleep 30
done

mkdir -p "$root/runs/seisdn_native_forge2d" \
  "$root/runs/seisdn_native_marmousi2"
cd "$repo/seisDN"
(
  while true; do
    forge_size=0
    marmousi_size=0
    if [[ -f "$root/runs/seisdn_native_forge2d/train.log" ]]; then
      forge_size=$(wc -c < "$root/runs/seisdn_native_forge2d/train.log")
    fi
    if [[ -f "$root/runs/seisdn_native_marmousi2/train.log" ]]; then
      marmousi_size=$(wc -c < "$root/runs/seisdn_native_marmousi2/train.log")
    fi
    echo "forge_log_bytes=${forge_size} marmousi_log_bytes=${marmousi_size}"
    sleep 300
  done
) &
monitor_pid=$!
CUDA_VISIBLE_DEVICES=2 $python train.py \
  --config "$root/configs/seisdn_forge2d.yaml" \
  > "$root/runs/seisdn_native_forge2d/train.log" 2>&1 &
forge_pid=$!

CUDA_VISIBLE_DEVICES=3 $python train.py \
  --config "$root/configs/seisdn_marmousi2.yaml" \
  > "$root/runs/seisdn_native_marmousi2/train.log" 2>&1 &
marmousi_pid=$!

wait "$forge_pid"
wait "$marmousi_pid"
kill "$monitor_pid" 2>/dev/null || true
echo "NATIVE2D_SEISDN_TRAINED"
