#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
run_root="$root/runs/seispace_marmousi2"
data_dir="$root/data/native_marmousi2"
python=/data/jatq/tools/miniconda3/envs/torch/bin/python

while true; do
  history="$run_root/prior/history.csv"
  if [[ -f "$history" ]]; then
    lines=$(wc -l < "$history")
  else
    lines=0
  fi
  if (( lines >= 81 )); then
    break
  fi
  sleep 60
done

cd "$root"
$python scripts/prepare_cached_prior2d.py \
  --data-dir "$data_dir" \
  --config configs/prior2d_wavelet.json \
  --checkpoint "$run_root/prior/best.pt" \
  --device cuda:4 > "$run_root/prepare_cached_prior.log" 2>&1

touch "$run_root/prior/applied.ok"
echo "MARMOUSI2_PRIOR_APPLIED"
