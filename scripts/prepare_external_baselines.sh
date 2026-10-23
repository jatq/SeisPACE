#!/usr/bin/env bash
set -euo pipefail

root=${SEISPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
third_party="$root/third_party"
mkdir -p "$third_party"

clone_if_missing() {
  local url=$1
  local destination=$2
  if [[ -d "$destination/.git" ]]; then
    echo "Using existing checkout: $destination"
  else
    git clone --depth 1 "$url" "$destination"
  fi
}

clone_if_missing https://github.com/javashs/SCRN.git "$third_party/SCRN"
clone_if_missing https://github.com/Liang-Ding/seisDN.git "$third_party/seisDN"
clone_if_missing https://github.com/mengchuangji/PS-SGMs-seismic.git \
  "$third_party/PS-SGMs-seismic"
clone_if_missing https://github.com/DeepWave-KAUST/MSMHA_Denoising-pub.git \
  "$third_party/MSMHA_Denoising-pub"

echo "External baseline repositories are available under $third_party"
