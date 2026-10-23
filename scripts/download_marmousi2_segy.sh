#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:-${SEISPACE_DATA_ROOT:-$root/data}/native_2d/marmousi2}
page=https://www.agl.uh.edu/downloads/downloads.htm
synthetic_url=https://www.agl.uh.edu/downloads/segy_time/SYNTHETIC_time.segy.zip
velocity_url=https://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz
synthetic_zip_sha256=c5ae44815a6fcfaca75621e5a46abc781c75dddcbaffd9c18784489b00510ee6
synthetic_segy_sha256=3843344f58657f69377fad448bfd9c6f3d24f86ec3b367d04699759d3151de51
velocity_gz_sha256=2c7a0c4170c95a15620c2146573711c61a4cef8dc27338ef6c211a81ffa68920
velocity_segy_sha256=2722a5de0645d8fcf5750f380893d4875383af9b24988ece0bd07b943f5b0c91

mkdir -p "$output_dir"

fetch() {
  local url=$1
  local output=$2
  local expected=$3
  if [[ -f "$output" ]] && \
    [[ $(sha256sum "$output" | cut -d' ' -f1) == "$expected" ]]; then
    return
  fi
  curl -L --fail --retry 4 --retry-all-errors -C - -o "$output" "$url"
  echo "$expected  $output" | sha256sum --check --status
}

fetch "$synthetic_url" "$output_dir/SYNTHETIC_time.segy.zip" "$synthetic_zip_sha256"
fetch "$velocity_url" "$output_dir/vp_marmousi-ii.segy.gz" "$velocity_gz_sha256"

unzip -oq "$output_dir/SYNTHETIC_time.segy.zip" -d "$output_dir"
gzip -dkf "$output_dir/vp_marmousi-ii.segy.gz"

echo "$synthetic_segy_sha256  $output_dir/SYNTHETIC_time.segy" \
  | sha256sum --check --status
echo "$velocity_segy_sha256  $output_dir/vp_marmousi-ii.segy" \
  | sha256sum --check --status

cat <<EOF
Downloaded Marmousi2 SEGY sources from:
$page

Synthetic time section:
$output_dir/SYNTHETIC_time.segy

Velocity model:
$output_dir/vp_marmousi-ii.segy
EOF
