#!/usr/bin/env bash
set -euo pipefail

output=${1:-${SEISPACE_DATA_ROOT:-$root/data}/native_2d/FORGE_78-32_iDASv3-P11_UTC190423213209.sgy}
url=https://pando-rgw01.chpc.utah.edu/silixa_das_apr_23_2019/FORGE_78-32_iDASv3-P11_UTC190423213209.sgy
target_size=153910800
target_sha256=f01a55168b7bd4c55f5b422cd5f506dc60301f7be75cd8816c3218e7e5760bc5

mkdir -p "$(dirname "$output")"
while true; do
  size=$(stat -c%s "$output" 2>/dev/null || echo 0)
  if (( size >= target_size )); then
    break
  fi
  echo "resume at ${size}/${target_size} bytes"
  curl -L -sS --max-time 90 --retry 2 --retry-delay 2 \
    --retry-all-errors -C - -o "$output" "$url" || true
  sleep 2
done
final_size=$(stat -c%s "$output")
if (( final_size != target_size )); then
  echo "unexpected final size ${final_size}" >&2
  exit 1
fi
echo "$target_sha256  $output" | sha256sum --check
ls -lh "$output"
