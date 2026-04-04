#!/usr/bin/env bash
set -euo pipefail

base_dir="output/FasterGS"
nsys_dir="/home/kenneth/Documents/NSYS"

for cfg_dir in "$base_dir"/*/; do
  [ -d "$cfg_dir" ] || continue

  cfg_name="$(basename "$cfg_dir")"
  scene="${cfg_name#*_}"
  scene="${scene%%_*}"

  if [ -z "$scene" ] || [ "$scene" = "$cfg_name" ]; then
    echo "Skipping '$cfg_name': could not extract scene name" >&2
    continue
  fi

  out_file="$nsys_dir/base--${scene}--4090"

  echo "Profiling $cfg_name -> $out_file"
  nsys \
    profile -t cuda,nvtx \
    -o "$out_file" \
    --force-overwrite true \
    python ./scripts/inference.py \
    -d "$cfg_dir" \
    -s test \
    -b
done

