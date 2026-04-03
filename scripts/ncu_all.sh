#!/usr/bin/env bash
set -euo pipefail

base_dir="output/FasterGS"
ncu_dir="/home/kenneth/Documents/NCU"

for cfg_dir in "$base_dir"/*/; do
  [ -d "$cfg_dir" ] || continue

  cfg_name="$(basename "$cfg_dir")"
  scene="${cfg_name#*_}"
  scene="${scene%%_*}"

  if [ -z "$scene" ] || [ "$scene" = "$cfg_name" ]; then
    echo "Skipping '$cfg_name': could not extract scene name" >&2
    continue
  fi

  out_file="$ncu_dir/blend_cu--warp-cull-bigger-fetch--${scene}--4090"

  echo "Profiling $cfg_name -> $out_file"
  ncu \
    -o "$out_file" \
    -f \
    -k blend_cu \
    --set detailed \
    --metric smsp__sass_thread_inst_executed_op_fp32_pred_on \
    --import-source yes \
    python ./scripts/inference.py \
      -d "$cfg_dir" \
      -s test
done

