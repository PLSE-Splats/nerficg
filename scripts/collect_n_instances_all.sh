#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="$REPO_ROOT/output/FasterGS"

shopt -s nullglob

for folder in "$BASE_DIR"/*/; do
  [[ -d "$folder" ]] || continue

  folder_name="$(basename "${folder%/}")"
  echo "Running inference for $folder_name"
  python "$REPO_ROOT/scripts/inference.py" -d "$folder" -s test

  source_file="$REPO_ROOT/n_instances.csv"
  renamed_file="$REPO_ROOT/${folder_name}--n_instances.csv"

  if [[ ! -f "$source_file" ]]; then
    echo "Expected $source_file after inference, but it was not created." >&2
    exit 1
  fi

  mv -f "$source_file" "$renamed_file"
  echo "Renamed n_instances.csv to ${folder_name}--n_instances.csv"
done
