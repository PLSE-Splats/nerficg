#!/usr/bin/env bash
set -euo pipefail

repo_root="$(pwd)"
config="$repo_root/src/Methods/FasterGS/FasterGSCudaBackend/FasterGSCudaBackend/rasterization/include/rasterization_config.h"
backend_dir="$repo_root/src/Methods/FasterGS/FasterGSCudaBackend"

cd "$repo_root"

for n in 1 2 3 4 5 6 7 8; do
  echo "=== warp_fetch_size x$n ==="

  python - "$config" "$n" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
n = sys.argv[2]
text = path.read_text(encoding='utf-8')
new_text, count = re.subn(
    r'DEF int warp_fetch_size = warp_size \* \d+;',
    f'DEF int warp_fetch_size = warp_size * {n};',
    text,
)
if count != 1:
    raise SystemExit(f'expected exactly one warp_fetch_size replacement, got {count}')
path.write_text(new_text, encoding='utf-8')
PY

  (cd "$backend_dir" && python -m pip install . --no-build-isolation > "$repo_root/build_${n}.log" 2>&1)
  python scripts/benchmark_all.py > "$repo_root/inference_${n}.log" 2>&1

  test -f all_fps.csv
  lines="$(wc -l < all_fps.csv)"
  echo "all_fps.csv line count: $lines"
  mv -f all_fps.csv "all_fps_${n}.csv"
  echo "saved all_fps_${n}.csv"
done
