#!/bin/bash
# Stage one Wave-2D cell (phase 2 + phase 3): cluster/stage/<jobname>/code with the wav2d_*.py modules and
# their one dependency (stk2d_head.py).  Data regenerate from the seed in-job (nothing synced).
# Usage: ./stage_wav2d.sh <jobname>
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
DST=$HERE/stage/$JOB
rm -rf "$DST"; mkdir -p "$DST/code" "$DST/out" "$DST/cache" "$DST/logs"
cp "$SEP"/wav2d_common.py "$SEP"/wav2d_bank.py "$SEP"/wav2d_head.py "$SEP"/wav2d_head_gates.py \
   "$SEP"/wav2d_rom.py "$SEP"/wav2d_rom_gates.py "$SEP"/stk2d_head.py "$DST/code/"
git -C "$SEP" rev-parse HEAD > "$DST/COMMIT.txt"
( cd "$DST" && find code COMMIT.txt -type f \( -name '*.py' -o -name 'COMMIT.txt' \) -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST ($(wc -l < "$DST/MANIFEST.sha256") files, commit $(cat "$DST/COMMIT.txt"))"
