#!/bin/bash
# Stage a phase-4 ladder job: code + the SRC_N phase-2 cache (bank + certified head) pulled from the cluster job dir.
# Usage: ./stage_ladder.sh <N> <BC> <SRC_N>
set -euo pipefail
NN=${1:?N}; BCX=${2:?BC}; SRCN=${3:?SRC_N}
JOB=ladder_n${NN}_${BCX}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
DST=$HERE/stage/$JOB
rm -rf "$DST"; mkdir -p "$DST/code" "$DST/out" "$DST/cache" "$DST/logs"
cp "$SEP"/wav2d_common.py "$SEP"/wav2d_bank.py "$SEP"/wav2d_head.py "$SEP"/wav2d_rom.py "$SEP"/wav2d_ladder.py "$SEP"/stk2d_head.py "$DST/code/"
SRCJOB=n${SRCN}_${BCX}
rsync -a --include='bank_*.npz' --include='head_*.npz' --exclude='*' tufts-login:/cluster/tufts/paralab/tawal01/wav2d/$SRCJOB/cache/ "$DST/cache/"
git -C "$SEP" rev-parse HEAD > "$DST/COMMIT.txt"
( cd "$DST" && find code cache COMMIT.txt -type f \( -name '*.py' -o -name '*.npz' -o -name 'COMMIT.txt' \) -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST ($(wc -l < "$DST/MANIFEST.sha256") files, commit $(cat "$DST/COMMIT.txt"))"
