#!/bin/bash
# Stage JOB B/C (train in-job at a large N, then the tensor driver): code only;
# everything regenerates from seeds in-job.  Usage: ./stage_big.sh <jobname>
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
DST=$HERE/stage/$JOB
rm -rf "$DST"
mkdir -p "$DST/code" "$DST/out" "$DST/logs"
cp "$SEP"/b1d_common.py "$SEP"/b1d_fast_common.py "$SEP"/b1d_tensor_common.py \
   "$SEP"/sep_b1d_scale.py "$SEP"/sep_b1d_tensor.py "$DST/code/"
git -C "$SEP" rev-parse HEAD > "$DST/COMMIT.txt"
( cd "$DST" && find code COMMIT.txt -type f -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files, commit $(cat "$DST/COMMIT.txt"))"
