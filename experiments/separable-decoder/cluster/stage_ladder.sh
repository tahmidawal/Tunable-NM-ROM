#!/bin/bash
# Stage JOB A (ladder on one GPU): code + the six committed b1ds artifacts.
# Usage: ./stage_ladder.sh <jobname>
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
DST=$HERE/stage/$JOB
rm -rf "$DST"
mkdir -p "$DST/code" "$DST/data" "$DST/out" "$DST/logs"
cp "$SEP"/b1d_common.py "$SEP"/b1d_fast_common.py "$SEP"/b1d_tensor_common.py \
   "$SEP"/sep_b1d_ladder.py "$DST/code/"
for NN in 128 256 512 1024 2048 4096; do
  cp "$SEP"/runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n$NN.pkl \
     "$SEP"/runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n${NN}_nodes.npz \
     "$SEP"/runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n$NN.json "$DST/data/"
done
git -C "$SEP" rev-parse HEAD > "$DST/COMMIT.txt"
( cd "$DST" && find code data COMMIT.txt -type f -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files, commit $(cat "$DST/COMMIT.txt"))"
