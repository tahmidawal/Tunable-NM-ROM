#!/bin/bash
# Stage one A100 confirmation job for the optimized rollout (sep_b1d_fast.py).
# code/ = the three self-contained modules; in/ = the committed scale-run
# artifacts for this N (checkpoint, node sets, baseline JSON) so nothing
# retrains and parity is checked against the exact committed baseline.
# Usage: ./stage_fast.sh <jobname> <N>
set -euo pipefail
JOB=${1:?jobname}; NN=${2:?N}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
SRC=$SEP/runs/b1dqf/b1ds_n$NN/out
DST=$HERE/stage/$JOB
rm -rf "$DST"
mkdir -p "$DST/code" "$DST/in" "$DST/out" "$DST/logs"
cp "$SEP"/b1d_common.py "$SEP"/b1d_fast_common.py "$SEP"/sep_b1d_fast.py "$DST/code/"
cp "$SRC/sep_b1d_scale_n$NN.pkl" "$SRC/sep_b1d_scale_n${NN}_nodes.npz" \
   "$SRC/sep_b1d_scale_n$NN.json" "$DST/in/"
( cd "$DST" && find code in -type f -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files)"
