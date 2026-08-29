#!/bin/bash
# Stage one 1D-Burgers TENSOR-arm job (2026-08-29): cluster/stage/<jobname>/.
# Ships the code plus the COMMITTED sep_b1d_scale artifacts for that N
# (checkpoint .pkl, node .npz, baseline .json) and E1's CPU-built Q (gate TX).
# Only the 8 test trajectories regenerate in-job (tridiagonal generator).
# Usage: ./stage_tensor.sh <jobname> <N>
set -euo pipefail
JOB=${1:?jobname}; NN=${2:?N}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")

DST=$HERE/stage/$JOB
rm -rf "$DST"
mkdir -p "$DST/code" "$DST/data" "$DST/out" "$DST/logs"
cp "$SEP"/b1d_common.py "$SEP"/b1d_fast_common.py "$SEP"/b1d_tensor_common.py \
   "$SEP"/sep_b1d_tensor.py "$DST/code/"
cp "$SEP"/runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n$NN.pkl \
   "$SEP"/runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n${NN}_nodes.npz \
   "$SEP"/runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n$NN.json \
   "$SEP"/runs/b1dtensor/audit/Q_n$NN.npy "$DST/data/"
git -C "$SEP" rev-parse HEAD > "$DST/COMMIT.txt"
( cd "$DST" && find code data COMMIT.txt -type f -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files, commit $(cat "$DST/COMMIT.txt"))"
