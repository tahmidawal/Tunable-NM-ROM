#!/bin/bash
# Stage one 1D-Burgers screening job: cluster/stage/<jobname>/.  The 1D
# testbed is deliberately self-contained (b1d_common.py + sep_b1d_screen.py,
# no deps/ tree, no checkpoints -- everything regenerates from seeds in-job).
# Usage: ./stage_b1d.sh <jobname>
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")

DST=$HERE/stage/$JOB
rm -rf "$DST"
mkdir -p "$DST/code" "$DST/out" "$DST/logs"
cp "$SEP"/b1d_common.py "$SEP"/sep_b1d_screen.py "$DST/code/"
( cd "$DST" && find code -type f -name '*.py' -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
echo "staged -> $DST  ($(wc -l < "$DST/MANIFEST.sha256") files)"
