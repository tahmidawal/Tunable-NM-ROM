#!/usr/bin/env bash
# G5-A5 durable retention: copy every snapshot chunk / state of an attempt to the local
# retention/ tree (outside Git) and verify against the collected REMOTE-SHA256SUMS.
# Remote copies are NOT deleted. Usage: cluster/retain.sh <attempt-name>
set -euo pipefail
cd "$(dirname "$0")/.."
A=$1; DIR=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/$A
SUMS=artifacts/$A/REMOTE-SHA256SUMS
[ -s "$SUMS" ] || { echo "collect first"; exit 2; }
mkdir -p retention/$A
grep -E 'snapshots_.*\.npz$|/[a-z_]*\.npz$' "$SUMS" | awk '{print $2}' > retention/$A/FILES
rsync -a --files-from=retention/$A/FILES tufts-login:$DIR/ retention/$A/
( cd retention/$A && grep -F -f FILES ../../$SUMS | sha256sum -c --quiet - ) && echo "retained+verified $(wc -l < retention/$A/FILES) files"
du -sh retention/$A
