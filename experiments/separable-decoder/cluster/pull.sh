#!/bin/bash
# Pull results/logs of one finished sepdec_n128 job into runs/, verify the
# in-job RESULTS.sha256, and (only on verified success) delete the remote dir.
# Usage: pull.sh <jobname> [--keep-remote]
set -euo pipefail
JOB=${1:?usage: pull.sh <jobname> [--keep-remote]}
KEEP=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
REMOTE=/cluster/tufts/paralab/tawal01/sepdec_n128/$JOB
DEST=$SEP/runs/sepdec_n128_$JOB
mkdir -p "$DEST"
rsync -a tufts-login:"$REMOTE"/out "$DEST"/
rsync -a tufts-login:"$REMOTE"/logs "$DEST"/
rsync -a tufts-login:"$REMOTE"/MANIFEST.sha256 "$DEST"/ 2>/dev/null || true
( cd "$DEST/out" && sha256sum -c RESULTS.sha256 --quiet ) && echo LOCAL-SHA-OK
if [ "$KEEP" != "--keep-remote" ]; then
  ssh tufts-login "rm -rf $REMOTE"
  echo "remote deleted: $REMOTE"
fi
echo "pulled -> $DEST"
