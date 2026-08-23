#!/bin/bash
# Pull a finished sepdec_n1024 job: rsync out/+logs/ into runs/, verify the
# job's own RESULTS.sha256, and (only after PASS) delete the remote job dir.
# Usage: ./pull_job.sh <jN> [--delete]
set -euo pipefail
J=${1:?usage: pull_job.sh <jN> [--delete]}
DEL=${2:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../runs/sepdec_n1024_$J"
REMOTE=/cluster/tufts/paralab/tawal01/sepdec_n1024/$J
mkdir -p "$DEST"
rsync -az "tufts-login:$REMOTE/out/" "$DEST/out/"
rsync -az "tufts-login:$REMOTE/logs/" "$DEST/logs/"
( cd "$DEST/out" && sha256sum -c RESULTS.sha256 )
echo "PULL-VERIFIED $J"
if [ "$DEL" = "--delete" ]; then
  ssh -o BatchMode=yes tufts-login "rm -rf $REMOTE"
  echo "REMOTE-DELETED $REMOTE"
fi
