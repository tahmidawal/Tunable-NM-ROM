#!/bin/bash
# Pull one job's outputs, verify the checksums the job wrote, re-verify the
# numbers locally from the fields, then delete the remote job directory.
# Large .npy fields land in a scratch directory and are never committed.
# Usage: pull.sh <jobname> <jobid> <scratch-dir>
set -euo pipefail
JOB="$1"; ID="$2"; SCRATCH="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NS=/cluster/tufts/paralab/tawal01/ns3dshift_20260922
SRC="$NS/$JOB"
DEST="$HERE/experiments/ns3d-shift/runs/$JOB"
mkdir -p "$DEST" "$SCRATCH/$JOB"

rsync -az "tufts-login:$SRC/output/" "$SCRATCH/$JOB/"
rsync -az "tufts-login:$SRC/OUTPUTS.sha256" "tufts-login:$SRC/COMMIT.txt" "$DEST/"
rsync -az "tufts-login:$SRC/logs/$ID.out" "$DEST/"
rsync -az "tufts-login:$SRC/logs/$ID.err" "$DEST/" 2>/dev/null || true

( cd "$SCRATCH/$JOB" && sed 's#  output/#  #' "$DEST/OUTPUTS.sha256" | sha256sum -c - )
echo "checksums verified"

cp "$SCRATCH/$JOB/summary.json" "$DEST/summary.json"
[ -f "$SCRATCH/$JOB/verify.json" ] && cp "$SCRATCH/$JOB/verify.json" "$DEST/verify.json"

PY=/home/tahmid/Dev/.venv/bin/python
"$PY" "$HERE/experiments/ns3d-shift/verify_ladder.py" --out "$SCRATCH/$JOB"
cp "$SCRATCH/$JOB/verify.json" "$DEST/verify.json"

ssh tufts-login "rm -rf $SRC"
echo "remote $SRC deleted"
