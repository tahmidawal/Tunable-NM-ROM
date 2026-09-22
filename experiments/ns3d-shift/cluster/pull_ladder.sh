#!/bin/bash
# Pull one ladder job, verify the checksums the job wrote, re-verify the numbers
# locally when the fields came with it, then delete the remote job directory.
#
# At the finer meshes the job deletes its own multi-GB .npy files after running the
# independent recomputation in-job, so they never cross the wire. The manifest still
# lists them -- it is the record of what the job produced -- so the local check runs
# with --ignore-missing and prints exactly which files it could not check. The
# in-job verify.json is then the independent recomputation of record.
#
# Usage: pull_ladder.sh <jobname> <jobid> <scratch-dir>
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

MANIFEST="$DEST/OUTPUTS.sha256"
( cd "$SCRATCH/$JOB" && sed 's#  output/#  #' "$MANIFEST" | sha256sum -c --ignore-missing - )
# every small artefact must be present and checked; only .npy may be absent
awk '{print $2}' "$MANIFEST" | sed 's#^output/##' | while read -r f; do
  case "$f" in
    *.npy) [ -f "$SCRATCH/$JOB/$f" ] || echo "  not transferred (deleted in-job): $f" ;;
    *) [ -f "$SCRATCH/$JOB/$f" ] || { echo "MISSING non-field artefact: $f" >&2; exit 1; } ;;
  esac
done
echo "checksums verified"

cp "$SCRATCH/$JOB/summary.json" "$DEST/summary.json"
cp "$SCRATCH/$JOB/verify.json" "$DEST/verify.json"

PY=/home/tahmid/Dev/.venv/bin/python
if compgen -G "$SCRATCH/$JOB/fields_*.npy" > /dev/null; then
  "$PY" "$HERE/experiments/ns3d-shift/verify_ladder.py" --out "$SCRATCH/$JOB"
  cp "$SCRATCH/$JOB/verify.json" "$DEST/verify.json"
  echo "re-verified locally from the pulled fields"
else
  echo "fields were deleted in-job; the in-job verify.json is the record"
fi

ssh tufts-login "rm -rf $SRC"
echo "remote $SRC deleted"
