#!/bin/bash
# Pull one job, verify the manifest the job wrote (fields were deleted in-job after the
# independent audit, so only they may be absent), then delete the remote directory.
# Usage: pull.sh <attempt_job> <jobid>
set -euo pipefail
JOB="$1"; ID="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC=/cluster/tufts/paralab/tawal01/nshead_20260923/$JOB
DEST="$HERE/experiments/ns3d-shift-head/runs/$JOB"
mkdir -p "$DEST/output"
rsync -az "tufts-login:$SRC/output/" "$DEST/output/"
rsync -az "tufts-login:$SRC/OUTPUTS.sha256" "tufts-login:$SRC/COMMIT.txt" "$DEST/"
rsync -az "tufts-login:$SRC/logs/" "$DEST/logs/"
MANIFEST="$DEST/OUTPUTS.sha256"
( cd "$DEST" && sha256sum -c --ignore-missing "$MANIFEST" | grep -v ': OK$' || true )
( cd "$DEST" && sha256sum -c --ignore-missing --quiet "$MANIFEST" )
awk '{print $2}' "$MANIFEST" | while read -r f; do
  case "$f" in
    output/fields_*.npy|output/dev_truth.npy) ;;
    *) [ -f "$DEST/$f" ] || { echo "MISSING artefact: $f" >&2; exit 1; } ;;
  esac
done
echo "checksums verified for $(wc -l < "$MANIFEST") manifest lines"
grep -q "jax_backend=gpu" "$DEST/logs/$ID.out" || { echo "no jax_backend=gpu in log" >&2; exit 1; }
[ -f "$DEST/output/verify.json" ] || { echo "no verify.json" >&2; exit 1; }
ssh tufts-login "rm -rf $SRC"
echo "remote $SRC deleted"
