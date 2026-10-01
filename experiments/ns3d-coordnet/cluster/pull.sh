#!/bin/bash
# Pull one job, verify every manifest the job wrote (only audited, deleted field files may
# be absent), check the GPU preflight and the exit records, then delete the remote dir.
# A failed job is pulled but its remote dir is kept unless FORCE_DELETE=1.
# Usage: pull.sh <job> <jobid>
set -euo pipefail
JOB="$1"; ID="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC=/cluster/tufts/paralab/tawal01/nscoord_20261001/$JOB
DEST="$HERE/experiments/ns3d-coordnet/runs/$JOB"
mkdir -p "$DEST/output"
rsync -az --exclude='fields_*.npy' --exclude='dev_truth.npy' "tufts-login:$SRC/output/" "$DEST/output/"
rsync -az "tufts-login:$SRC/OUTPUTS.sha256" "tufts-login:$SRC/COMMIT.txt" "$DEST/"
rsync -az "tufts-login:$SRC/logs/" "$DEST/logs/"
check() {  # <manifest> <base>
  ( cd "$2" && sha256sum -c --ignore-missing --quiet "$1" )
  awk '{print $2}' "$1" | while read -r f; do
    case "$f" in
      *fields_*.npy|*dev_truth.npy|*.OUTPUTS.sha256) ;;
      *) [ -f "$2/$f" ] || { echo "MISSING artefact: $f" >&2; exit 1; } ;;
    esac
  done
}
check "$DEST/OUTPUTS.sha256" "$DEST"
for m in "$DEST"/output/*.OUTPUTS.sha256; do [ -e "$m" ] && check "$m" "$DEST"; done
echo "checksums verified"
LOG="$DEST/logs/$ID.out"
grep -q "jax_backend=gpu" "$LOG" || { echo "no jax_backend=gpu in log" >&2; exit 1; }
OK=1
grep -E "RUN_EXIT=|ALL_EXIT=" "$LOG" || OK=0
if grep -qE "RUN_EXIT=[^0]|VERIFY_EXIT=[^0]|ALL_EXIT=[^0]" "$LOG"; then OK=0; fi
if [[ "$OK" -eq 1 || "${FORCE_DELETE:-0}" -eq 1 ]]; then
  ssh tufts-login "rm -rf $SRC"; echo "remote $SRC deleted"
else
  echo "job did not pass cleanly; remote $SRC KEPT (FORCE_DELETE=1 to delete after inspection)"
fi
