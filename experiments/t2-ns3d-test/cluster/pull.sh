#!/bin/bash
# pull.sh <job-name> [--delete]: copy one job directory's output/logs/manifests into runs/<job>/,
# verify every manifest checksum, require jax_backend=gpu, and (with --delete) remove the remote dir.
set -euo pipefail
JOB="$1"; DEL="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC=/cluster/tufts/paralab/tawal01/t2ntest_20260925/$JOB
DEST="$HERE/runs/$JOB"
mkdir -p "$DEST"
rsync -az "tufts-login:$SRC/output" "tufts-login:$SRC/logs" "tufts-login:$SRC/OUTPUTS.sha256" "tufts-login:$SRC/COMMIT.txt" \
  "tufts-login:$SRC/checkpoints/CHECKPOINTS.sha256" "$DEST/"
( cd "$DEST" && sha256sum -c --quiet OUTPUTS.sha256 )
echo "checksums verified: $(wc -l < "$DEST/OUTPUTS.sha256") files"
grep -q "jax_backend=gpu" "$DEST"/logs/*.out || { echo "no jax_backend=gpu" >&2; exit 1; }
grep -h "RUN_EXIT" "$DEST"/logs/*.out || true
if [[ "$DEL" == "--delete" ]]; then ssh tufts-login "rm -rf $SRC"; echo "remote $SRC deleted"; fi
