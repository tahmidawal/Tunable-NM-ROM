#!/bin/bash
# pull.sh <job> [--delete-large]: copy logs, OUTPUTS.sha256 and every small JSON of one job dir into runs/<job>/
# (never checkpoints or fields), verify their checksums, require jax_backend=gpu in the log. With --delete-large,
# remove the remote checkpoints/fields afterwards (keeps the small files on the cluster).
set -euo pipefail
JOB="$1"; DEL="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC=/cluster/tufts/paralab/tawal01/opsall_20260924/b3d/$JOB
DEST="$HERE/runs/$JOB"
mkdir -p "$DEST"
rsync -az --prune-empty-dirs --exclude='code/' --exclude='cache/' --exclude='fields/' --include='*/' --include='*.json' --include='TRAIN_DONE' --include='*.out' --include='*.err' \
  --include='OUTPUTS.sha256' --include='MANIFEST.sha256' --include='COMMIT.txt' --exclude='*' \
  "tufts-login:$SRC/" "$DEST/"
rm -rf "$DEST/code"
if [[ -f "$DEST/OUTPUTS.sha256" ]]; then
  ( cd "$DEST" && grep -v '/fields/' OUTPUTS.sha256 | grep -E '\.json$|TRAIN_DONE$' | sha256sum -c --quiet ) && echo "checksums verified"
fi
grep -q "jax_backend=gpu" "$DEST"/logs/*.out || { echo "no jax_backend=gpu" >&2; exit 1; }
grep -h "TRAIN_EXIT\|PANEL_EXIT\|ALL-DONE" "$DEST"/logs/*.out || true
if [[ "$DEL" == "--delete-large" ]]; then
  ssh tufts-login "find $SRC -name '*.pt' -delete; rm -rf $SRC/output/panel/fields $SRC/cache $SRC/tmp"; echo "remote large files deleted"
fi
