#!/usr/bin/env bash
# Pull one completed cell with checksums, audit mandatory log guards, then remove
# only that exact cell directory from the cluster share.
set -euo pipefail

cell="$1"
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell $cell" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
REMOTE="$REMOTE_ROOT/$cell"
DEST="$EXP/runs/$cell"
[[ ! -e "$DEST" ]] || { echo "local result already exists: $DEST" >&2; exit 3; }

ssh tufts-login "test -d '$REMOTE/out'"
log_text="$(ssh tufts-login "cat '$REMOTE'/logs/*.out '$REMOTE'/logs/*.err 2>/dev/null")"
grep -q 'jax_backend=gpu' <<<"$log_text"
grep -q 'ALL-DONE' <<<"$log_text"
if grep -Eqi 'out of memory|oom|captured.*constant|no space left|disk quota|traceback|FAILED rc=' <<<"$log_text"; then
  echo "refusing result: failure marker in cluster logs" >&2
  exit 4
fi

mkdir -p "$DEST"
ssh tufts-login "cd '$REMOTE' && find out logs -type f -exec sha256sum {} \\; | sort; sha256sum run.sbatch MANIFEST.sha256" > "$DEST/REMOTE.sha256"
scp -q -r "tufts-login:$REMOTE/out" "tufts-login:$REMOTE/logs" \
  "tufts-login:$REMOTE/run.sbatch" "tufts-login:$REMOTE/MANIFEST.sha256" "$DEST/"
(cd "$DEST" && sha256sum -c REMOTE.sha256)
ssh tufts-login "rm -rf '$REMOTE'"
echo "pulled, verified, and removed remote cell $cell -> $DEST"
