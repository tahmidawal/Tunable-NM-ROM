#!/usr/bin/env bash
# pull.sh <cell> : copy out/ + logs/ back into runs/<cell>/ with checksum verify.
set -euo pipefail
cell="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/wlat/$cell"
DEST="$EXP/runs/$cell"
mkdir -p "$DEST"
ssh tufts-login "cd $REMOTE && find out logs -type f -exec sha256sum {} \; | sort" > "$DEST/.remote.sha256"
scp -q -r "tufts-login:$REMOTE/out" "tufts-login:$REMOTE/logs" "$DEST/"
(cd "$DEST" && find out logs -type f -exec sha256sum {} \; | sort > .local.sha256)
if ! diff -q "$DEST/.remote.sha256" "$DEST/.local.sha256" >/dev/null; then
  echo "CHECKSUM MISMATCH pulling $cell" >&2; exit 3; fi
echo "pulled $cell -> $DEST ($(cd "$DEST" && find out -type f | wc -l) files)"
