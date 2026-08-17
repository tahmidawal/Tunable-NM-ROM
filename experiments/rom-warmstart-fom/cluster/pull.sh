#!/usr/bin/env bash
# pull.sh <cell> : copy out/ + logs/ back into runs/<cell>/, verify checksums,
# and print the remote dir so it can be deleted (the group share is ~99% full).
set -euo pipefail
cell="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/wsfom/$cell"
DST="$EXP/runs/$cell"
mkdir -p "$DST"
ssh tufts-login "cd $REMOTE && find out logs -type f -exec sha256sum {} \; | sort" > "$DST/REMOTE.sha256"
scp -q -r "tufts-login:$REMOTE/out" "tufts-login:$REMOTE/logs" "$DST/"
(cd "$DST" && find out logs -type f -exec sha256sum {} \; | sort > LOCAL.sha256)
if ! diff -q "$DST/REMOTE.sha256" "$DST/LOCAL.sha256" >/dev/null; then
  echo "CHECKSUM MISMATCH pulling $cell" >&2; diff "$DST/REMOTE.sha256" "$DST/LOCAL.sha256" >&2; exit 3
fi
echo "pulled $cell -> $DST (checksums OK).  Now delete: ssh tufts-login 'rm -rf $REMOTE'"
