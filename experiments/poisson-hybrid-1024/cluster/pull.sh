#!/usr/bin/env bash
# pull.sh <cell>: checksum, pull, verify, then delete the explicit remote job directory.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: pull.sh <cell>" >&2; exit 2; }
cell="$1"
[[ "$cell" =~ ^[a-z0-9_]+$ ]] || { echo "unsafe cell name: $cell" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/hybp1024/$cell"
DST="$EXP/runs/$cell"
mkdir -p "$DST"
ssh tufts-login "cd '$REMOTE' && { find out logs -type f -exec sha256sum {} \;; sha256sum MANIFEST.sha256 run.sbatch; } | sort" \
  > "$DST/REMOTE.sha256"
scp -q -r "tufts-login:$REMOTE/out" "tufts-login:$REMOTE/logs" "$DST/"
scp -q "tufts-login:$REMOTE/MANIFEST.sha256" "tufts-login:$REMOTE/run.sbatch" "$DST/"
(cd "$DST" && { find out logs -type f -exec sha256sum {} \;; \
  sha256sum MANIFEST.sha256 run.sbatch; } | sort > LOCAL.sha256)
diff -u "$DST/REMOTE.sha256" "$DST/LOCAL.sha256"
echo "checksums OK; deleting explicit completed job directory $REMOTE"
ssh tufts-login "test '$REMOTE' = '/cluster/tufts/paralab/tawal01/hybp1024/$cell' && rm -rf '$REMOTE'"
