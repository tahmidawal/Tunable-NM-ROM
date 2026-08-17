#!/usr/bin/env bash
# pull.sh [cell ...] : for every FINISHED cell (no job of that name queued and the
# log contains ALL-DONE), copy out/ + logs/ FLAT into runs/followup/<cell>/, verify
# every file by sha256 (basename -> hash) against the cluster copy, and DELETE the
# cluster cell directory (the group share runs ~90% full).  Running cells are
# skipped; a cell without ALL-DONE is pulled for diagnosis but NOT deleted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/../.." && pwd)"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/pobj2
DEST="$EXP/runs/followup"
cells=("$@")
if [[ ${#cells[@]} -eq 0 ]]; then
  mapfile -t cells < <(ssh tufts-login "ls $REMOTE_ROOT 2>/dev/null")
fi
queued="$(ssh tufts-login "squeue -u tawal01 -h -o '%j'")"
for c in "${cells[@]}"; do
  [[ "$c" =~ ^[A-Za-z0-9_]+$ ]] || { echo "skip bad cell name '$c'"; continue; }
  if grep -qx "$c" <<<"$queued"; then echo "$c: still queued/running, skipping"; continue; fi
  done_ok=$(ssh tufts-login "grep -l ALL-DONE $REMOTE_ROOT/$c/logs/*.out 2>/dev/null | wc -l")
  mkdir -p "$DEST/$c"
  scp -q -r "tufts-login:$REMOTE_ROOT/$c/out/." "$DEST/$c/" 2>/dev/null
  scp -q "tufts-login:$REMOTE_ROOT/$c/logs/"* "$DEST/$c/" 2>/dev/null
  rem=$(ssh tufts-login "cd $REMOTE_ROOT/$c && find out logs -type f -exec sha256sum {} + 2>/dev/null | awk '{n=split(\$2,p,\"/\"); print p[n], \$1}' | sort")
  loc=$(cd "$DEST/$c" && find . -maxdepth 1 -type f -exec sha256sum {} + 2>/dev/null | awk '{n=split($2,p,"/"); print p[n], $1}' | sort)
  if [[ "$rem" != "$loc" ]]; then
    echo "$c: CHECKSUM MISMATCH after pull -- keeping the cluster copy" >&2
    diff <(echo "$rem") <(echo "$loc") | head -5
    continue
  fi
  nf=$(echo "$rem" | wc -l)
  if [[ "$done_ok" == "0" ]]; then
    echo "$c: pulled $nf files but NO ALL-DONE in the log -- cluster dir KEPT for diagnosis"
    continue
  fi
  ssh tufts-login "rm -rf $REMOTE_ROOT/$c"
  echo "$c: pulled $nf files (verified) and cluster dir deleted"
done
