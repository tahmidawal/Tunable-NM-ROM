#!/usr/bin/env bash
# pull.sh [cell ...]
#   For every FINISHED cell (no job of that name queued), copy out/ + logs/ into
#   runs/<cell>/, verify every out/ file by sha256 against the cluster copy, and
#   -- only when the log contains ALL-DONE and jax_backend=gpu -- DELETE the
#   cluster cell directory (the group share runs ~90% full).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_DIR="$(dirname "$HERE")"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/ctol
DEST="$CELL_DIR/runs"
cells=("$@")
if [[ ${#cells[@]} -eq 0 ]]; then
  mapfile -t cells < <(ssh tufts-login "ls $REMOTE_ROOT 2>/dev/null")
fi
queued="$(ssh tufts-login "squeue -u tawal01 -h -o '%j'")"
for c in "${cells[@]}"; do
  [[ "$c" =~ ^[A-Za-z0-9_]+$ ]] || { echo "skip bad cell name '$c'"; continue; }
  if grep -qx "$c" <<<"$queued"; then echo "$c: still queued/running, skipping"; continue; fi
  done_ok=$(ssh tufts-login "grep -l ALL-DONE $REMOTE_ROOT/$c/logs/*.out 2>/dev/null | wc -l")
  gpu_ok=$(ssh tufts-login "grep -l 'jax_backend=gpu' $REMOTE_ROOT/$c/logs/*.out 2>/dev/null | wc -l")
  mkdir -p "$DEST/$c"
  scp -q -r "tufts-login:$REMOTE_ROOT/$c/out/." "$DEST/$c/" 2>/dev/null
  scp -q "tufts-login:$REMOTE_ROOT/$c/logs/"* "$DEST/$c/" 2>/dev/null
  rem=$(ssh tufts-login "cd $REMOTE_ROOT/$c && find out -type f -exec sha256sum {} + 2>/dev/null | awk '{n=split(\$2,p,\"/\"); print p[n], \$1}' | sort")
  names=$(awk '{print $1}' <<<"$rem")
  loc=$(cd "$DEST/$c" && for f in $names; do [[ -f "$f" ]] && echo "$f $(sha256sum "$f" | cut -d' ' -f1)"; done | sort)
  if [[ -z "$rem" || "$rem" != "$loc" ]]; then
    echo "$c: PAYLOAD CHECKSUM MISMATCH -- keeping the cluster copy" >&2
    diff <(echo "$rem") <(echo "$loc") | head -6
    continue
  fi
  echo "$rem" > "$DEST/$c/.remote.sha256"
  nf=$(echo "$rem" | wc -l)
  if [[ "$done_ok" == "0" || "$gpu_ok" == "0" ]]; then
    echo "$c: pulled $nf out-files but ALL-DONE=$done_ok jax_backend=gpu=$gpu_ok -- cluster dir KEPT"
    continue
  fi
  ssh tufts-login "rm -rf $REMOTE_ROOT/$c"
  echo "$c: pulled $nf out-files (sha256-verified, ALL-DONE, gpu) and the cluster dir was deleted"
done
