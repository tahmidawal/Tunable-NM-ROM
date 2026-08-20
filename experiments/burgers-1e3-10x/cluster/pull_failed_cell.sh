#!/usr/bin/env bash
# Preserve a failed cell with scheduler evidence and checksums, then clean it.
# Usage: pull_failed_cell.sh CELL JOB_ID
set -euo pipefail
cell="$1"
job_id="$2"
[[ "$cell" =~ ^[a-z0-9_]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid job id" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
LOCAL="$EXP/runs/$cell"
[[ ! -e "$LOCAL" ]] || { echo "local run already exists: $LOCAL" >&2; exit 3; }

ssh tufts-login "test -f '$REMOTE/logs/$job_id.out'; test -f '$REMOTE/logs/$job_id.err'; ! grep -q '^ALL-DONE$' '$REMOTE/logs/$job_id.out'; sacct -j '$job_id' --format=JobID,JobName%28,State,Elapsed,ExitCode,NodeList%20,AllocTRES%60 > '$REMOTE/SACCT.txt'; cd '$REMOTE' && find logs out -type f -exec sha256sum {} \; | sort > REMOTE.sha256 && sha256sum SACCT.txt >> REMOTE.sha256"
mkdir -p "$LOCAL"
scp -q -r "tufts-login:$REMOTE/logs" "tufts-login:$REMOTE/out" \
  "tufts-login:$REMOTE/MANIFEST.sha256" "tufts-login:$REMOTE/REMOTE.sha256" \
  "tufts-login:$REMOTE/SACCT.txt" "$LOCAL/"
(cd "$LOCAL" && sha256sum -c REMOTE.sha256)
(cd "$LOCAL" && find . -type f -not -name LOCAL.sha256 -exec sha256sum {} \; | sort > LOCAL.sha256)
ssh tufts-login "test '$REMOTE' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell'; rm -rf '$REMOTE'"
echo "$LOCAL"
