#!/usr/bin/env bash
# pull_cell.sh CELL JOB_ID
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

ssh tufts-login "test -f '$REMOTE/logs/$job_id.out'; grep -q '^ALL-DONE$' '$REMOTE/logs/$job_id.out'; grep -q 'jax_backend=gpu' '$REMOTE/logs/$job_id.out'; test ! -s '$REMOTE/logs/$job_id.err'; cd '$REMOTE' && find logs out -type f -exec sha256sum {} \; | sort > REMOTE.sha256"
mkdir -p "$LOCAL"
scp -q -r "tufts-login:$REMOTE/logs" "tufts-login:$REMOTE/out" \
  "tufts-login:$REMOTE/MANIFEST.sha256" "tufts-login:$REMOTE/REMOTE.sha256" "$LOCAL/"
(cd "$LOCAL" && sha256sum -c REMOTE.sha256)
if rg -n -i 'captured.large.constant|out of memory|(^|[^[:alpha:]])oom([^[:alpha:]]|$)|disk.full|no space|traceback|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
  "$LOCAL/logs/$job_id.out" "$LOCAL/logs/$job_id.err"; then
  echo "health-warning pattern found; inspect before accepting" >&2
  exit 4
fi
(cd "$LOCAL" && find . -type f -not -name LOCAL.sha256 -exec sha256sum {} \; | sort > LOCAL.sha256)
ssh tufts-login "test '$REMOTE' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell'; rm -rf '$REMOTE'"
echo "$LOCAL"
