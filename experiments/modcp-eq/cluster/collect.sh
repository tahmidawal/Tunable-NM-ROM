#!/usr/bin/env bash
set -euo pipefail
ATTEMPT=${1:?usage: collect.sh attempt numeric-job-id}
JOB=${2:?numeric job id required}
[[ "$ATTEMPT" =~ ^[a-zA-Z0-9]+$ && "$JOB" =~ ^[0-9]+$ ]] || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/$ATTEMPT
LOCAL="$HERE/../runs/$ATTEMPT"
[[ ! -e "$LOCAL" ]] || { echo 'local archive already exists' >&2; exit 2; }
SSH_OPTIONS=(-o Hostname=login-p02.pax.tufts.edu -o HostKeyAlias=login-prod.pax.tufts.edu -o BatchMode=yes -o ConnectTimeout=15)
QUEUED=$(ssh "${SSH_OPTIONS[@]}" tufts-login "squeue -h -j '$JOB' -o '%i'")
[[ -z "$QUEUED" ]] || { echo 'job is still queued/running' >&2; exit 2; }
ssh "${SSH_OPTIONS[@]}" tufts-login "cd '$REMOTE' && sha256sum -c MANIFEST.sha256 --quiet && find out logs code -type f -print0 | sort -z | xargs -0 sha256sum > COLLECT.sha256"
mkdir -p "$LOCAL"
for NAME in out logs code COMMIT.txt PROVENANCE.json MANIFEST.sha256 COLLECT.sha256 run.sbatch; do
  scp "${SSH_OPTIONS[@]}" -r "tufts-login:$REMOTE/$NAME" "$LOCAL/"
done
(cd "$LOCAL" && sha256sum -c COLLECT.sha256 --quiet)
ssh "${SSH_OPTIONS[@]}" tufts-login "test -d '$REMOTE' && rm -rf -- '$REMOTE' && test ! -e '$REMOTE'"
printf '%s\n' "$LOCAL"
