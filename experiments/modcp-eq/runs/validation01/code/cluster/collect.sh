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
QUEUED=$(ssh "${SSH_OPTIONS[@]}" tufts-login "squeue -h -u tawal01 -o '%i'")
if rg -qx "$JOB" <<< "$QUEUED"; then echo 'job is still queued/running' >&2; exit 2; fi
ssh "${SSH_OPTIONS[@]}" tufts-login "cd '$REMOTE' && sed '/  out\//d' MANIFEST.sha256 | sha256sum -c --quiet && find out logs code -type f -print0 | sort -z | xargs -0 sha256sum > COLLECT.sha256"
mkdir -p "$LOCAL"
for NAME in out logs code COMMIT.txt PROVENANCE.json MANIFEST.sha256 COLLECT.sha256 run.sbatch; do
  scp "${SSH_OPTIONS[@]}" -r "tufts-login:$REMOTE/$NAME" "$LOCAL/"
done
if ssh "${SSH_OPTIONS[@]}" tufts-login "test -f '$REMOTE/RESUME_ARTIFACTS.json'"; then
  scp "${SSH_OPTIONS[@]}" "tufts-login:$REMOTE/RESUME_ARTIFACTS.json" "$LOCAL/"
fi
(cd "$LOCAL" && sha256sum -c COLLECT.sha256 --quiet)
ssh "${SSH_OPTIONS[@]}" tufts-login "test -d '$REMOTE' && rm -rf -- '$REMOTE' && test ! -e '$REMOTE'"
printf '%s\n' "$LOCAL"
