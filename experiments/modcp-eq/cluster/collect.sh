#!/usr/bin/env bash
set -euo pipefail
ATTEMPT=${1:?usage: collect.sh attempt numeric-job-id}
JOB=${2:?numeric job id required}
[[ "$ATTEMPT" =~ ^[a-zA-Z0-9]+$ && "$JOB" =~ ^[0-9]+$ ]] || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/$ATTEMPT
LOCAL="$HERE/../runs/$ATTEMPT"
[[ ! -e "$LOCAL" ]] || { echo 'local archive already exists' >&2; exit 2; }
ARCHIVE="$HERE/../raw-archives/$ATTEMPT.tar"
[[ ! -e "$ARCHIVE" ]] || { echo 'raw tar already exists' >&2; exit 2; }
SSH_OPTIONS=(-o Hostname=login-p02.pax.tufts.edu -o HostKeyAlias=login-prod.pax.tufts.edu -o BatchMode=yes -o ConnectTimeout=15)
QUEUED=$(ssh "${SSH_OPTIONS[@]}" tufts-login "squeue -h -u tawal01 -o '%i'")
if rg -qx "$JOB" <<< "$QUEUED"; then echo 'job is still queued/running' >&2; exit 2; fi
ssh "${SSH_OPTIONS[@]}" tufts-login "test -f '$REMOTE/logs/$JOB.out'"
ssh "${SSH_OPTIONS[@]}" tufts-login "cd '$REMOTE' && sed '/  out\//d' MANIFEST.sha256 | sha256sum -c --quiet && find out logs code -type f -print0 | sort -z | xargs -0 sha256sum > COLLECT.sha256"
ssh "${SSH_OPTIONS[@]}" tufts-login "cd '$REMOTE' && tar -cf COLLECT.tar out logs code COMMIT.txt PROVENANCE.json MANIFEST.sha256 COLLECT.sha256 run.sbatch && if test -f RESUME_ARTIFACTS.json; then tar -rf COLLECT.tar RESUME_ARTIFACTS.json; fi && sha256sum COLLECT.tar > COLLECT.tar.sha256"
mkdir -p "$LOCAL" "$(dirname "$ARCHIVE")"
scp "${SSH_OPTIONS[@]}" "tufts-login:$REMOTE/COLLECT.tar.sha256" "$LOCAL/PULL_ARCHIVE.sha256"
scp "${SSH_OPTIONS[@]}" "tufts-login:$REMOTE/COLLECT.tar" "$ARCHIVE"
EXPECTED=$(cut -d' ' -f1 "$LOCAL/PULL_ARCHIVE.sha256")
ACTUAL=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
[[ "$EXPECTED" == "$ACTUAL" ]] || { echo 'transport tar checksum mismatch' >&2; exit 2; }
tar -xf "$ARCHIVE" -C "$LOCAL"
(cd "$LOCAL" && sha256sum -c COLLECT.sha256 --quiet)
/home/tahmid/Dev/.venv/bin/python "$HERE/archive_raw.py" "$LOCAL" "$ARCHIVE"
ssh "${SSH_OPTIONS[@]}" tufts-login "test -d '$REMOTE' && rm -rf -- '$REMOTE' && test ! -e '$REMOTE'"
printf '%s\n' "$LOCAL"
