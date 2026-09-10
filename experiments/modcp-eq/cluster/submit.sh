#!/usr/bin/env bash
set -euo pipefail
ATTEMPT=${1:?usage: submit.sh attempt smoke-or-train-or-all}
PHASE=${2:?phase required}
[[ "$ATTEMPT" =~ ^[a-zA-Z0-9]+$ ]] || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE=$(/home/tahmid/Dev/.venv/bin/python "$HERE/stage.py" "$ATTEMPT" "$PHASE")
REMOTE=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/$ATTEMPT
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1)
ssh "${SSH_OPTIONS[@]}" tufts-login "squeue -u tawal01 -o '%.12i %.32j %.8T %.10M %.18R'; df -h /cluster/tufts/paralab/tawal01; test ! -e '$REMOTE'; mkdir -p '$REMOTE'"
scp "${SSH_OPTIONS[@]}" -r "$STAGE"/. "tufts-login:$REMOTE/"
ssh "${SSH_OPTIONS[@]}" tufts-login "cd '$REMOTE' && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
ssh "${SSH_OPTIONS[@]}" tufts-login "squeue -u tawal01 -o '%.12i %.32j %.8T %.10M %.18R'"
