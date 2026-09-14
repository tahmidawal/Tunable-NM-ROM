#!/usr/bin/env bash
set -euo pipefail
ATTEMPT=${1:?usage: submit.sh attempt}
[[ "$ATTEMPT" =~ ^[a-zA-Z0-9]+$ ]] || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/tahmid/Dev/.venv/bin/python
STAGE=$("$PY" "$HERE/stage.py" "$ATTEMPT")
REMOTE=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/$ATTEMPT
ssh tufts-login "squeue -u tawal01 -o '%.12i %.32j %.8T %.10M %.18R'; df -h /cluster/tufts/paralab/tawal01; test ! -e '$REMOTE'; mkdir -p '$REMOTE'"
scp -r "$STAGE"/. "tufts-login:$REMOTE/"
ssh tufts-login "cd '$REMOTE' && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
ssh tufts-login "squeue -u tawal01 -o '%.12i %.32j %.8T %.10M %.18R'"
