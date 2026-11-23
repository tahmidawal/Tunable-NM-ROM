#!/bin/bash
# One new directory and one active job per lane. Input code and checkpoints only.
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
LANE=experiments/jcp-phase-confirm
NS=/cluster/tufts/paralab/tawal01/jcp-phase-confirm-20261119
JOB="${1:?unique pilot id required}"
CONFIG="${2:-configs/confirm.json}"
[[ "$CONFIG" =~ ^configs/[a-z0-9_]+\.json$ ]] || exit 2
[[ "$JOB" =~ ^[a-z0-9]+$ ]] || exit 2
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[[ -z "$(git status --porcelain -- "$LANE")" ]] || { echo 'Commit and mirror lane before staging'; exit 2; }
DEST="$NS/$JOB"
ssh tufts-login "test \"\$(cat $NS/OWNER.txt)\" = 'jcp_phase_confirmation 2026-11-19 exp/2026-11-19-jcp-phase-confirm' && test ! -e $DEST && mkdir -p $DEST/experiments $DEST/logs"
PACKAGE=$(mktemp)
trap 'rm -f "$PACKAGE"' EXIT
tar --exclude='runs' --exclude='__pycache__' -czf "$PACKAGE" "$LANE"
scp "$PACKAGE" "tufts-login:$DEST/code.tar.gz"
COMMIT=$(git rev-parse HEAD)
ssh tufts-login "cd $DEST && tar -xzf code.tar.gz && rm code.tar.gz && printf '%s\\n' '$COMMIT' > COMMIT.txt && find experiments -type f -print0 | sort -z | xargs -0 sha256sum > INPUTS.sha256"
ssh tufts-login 'squeue -u tawal01 -o "%.12i %.30j %.10T %.10M %R"'
if ssh tufts-login 'squeue -u tawal01 -h -o "%j"' | rg -q '^ctol_jpc_'; then echo 'Lane already has active job'; exit 2; fi
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=ctol_jpc_$JOB --gres=gpu:a100:1 --constraint=a100-80G --mem=64G --time=00:20:00 --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err experiments/jcp-phase-confirm/cluster/job.sbatch $LANE/$CONFIG")
printf '%s\n' "$ID" > "$LANE/runs/$JOB.jobid"
ssh tufts-login 'squeue -u tawal01 -o "%.12i %.30j %.10T %.10M %R"'
printf 'JOB_ID=%s DEST=%s\n' "$ID" "$DEST"
