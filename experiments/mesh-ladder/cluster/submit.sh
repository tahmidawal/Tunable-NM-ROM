#!/bin/bash
# Stage, copy straight into a fresh paralab attempt directory, and submit one job.
# Never reuses an attempt directory and never stages through the login node's /tmp.
set -euo pipefail
PDE="${1:?usage: submit.sh <burgers|poisson> <attempt> [gpu] [mem] [time]}"
ATTEMPT="${2:?attempt name}"
GPU="${3:-a100}"
MEM="${4:-96G}"
TIME="${5:-04:00:00}"
NS=/cluster/tufts/paralab/tawal01/mrladder_20260914
REMOTE="$NS/$ATTEMPT"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STAGE="$(mktemp -d)/code"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

/home/tahmid/Dev/.venv/bin/python "$HERE/stage.py" --out "$STAGE"

echo "--- squeue BEFORE ---"
ssh tufts-login "squeue -u tawal01 -o '%i %j %T %Z' "
ssh tufts-login "test ! -e $REMOTE || { echo 'attempt directory already exists: $REMOTE' >&2; exit 1; }"
ssh tufts-login "mkdir -p $REMOTE/out && df -h $NS | tail -1"

sed -e "s|__PDE__|$PDE|g" -e "s|__GPU__|$GPU|g" -e "s|__MEM__|$MEM|g" \
    -e "s|__TIME__|$TIME|g" -e "s|__REMOTE__|$REMOTE|g" -e "s|__COMMIT__|$COMMIT|g" \
    "$HERE/run.sbatch.template" > "$STAGE/../run.sbatch"

rsync -a --checksum "$STAGE" "tufts-login:$REMOTE/"
scp -q "$STAGE/../run.sbatch" "tufts-login:$REMOTE/run.sbatch"
ssh tufts-login "cd $REMOTE/code && sha256sum -c MANIFEST.sha256 --quiet && echo 'MANIFEST VERIFIED'"

JOB=$(ssh tufts-login "cd $REMOTE && sbatch --parsable run.sbatch")
echo "JOB=$JOB REMOTE=$REMOTE PDE=$PDE GPU=$GPU COMMIT=$COMMIT"
echo "--- squeue AFTER ---"
ssh tufts-login "squeue -u tawal01 -o '%i %j %T %Z'"
echo "$JOB" > "$HERE/../runs/$ATTEMPT.jobid" 2>/dev/null || true
