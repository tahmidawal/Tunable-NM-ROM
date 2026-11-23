#!/usr/bin/env bash
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
LANE=experiments/jcp-ns-initializer
RUN_ID=${1:?usage: submit_fresh.sh fNN}
[[ "$RUN_ID" =~ ^f[0-9]{2}$ ]] || exit 2
NS=/cluster/tufts/paralab/tawal01/jcp-ns-initializer-20261123
REMOTE="$NS/$RUN_ID"
LOCAL="$ROOT/$LANE/runs/$RUN_ID"
[[ -f "$LANE/FROZEN_PROTOCOL.json" && -f "$LANE/FRESH_COHORT_REGISTRATION.json" ]] || exit 2
[[ ! -e "$LOCAL" ]] || { echo 'run directory already exists' >&2; exit 2; }
git diff --quiet HEAD -- "$LANE" || { echo 'commit all experiment files first' >&2; exit 2; }
[[ -z "$(git ls-files --others --exclude-standard "$LANE")" ]] || { echo 'untracked experiment files' >&2; exit 2; }
mkdir -p "$LOCAL"
git rev-parse HEAD > "$LOCAL/COMMIT.txt"
git archive --format=tar.gz --output="$LOCAL/code.tar.gz" HEAD \
  "$LANE/PROTOCOL.md" "$LANE"/*.json "$LANE"/*.py \
  "$LANE/vendor" "$LANE/assets" "$LANE/configs" "$LANE/cluster"
cp "$LANE/cluster/fresh.sbatch" "$LOCAL/job.sbatch"
sed -i "s/ctol_jni_f01/ctol_jni_$RUN_ID/" "$LOCAL/job.sbatch"
cd "$LOCAL"
sha256sum COMMIT.txt code.tar.gz job.sbatch > INPUTS.sha256
ssh tufts-login 'squeue -u tawal01; df -h /cluster/tufts/paralab'
ACTIVE=$(ssh tufts-login "squeue -u tawal01 -h -o '%j'" | /usr/bin/awk '/^ctol_jni_/{n++} END{print n+0}')
[[ "$ACTIVE" -eq 0 ]] || { echo 'initializer job already active' >&2; exit 3; }
ssh tufts-login "mkdir -p '$NS'; mkdir '$REMOTE'"
scp COMMIT.txt code.tar.gz job.sbatch INPUTS.sha256 "tufts-login:$REMOTE/"
ssh tufts-login "cd '$REMOTE' && sha256sum -c INPUTS.sha256 && mkdir code && tar -xzf code.tar.gz -C code"
JOB_ID=$(ssh tufts-login "cd '$REMOTE' && sbatch --parsable job.sbatch")
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "unexpected scheduler reply $JOB_ID" >&2; exit 4; }
printf '%s\n' "$JOB_ID" > JOB_ID.txt
printf '%s\n' "$REMOTE" > REMOTE.txt
ssh tufts-login 'squeue -u tawal01'
echo "submitted $JOB_ID at $REMOTE"
