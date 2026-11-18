#!/usr/bin/env bash
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
LANE=experiments/jcp-paired-cost
RUN_ID=${1:?usage: submit.sh pNN [pilot.json|refined_dev16.json|three_head_dev16.json] [30|45]}
CONFIG_NAME=${2:-pilot.json}
JOB_MINUTES=${3:-30}
[[ "$RUN_ID" =~ ^p[0-9]{2}$ ]] || { echo 'require exact run id pNN' >&2; exit 2; }
[[ "$CONFIG_NAME" == pilot.json || "$CONFIG_NAME" == refined_dev16.json || "$CONFIG_NAME" == three_head_dev16.json ]] || exit 2
[[ "$JOB_MINUTES" == 30 || "$JOB_MINUTES" == 45 ]] || exit 2
NS=/cluster/tufts/paralab/tawal01/jcp-paired-cost-20261118-488b0837
REMOTE="$NS/$RUN_ID"
LOCAL="$ROOT/$LANE/runs/$RUN_ID"
[[ ! -e "$LOCAL" ]] || { echo 'run directory already exists' >&2; exit 2; }
git diff --quiet HEAD -- "$LANE" || { echo 'commit lane code before staging' >&2; exit 2; }
mkdir -p "$LOCAL"
git rev-parse HEAD > "$LOCAL/COMMIT.txt"
git archive --format=tar.gz --output="$LOCAL/code.tar.gz" HEAD "$LANE"
cp "$LANE/cluster/job.sbatch" "$LOCAL/job.sbatch"
sed -i "s/ctol_jpc_p01/ctol_jpc_$RUN_ID/" "$LOCAL/job.sbatch"
cd "$LOCAL"
sha256sum COMMIT.txt code.tar.gz job.sbatch > INPUTS.sha256
ssh tufts-login 'squeue -u tawal01; df -h /cluster/tufts/paralab'
ACTIVE=$(ssh tufts-login "squeue -u tawal01 -h -o '%j'" | /usr/bin/awk '/^ctol_j(hm|wv|pc)_/{n++} END{print n+0}')
[[ "$ACTIVE" -lt 2 ]] || { echo 'two campaign jobs already active; defer submission' >&2; exit 3; }
ssh tufts-login "mkdir -p '$NS'; mkdir '$REMOTE'"
scp COMMIT.txt code.tar.gz job.sbatch INPUTS.sha256 "tufts-login:$REMOTE/"
ssh tufts-login "cd '$REMOTE' && sha256sum -c INPUTS.sha256 && mkdir code && tar -xzf code.tar.gz -C code"
JOB_ID=$(ssh tufts-login "cd '$REMOTE' && sbatch --parsable --time=00:$JOB_MINUTES:00 job.sbatch 'experiments/jcp-paired-cost/configs/$CONFIG_NAME'")
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "unexpected scheduler reply: $JOB_ID" >&2; exit 4; }
printf '%s\n' "$JOB_ID" > JOB_ID.txt
printf '%s\n' "$REMOTE" > REMOTE.txt
ssh tufts-login 'squeue -u tawal01'
echo "submitted $JOB_ID at $REMOTE"
