#!/usr/bin/env bash
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
LANE=experiments/jcp-paired-cost
RUN_ID=${1:?usage: pull.sh pNN}
[[ "$RUN_ID" =~ ^p[0-9]{2}$ ]] || exit 2
LOCAL="$ROOT/$LANE/runs/$RUN_ID"
REMOTE=$(cat "$LOCAL/REMOTE.txt")
JOB_ID=$(cat "$LOCAL/JOB_ID.txt")
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || exit 2
[[ "$REMOTE" == "/cluster/tufts/paralab/tawal01/jcp-paired-cost-20261118-488b0837/$RUN_ID" ]] || exit 2
ACTIVE=$(ssh tufts-login "squeue -h -j '$JOB_ID' -o '%i'")
[[ -z "$ACTIVE" ]] || { echo 'job still queued/running; do not collect terminal evidence' >&2; exit 3; }
ssh tufts-login "sacct -j '$JOB_ID' --format=JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList,ExitCode -P" > "$LOCAL/sacct.txt"
# Caches and temporary compiler products are not scientific outputs. Preserve all
# code, generated model assets, configs, raw measurements and logs before deletion.
ssh tufts-login bash -s -- "$REMOTE" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$1"
ITEMS=(code COMMIT.txt INPUTS.sha256 job.sbatch job.log)
if [[ -d output ]]; then ITEMS+=(output); fi
find "${ITEMS[@]}" -type f -print0 | sort -z | xargs -0 -r sha256sum > COMPLETE.sha256
tar -czf result.tar.gz "${ITEMS[@]}" COMPLETE.sha256
sha256sum result.tar.gz > RESULT_ARCHIVE.sha256
REMOTE_SCRIPT
scp "tufts-login:$REMOTE/result.tar.gz" "tufts-login:$REMOTE/RESULT_ARCHIVE.sha256" "$LOCAL/"
cd "$LOCAL"
sha256sum -c RESULT_ARCHIVE.sha256
tar -xzf result.tar.gz
sha256sum -c COMPLETE.sha256 > verification.txt
# Verify the exact owned path again immediately before removing only the duplicate.
[[ "$REMOTE" == "/cluster/tufts/paralab/tawal01/jcp-paired-cost-20261118-488b0837/$RUN_ID" ]]
ssh tufts-login "test -f '$REMOTE/COMPLETE.sha256' && rm -rf -- '$REMOTE'"
echo "checksum-verified and removed remote duplicate: $JOB_ID $RUN_ID"
