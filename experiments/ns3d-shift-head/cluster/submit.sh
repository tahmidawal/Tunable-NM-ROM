#!/bin/bash
# Stage this worktree's python into ONE fresh job directory and submit.
# Usage: submit.sh <attempt_job> <config> <gpu-gres> <mem> [frozen-dir-relative]
set -euo pipefail
JOB="$1"; CONFIG="$2"; GRES="$3"; MEM="$4"; FROZEN="${5:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NS=/cluster/tufts/paralab/tawal01/nshead_20260923
DEST="$NS/$JOB"
if ssh tufts-login "test -e $DEST"; then echo "refusing: $DEST exists (dirs are never reused)"; exit 1; fi
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments"
for pkg in ns3d ns2d separable-decoder ns3d-grok ns3d-shift ns3d-shift-head; do
  rsync -az --delete \
    --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' \
    --exclude='__pycache__/' \
    --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' \
    --include='*.sh' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
if [[ -n "$FROZEN" ]]; then
  rsync -az "$HERE/$FROZEN/" "tufts-login:$DEST/frozen/"
fi
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
echo "--- squeue BEFORE"
ssh tufts-login 'squeue -u tawal01 -o "%.10i %.24j %.8T %.10M %R"'
ARGS="$JOB experiments/ns3d-shift-head/$CONFIG"
[[ -n "$FROZEN" ]] && ARGS="$ARGS frozen"
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=nshead_$JOB --gres=$GRES --mem=$MEM \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err \
  experiments/ns3d-shift-head/cluster/job.sbatch $ARGS")
echo "submitted job $ID into $DEST"
sleep 5
echo "--- squeue AFTER"
ssh tufts-login 'squeue -u tawal01 -o "%.10i %.24j %.8T %.10M %R"'
echo "$ID"
