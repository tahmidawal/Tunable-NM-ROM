#!/bin/bash
# Stage this worktree's python into one isolated job directory and submit.
# Usage: submit.sh <jobname>     (jobname must match cluster/<jobname>.sbatch)
set -euo pipefail
JOB="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NS=/cluster/tufts/paralab/tawal01/ns3dshift_20260922
DEST="$NS/$JOB"

ssh tufts-login "mkdir -p $DEST/logs $DEST/output"
for pkg in ns3d ns2d separable-decoder ns3d-grok ns3d-shift; do
  rsync -az --delete \
    --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' \
    --include='*.sh' --exclude='*' \
    --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"

echo "--- squeue BEFORE"
ssh tufts-login 'squeue -u tawal01 -o "%.10i %.22j %.8T %.10M %R"'
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable experiments/ns3d-shift/cluster/$JOB.sbatch")
echo "submitted job $ID into $DEST"
sleep 5
echo "--- squeue AFTER"
ssh tufts-login 'squeue -u tawal01 -o "%.10i %.22j %.8T %.10M %R"'
echo "$ID"
