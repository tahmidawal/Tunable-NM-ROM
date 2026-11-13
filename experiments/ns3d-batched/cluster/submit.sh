#!/bin/bash
# Stage the lane's python into ONE fresh job directory and submit.
# Usage: submit.sh <job> <mem> <time> [args to batched.py...]
set -euo pipefail
JOB="$1"; MEM="$2"; TIME="$3"; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NS=/cluster/tufts/paralab/tawal01/rebuttal-ns3d-batched
DEST="$NS/$JOB"
if ssh tufts-login "test -e $DEST"; then echo "refusing: $DEST exists (dirs are never reused)"; exit 1; fi
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments"
for pkg in ns3d ns2d separable-decoder ns3d-grok ns3d-shift ns3d-shift-head ns3d-operators ns3d-hard ns3d-batched; do
  rsync -az --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' \
    --exclude='__pycache__/' --exclude='figures/' --exclude='cluster/' \
    --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' --include='*.sh' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
rsync -az "$HERE/experiments/ns3d-batched/cluster/" "tufts-login:$DEST/experiments/ns3d-batched/cluster/"
git -c gc.auto=0 -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=nb_$JOB --gres=gpu:1 --constraint="a100-80G|h200-141G" --mem=$MEM --time=$TIME \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err experiments/ns3d-batched/cluster/job.sbatch $*")
echo "submitted job $ID into $DEST"
