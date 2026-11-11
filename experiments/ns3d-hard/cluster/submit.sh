#!/bin/bash
# Stage this lane's python into ONE fresh job directory and submit.
# Usage: submit.sh <job> <gpu-gres> <mem> <time> <script.py> [args...]
set -euo pipefail
JOB="$1"; GRES="$2"; MEM="$3"; TIME="$4"; shift 4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NS=/cluster/tufts/paralab/tawal01/rebuttal-ns3d-hard
DEST="$NS/$JOB"
if ssh tufts-login "test -e $DEST"; then echo "refusing: $DEST exists (dirs are never reused)"; exit 1; fi
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments"
for pkg in ns3d ns2d separable-decoder ns3d-grok ns3d-shift ns3d-shift-head ns3d-operators ns3d-hard; do
  rsync -az --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' \
    --exclude='__pycache__/' --exclude='results/' --exclude='figures/' \
    --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' --include='*.sh' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
git -c gc.auto=0 -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=nh_$JOB --gres=$GRES --mem=$MEM --time=$TIME \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err experiments/ns3d-hard/cluster/job.sbatch $*")
echo "submitted job $ID into $DEST"
