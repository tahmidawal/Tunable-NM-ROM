#!/bin/bash
# Stage committed code + frozen inputs for ONE mesh into a fresh job dir and submit.
# Usage: submit.sh <job q32|q64|q96> <mem> <time>
set -euo pipefail
JOB="$1"; MEM="$2"; TIME="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
L=experiments/ns3d-offmesh-quadrature
NS=/cluster/tufts/paralab/tawal01/ns3d-offmesh-q-20261105
DEST="$NS/$JOB"
CFG="$L/configs/$JOB.json"
if [[ -n "$(git -C "$HERE" status --porcelain -- experiments/)" ]]; then
  echo "refusing: uncommitted changes under experiments/"; git -C "$HERE" status --short -- experiments/; exit 1
fi
if ssh tufts-login "test -e $DEST"; then echo "refusing: $DEST exists (dirs are never reused)"; exit 1; fi
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments"
for pkg in ns3d ns2d separable-decoder ns3d-grok ns3d-shift ns3d-shift-head ns3d-coordnet; do
  rsync -az --exclude='runs/' --exclude='__pycache__/' --include='*/' --include='*.py' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
rsync -az --exclude='smoke/' --exclude='runs/' --exclude='__pycache__/' "$HERE/$L/" "tufts-login:$DEST/$L/"
for F1 in $(/home/tahmid/Dev/.venv/bin/python -c "import json,sys; c=json.load(open(sys.argv[1])); print(c['bank_file'], c['head_file'], c['reference_summary'])" "$HERE/$CFG"); do
  rsync -azR "$HERE/./$F1" "tufts-login:$DEST/"
done
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
echo "--- squeue BEFORE"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --constraint=${GPU_CONSTRAINT:-a100-80G} --job-name=offq_$JOB --gres=gpu:1 --mem=$MEM --time=$TIME \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err $L/cluster/job.sbatch $CFG")
echo "submitted job $ID into $DEST"
sleep 5
echo "--- squeue AFTER"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
echo "$ID"
