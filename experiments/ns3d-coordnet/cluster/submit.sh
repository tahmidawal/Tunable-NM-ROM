#!/bin/bash
# Stage this worktree's python + configs + frozen inputs into ONE fresh job directory and
# submit. Usage: submit.sh <job> <mode bank|rom|test> <config-relative-to-lane> <gres> <mem> <time> [frozen-dir-relative-to-lane]
set -euo pipefail
JOB="$1"; MODE="$2"; CONFIG="$3"; GRES="$4"; MEM="$5"; TIME="$6"; FROZEN="${7:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NS=/cluster/tufts/paralab/tawal01/nscoord_20261001
DEST="$NS/$JOB"
if ssh tufts-login "test -e $DEST"; then echo "refusing: $DEST exists (dirs are never reused)"; exit 1; fi
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments"
for pkg in ns3d ns2d separable-decoder ns3d-grok ns3d-shift ns3d-shift-head ns3d-operators ns3d-coordnet; do
  rsync -az \
    --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' \
    --exclude='__pycache__/' --exclude='deps/' --exclude='operators/' \
    --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' \
    --include='*.sh' --include='frozen/**' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
# every coordnet bank file a config names (repo-relative), staged at the same path
for C in ${CONFIG//,/ }; do
  BF=$(/home/tahmid/Dev/.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('coordnet_bank_file',''))" "$HERE/experiments/ns3d-coordnet/$C")
  if [[ -n "$BF" ]]; then
    [[ -f "$HERE/$BF" ]] || { echo "bank file $BF missing"; exit 1; }
    rsync -azR "$HERE/./$BF" "tufts-login:$DEST/"
  fi
done
if [[ -n "$FROZEN" ]]; then
  rsync -az "$HERE/experiments/ns3d-coordnet/$FROZEN/" "tufts-login:$DEST/frozen/"
fi
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
echo "--- squeue BEFORE"
ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
CFGS=""
for C in ${CONFIG//,/ }; do CFGS="$CFGS,experiments/ns3d-coordnet/$C"; done
ARGS="$MODE ${CFGS#,}"
[[ -n "$FROZEN" ]] && ARGS="$ARGS frozen"
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=nscoord_$JOB --gres=$GRES --mem=$MEM --time=$TIME \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err \
  experiments/ns3d-coordnet/cluster/job.sbatch $ARGS")
echo "submitted job $ID into $DEST"
sleep 5
echo "--- squeue AFTER"
ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
echo "$ID"
