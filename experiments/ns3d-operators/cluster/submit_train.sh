#!/bin/bash
# submit_train.sh <job-name> <mesh> <arm> <gres> <constraint> <mem>
set -euo pipefail
JOB="$1"; MESH="$2"; ARM="$3"; GRES="$4"; CONS="$5"; MEM="$6"
NS=/cluster/tufts/paralab/tawal01/nsops_20260923
DEST="$NS/$JOB"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/stage.sh" "$DEST"
echo "--- squeue BEFORE"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.24j %.8T %.10M %R"'
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=nsops_$JOB --gres=$GRES --constraint=$CONS --mem=$MEM \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err experiments/ns3d-operators/cluster/train.sbatch $MESH $ARM")
echo "submitted job $ID into $DEST"
sleep 3
echo "--- squeue AFTER"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.24j %.8T %.10M %R"'
echo "$JOB $ID" >> "$HERE/../runs/JOBS.txt"
