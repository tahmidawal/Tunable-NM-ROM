#!/bin/bash
# submit.sh <job-name> <config (relative to experiments/ns3d-test)> <gres> <constraint> <mem> [jax fraction]
set -euo pipefail
JOB="$1"; CONFIG="$2"; GRES="$3"; CONS="$4"; MEM="$5"
NS=/cluster/tufts/paralab/tawal01/ns3test_20260925
DEST="$NS/$JOB"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "--- squeue BEFORE"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
"$HERE/stage.sh" "$DEST"
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=ns3test_$JOB --gres=$GRES --constraint=$CONS --mem=$MEM \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err experiments/ns3d-test/cluster/test.sbatch $CONFIG ${6:-0.5}")
echo "submitted job $ID into $DEST"
sleep 3
echo "--- squeue AFTER"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
mkdir -p "$HERE/../runs"
echo "$JOB $ID $GRES $CONS $(date -Iseconds)" >> "$HERE/../runs/JOBS.txt"
