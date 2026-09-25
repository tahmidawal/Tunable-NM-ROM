#!/bin/bash
# submit.sh <job-name> <mesh> <config (relative to experiments/t2-ns3d-test)> [jax-frac] [--smoke]
set -euo pipefail
JOB="$1"; MESH="$2"; CONFIG="$3"; JFRAC="${4:-0.5}"; SMOKE="${5:-}"
NS=/cluster/tufts/paralab/tawal01/t2ntest_20260925
DEST="$NS/$JOB"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "--- squeue BEFORE"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
"$HERE/stage.sh" "$DEST" "$MESH"
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=t2nt_$JOB --gres=gpu:a100:1 --constraint=a100-80G --mem=64G \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err experiments/t2-ns3d-test/cluster/panel.sbatch $CONFIG $JFRAC $SMOKE")
echo "submitted job $ID into $DEST"
sleep 3
echo "--- squeue AFTER"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"'
echo "$JOB $ID gpu:a100:1 a100-80G $(date -Iseconds)" >> "$HERE/../runs/JOBS.txt"
