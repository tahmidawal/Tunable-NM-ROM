#!/bin/bash
# usage: cluster/submit.sh <attempt>   (after cluster/stage.py). One job per directory; squeue before and
# after; refuses if this lane already has 6 jobs or the remote directory exists. (from poisson-bank-knob-3d)
set -euo pipefail
A=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/pl3test_20260925
echo "--- squeue before"; ssh tufts-login "squeue -u tawal01 -o '%.10i %.22j %.8T %.10M %.12P %N'"
MINE=$(ssh tufts-login "squeue -u tawal01 -h -o %j | grep -c ^pl3t_ || true")
[ "$MINE" -lt 6 ] || { echo "this lane already has $MINE jobs"; exit 3; }
ssh tufts-login "squeue -u tawal01 -h -o %j | grep -qx pl3t_$A" && { echo "pl3t_$A already queued"; exit 4; }
ssh tufts-login "mkdir -p $NS && test ! -e $NS/$A"
rsync -a "$HERE/runs/$A/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch --parsable run.sbatch" | tee "$HERE/runs/$A/JOBID.txt"
echo "--- squeue after"; ssh tufts-login "squeue -u tawal01 -o '%.10i %.22j %.8T %.10M %.12P %N'"
