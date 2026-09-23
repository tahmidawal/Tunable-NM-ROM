#!/bin/bash
# usage: cluster/submit.sh <attempt>   (after cluster/stage.py). One job per directory; squeue before and
# after; refuses if the remote directory exists or this lane already has 2 jobs queued/running.
set -euo pipefail
A=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/specfom_20260923
echo "--- squeue before"; ssh tufts-login "squeue -u tawal01 -o '%.10i %.18j %.8T %.10M %R'"
MINE=$(ssh tufts-login "squeue -u tawal01 -h -o %j | grep -c ^specfom_ || true")
[ "$MINE" -lt 2 ] || { echo "this lane already has $MINE jobs"; exit 3; }
ssh tufts-login "mkdir -p $NS && test ! -e $NS/$A"
rsync -a "$HERE/runs/$A/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch --parsable run.sbatch" | tee "$HERE/runs/$A/JOBID.txt"
echo "--- squeue after"; ssh tufts-login "squeue -u tawal01 -o '%.10i %.18j %.8T %.10M %R'"
