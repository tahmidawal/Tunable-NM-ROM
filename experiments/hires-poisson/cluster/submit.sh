#!/bin/bash
# usage: cluster/submit.sh <attempt>   (after cluster/stage.py). One job per directory;
# squeue is printed before and after; refuses if the remote directory already exists.
set -euo pipefail
A=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/hires_p_20260920
echo "--- squeue before"; ssh tufts-login "squeue -u tawal01 -o '%.10i %.14j %.8T %.10M %Z'"
RUNNING=$(ssh tufts-login "squeue -u tawal01 -h -t RUNNING | wc -l")
[ "$RUNNING" -lt 6 ] || { echo "account already has $RUNNING running jobs"; exit 3; }
ssh tufts-login "mkdir -p $NS && test ! -e $NS/$A"
rsync -a "$HERE/runs/$A/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch --parsable run.sbatch" | tee "$HERE/runs/$A/JOBID.txt"
echo "--- squeue after"; ssh tufts-login "squeue -u tawal01 -o '%.10i %.14j %.8T %.10M %Z'"
