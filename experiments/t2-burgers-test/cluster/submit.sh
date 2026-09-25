#!/bin/bash
# t2-burgers-test: submit ONE staged attempt with the protocol's guards (adapted from burgers-compare-hires).
#   cluster/submit.sh <attempt>
# Refuses if t2b_<attempt> is already queued or its remote dir exists; squeue before AND after; touches no other job.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/t2btest_20260925
LOCAL="$ROOT/experiments/t2-burgers-test/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "t2b_$A"; then echo "t2b_$A already queued: refusing"; exit 3; fi
ssh tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
