#!/bin/bash
# burgers-eq-tol-knobs: submit ONE staged attempt. Refuses if b2eqk_<attempt> is already queued or its remote dir exists;
# squeue before and after. Lane budget: <= 4 lane jobs running or pending (account cap 10 GPUs, shared).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/b2eqk_20260924
LOCAL="$ROOT/experiments/burgers-eq-tol-knobs/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
NLANE=$(echo "$Q" | awk '$2 ~ /^b2eqk_/ && ($3=="RUNNING" || $3=="PENDING")' | grep -c . || true)
echo "--- squeue before (lane running+pending=$NLANE)"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "b2eqk_$A"; then echo "b2eqk_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 4 ] || { echo "lane budget reached"; exit 4; }
ssh tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
