#!/bin/bash
# burgers-compare-hires: submit ONE staged attempt with the protocol's guards.
#   cluster/submit.sh <attempt>
# Refuses if bcmp_<attempt> is already queued or its remote dir exists; waits while this lane (bcmp_*) has
# 2 or more jobs running+pending (lane cap: at most 2 running); squeue before AND after; never touches other jobs.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/bcmp_20260923
LOCAL="$ROOT/experiments/burgers-compare-hires/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
while true; do
  Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
  NLANE=$(echo "$Q" | awk '$2 ~ /^bcmp_/ && ($3=="RUNNING" || $3=="PENDING")' | grep -c . || true)
  echo "$(date +%T) lane running+pending=$NLANE"
  if echo "$Q" | awk '{print $2}' | grep -qx "bcmp_$A"; then echo "bcmp_$A already queued: refusing"; exit 3; fi
  if [ "$NLANE" -lt 2 ]; then break; fi
  sleep 120
done
echo "--- squeue before"; echo "$Q"
ssh tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
