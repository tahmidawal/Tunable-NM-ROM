#!/bin/bash
# burgers-repanel: submit ONE staged attempt with the protocol's guards.
#   cluster/submit.sh <attempt>
# Waits until the account has < 6 running jobs and this lane (brep_*) has NONE running or pending (lane
# budget: <= 1 running job); refuses if a job named
# brep_<attempt> is already queued; rsyncs runs/<attempt>/ to its own namespace dir; squeue before and after.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/brepanel_20260922
LOCAL="$ROOT/experiments/burgers-repanel/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
while true; do
  Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
  NRUN=$(echo "$Q" | awk '$3=="RUNNING"' | grep -c . || true)
  NLANE=$(echo "$Q" | awk '$2 ~ /^brep_/ && ($3=="RUNNING" || $3=="PENDING")' | grep -c . || true)
  echo "$(date +%T) account running=$NRUN lane running+pending=$NLANE"
  if echo "$Q" | awk '{print $2}' | grep -qx "brep_$A"; then echo "brep_$A already queued: refusing"; exit 3; fi
  if [ "$NRUN" -lt 6 ] && [ "$NLANE" -lt 1 ]; then break; fi
  sleep 120
done
echo "--- squeue before"; echo "$Q"
ssh tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
