#!/bin/bash
# t3-burgers-test: submit ONE staged attempt (burgers-compare-hires/cluster/submit.sh @ 07c0e3e8, t3bt_ namespace, cap 3).
#   cluster/submit.sh <attempt>
# Refuses if t3bt_<attempt> is already queued or its remote dir exists; squeue before AND after; never touches other jobs.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/t3btest_20260925
LOCAL="$ROOT/experiments/t3-burgers-test/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
if echo "$Q" | awk '{print $2}' | grep -qx "t3bt_$A"; then echo "t3bt_$A already queued: refusing"; exit 3; fi
NLANE=$(echo "$Q" | awk '$2 ~ /^t3bt_/' | grep -c . || true)
[ "$NLANE" -lt 3 ] || { echo "lane cap reached"; exit 4; }
echo "--- squeue before"; echo "$Q"
ssh tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
