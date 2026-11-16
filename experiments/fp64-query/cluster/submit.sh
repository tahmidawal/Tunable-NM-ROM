#!/bin/bash
# fp64-query: submit ONE staged attempt; refuses a duplicate name or an existing remote dir; lane cap 2 GPU jobs.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/fp64-query-20261115
[ -f "$LANE/runs/$A/run.sbatch" ]
Q=$(ssh -o BatchMode=yes tufts-login 'squeue -u $USER -h -o "%i %j %T"')
N=$(echo "$Q" | awk '$2 ~ /^f64_/' | grep -c . || true)
echo "--- before (lane jobs=$N, account jobs=$(echo "$Q" | grep -c . || true))"; echo "$Q" | awk '$2 ~ /^f64_/' || true
if echo "$Q" | awk '{print $2}' | grep -qx "f64_$A"; then echo "already queued"; exit 3; fi
[ "$N" -lt 2 ] || { echo "lane cap reached"; exit 4; }
ssh -o BatchMode=yes tufts-login "mkdir -p $NS && test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LANE/runs/$A/" "tufts-login:$NS/$A/"
ssh -o BatchMode=yes tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- after"; ssh -o BatchMode=yes tufts-login 'squeue -u $USER -h -o "%i %j %T %R"' | awk '$2 ~ /^f64_/'
