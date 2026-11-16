#!/bin/bash
# b2d-query-speed: submit ONE staged attempt; refuses a duplicate name or an existing remote dir; lane cap 3 GPU jobs.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/b2d-query-speed-20261114
[ -f "$LANE/runs/$A/run.sbatch" ]
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
N=$(echo "$Q" | awk '$2 ~ /^qs_/' | grep -c . || true)
echo "--- before (lane jobs=$N, account jobs=$(echo "$Q" | grep -c . || true))"; echo "$Q" | awk '$2 ~ /^qs_/' || true
if echo "$Q" | awk '{print $2}' | grep -qx "qs_$A"; then echo "already queued"; exit 3; fi
[ "$N" -lt 3 ] || { echo "lane cap reached"; exit 4; }
ssh tufts-login "mkdir -p $NS && test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LANE/runs/$A/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- after"; ssh tufts-login 'squeue -u $USER -h -o "%i %j %T %R"' | awk '$2 ~ /^qs_/'
