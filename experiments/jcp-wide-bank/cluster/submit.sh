#!/bin/bash
# jcp-wide-bank: submit ONE staged attempt. Refuses if jw_<attempt> is queued, its remote dir exists, or the lane
# already has a job running or pending (lane cap = 1 concurrent cluster job). squeue before and after.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcpwide
LOCAL="$ROOT/experiments/jcp-wide-bank/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"') || { echo "squeue failed: refusing"; exit 6; }
NLANE=$(echo "$Q" | awk '$2 ~ /^jw_/' | grep -c . || true)     # any non-terminal state counts (squeue lists only those)
echo "--- squeue before (lane running+pending=$NLANE)"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "jw_$A"; then echo "jw_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 1 ] || { echo "lane cap (1) reached"; exit 4; }
ssh tufts-login "mkdir $NS/.submit.lock" || { echo "another lane submission in progress"; exit 5; }
trap 'ssh tufts-login "rmdir $NS/.submit.lock"' EXIT
QL=$(ssh tufts-login 'squeue -u $USER -h -o "%j"') || { echo "squeue failed: refusing (fail closed)"; exit 6; }
N2=$(printf '%s\n' "$QL" | grep -c '^jw_' || true)
[ "$N2" -lt 1 ] || { echo "lane cap (1) reached (re-check under lock)"; exit 4; }
ssh tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
