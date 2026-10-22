#!/bin/bash
# imp-hrcost: submit ONE staged attempt through the campaign lock (COMMON-BRIEF 6). Refuses if ihc_<attempt> is queued,
# if the lane already has a job (lane cap 1), or if the remote dir exists. squeue before and after.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/imp-hrcost
LOCAL="$LANE/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -q "^ihc_"; then echo "lane cap 1: an ihc_ job is queued/running"; exit 4; fi
timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet"
HRS=$(/home/tahmid/Dev/.venv/bin/python -c "import json;print(json.load(open('$LOCAL/RESERVATION.json'))['time_limit_hours'])")
/home/tahmid/Dev/.venv/bin/python "$LANE/ledger.py" reserve "$A" "$HRS" > /dev/null
# lane cap (1) and campaign cap (8) are both checked INSIDE the shared lock (code audit 1)
OUT=$(timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c 'n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; l=\$(squeue -u tawal01 -h -o %j | grep -c ^ihc_); [ \$l -lt 1 ] || exit 76; cd $NS/$A && sbatch run.sbatch'") || { rc=$?; echo "sbatch refused rc=$rc (reservation kept until released by hand: ledger.py)"; exit $rc; }
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/ {print $4}')
[ -n "$JID" ]
echo "$JID" > "$LOCAL/JOBID"
echo "--- squeue after"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
