#!/bin/bash
# jcp-p1-scaling: submit ONE staged attempt. Refuses if p1s_<attempt> is queued or its remote dir exists, if the lane has
# >= 4 jobs or the account >= 8 jobs (campaign cap), or if the GPU-hour reservation (DESIGN A1.11, A2.6) would exceed 20.
# squeue before and after (duplicate-submit rule). Records the job in runs/GPU-LEDGER.json.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-p1-scaling
LOCAL="$LANE/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -h -o "%i %j %T"')
NLANE=$(echo "$Q" | awk '$2 ~ /^p1s_/' | grep -c . || true)
NALL=$(echo "$Q" | grep -c . || true)
echo "--- squeue before (lane=$NLANE account=$NALL)"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "p1s_$A"; then echo "p1s_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 4 ] || { echo "lane cap 4 reached"; exit 4; }
[ "$NALL" -lt 8 ] || { echo "campaign cap 8 reached"; exit 5; }
HOURS=$(python3 -c "import json;print(json.load(open('$LOCAL/RESERVATION.json'))['time_limit_hours'])")
python3 "$LANE/ledger.py" check "$HOURS" || { echo "GPU-hour reservation would exceed the cap"; exit 6; }
timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a --exclude RESERVATION.json "$LOCAL/" "tufts-login:$NS/$A/"
OUT=$(timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch")
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/ {print $4}')
[ -n "$JID" ]
python3 "$LANE/ledger.py" add "$A" "$JID" "$HOURS"
echo "--- squeue after"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
