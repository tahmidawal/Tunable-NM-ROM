#!/bin/bash
# jcp-regimes: submit ONE staged attempt (pattern of jcp-qualify's submit.sh). Refuses if jg_<attempt> is queued or its
# remote dir exists, if the lane has >= 3 jobs (lane cap), or if the GPU-hour reservation fails. The sbatch itself runs
# under the campaign-wide flock with the account cap of 8 (COMMON-BRIEF §14). squeue before and after.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
PY=/home/tahmid/Dev/.venv/bin/python
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-regimes
LOCAL="$LANE/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
exec 9>"$LANE/runs/SUBMIT.lock"; flock 9
TL=$(grep -oP '(?<=#SBATCH --time=)\S+' "$LOCAL/run.sbatch")
HOURS=$("$PY" -c "h,m,s='$TL'.split(':');print(int(h)+int(m)/60+int(s)/3600)")
RH=$("$PY" -c "import json;print(json.load(open('$LOCAL/RESERVATION.json'))['time_limit_hours'])")
"$PY" -c "import sys; sys.exit(0 if abs($HOURS-$RH)<1e-9 else 1)" || { echo "time limit $HOURS != reservation $RH"; exit 7; }
Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -h -o "%i %j %T"')
NLANE=$(echo "$Q" | awk '$2 ~ /^jg_/' | grep -c . || true)
NALL=$(echo "$Q" | grep -c . || true)
echo "--- squeue before (lane=$NLANE account=$NALL)"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "jg_$A"; then echo "jg_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 3 ] || { echo "lane cap 3 reached"; exit 4; }
"$PY" "$LANE/ledger.py" reserve "$A" "$HOURS" || { echo "GPU-hour reservation refused"; exit 6; }
trap '"$PY" "$LANE/ledger.py" release "$A"; echo "staging failed before sbatch: reservation released"' ERR
timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet"
trap '"$PY" "$LANE/ledger.py" ambiguous "$A"; echo "sbatch outcome unknown: reservation KEPT; settle by hand"' ERR
set +e
OUT=$(timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c 'n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; cd $NS/$A && sbatch run.sbatch'")
RC=$?
set -e
if [ $RC -eq 75 ]; then "$PY" "$LANE/ledger.py" release "$A"; echo "account at cap 8: not submitted (reservation released)"; trap - ERR; exit 75; fi
[ $RC -eq 0 ] || false
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/ {print $4}')
[ -n "$JID" ]
trap - ERR
"$PY" "$LANE/ledger.py" bind "$A" "$JID"
echo "--- squeue after"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
