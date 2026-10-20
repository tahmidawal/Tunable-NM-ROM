#!/bin/bash
# jcp-fom2d: submit ONE staged attempt. Refuses if f2d_<attempt> is queued or its remote dir exists, if the lane has
# >= 3 jobs or the account >= 8 jobs (campaign cap), or if the atomic GPU-hour reservation (DESIGN A1.11, A2.6) fails.
# The reserved hours are the sbatch --time limit read from run.sbatch (checked against RESERVATION.json).
# squeue before and after (duplicate-submit rule).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
PY=/home/tahmid/Dev/.venv/bin/python
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-fom2d
LOCAL="$LANE/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
exec 9>"$LANE/runs/SUBMIT.lock"; flock 9          # one lane submission at a time: cap checks + reservation + sbatch
TL=$(grep -oP '(?<=#SBATCH --time=)\S+' "$LOCAL/run.sbatch")
HOURS=$("$PY" -c "h,m,s='$TL'.split(':');print(int(h)+int(m)/60+int(s)/3600)")
RH=$("$PY" -c "import json;print(json.load(open('$LOCAL/RESERVATION.json'))['time_limit_hours'])")
"$PY" -c "import sys; sys.exit(0 if abs($HOURS-$RH)<1e-9 else 1)" || { echo "time limit $HOURS != reservation $RH"; exit 7; }
Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -h -o "%i %j %T"') || { echo "squeue failed"; exit 9; }
NLANE=$(echo "$Q" | awk '$2 ~ /^f2d_/' | grep -c . || true)
NALL=$(echo "$Q" | grep -c . || true)
echo "--- squeue before (lane=$NLANE account=$NALL)"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "f2d_$A"; then echo "f2d_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 3 ] || { echo "lane cap 3 reached"; exit 4; }
[ "$NALL" -lt 8 ] || { echo "campaign cap 8 reached"; exit 5; }
QA=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o %i") || { echo "squeue failed"; exit 9; }
ACC=$(echo "$QA" | grep -c . || true)
# A5: lane-side margin against the cross-lane race (no shared lock exists): submit only with >= 2 free campaign slots
[ "$ACC" -le 6 ] || { echo "account at $ACC jobs: margin rule (<= 6 before submit, campaign cap 8)"; exit 5; }
"$PY" "$LANE/ledger.py" reserve "$A" "$HOURS" || { echo "GPU-hour reservation refused"; exit 6; }
# before sbatch is attempted a failure is unambiguous: release; after, keep the reservation (AMBIGUOUS) until squeue/sacct
# by the unique job name f2d_<attempt> settles it (ledger.py settle)
trap '"$PY" "$LANE/ledger.py" release "$A"; echo "staging failed before sbatch: reservation released"' ERR
timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet"
QR=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o %i") || { "$PY" "$LANE/ledger.py" release "$A"; echo "squeue failed: released"; exit 9; }
ACCR=$(echo "$QR" | grep -c . || true)
[ "$ACCR" -le 6 ] || { "$PY" "$LANE/ledger.py" release "$A"; echo "account at $ACCR jobs just before sbatch: margin rule, released"; exit 5; }   # A5.2
trap '"$PY" "$LANE/ledger.py" ambiguous "$A"; echo "sbatch outcome unknown: reservation KEPT; run ledger.py settle $A"' ERR
OUT=$(timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sbatch run.sbatch")
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/ {print $4}')
[ -n "$JID" ]
trap - ERR
"$PY" "$LANE/ledger.py" bind "$A" "$JID"
# campaign cap (8) is account-wide but the lock above is lane-local: re-check after submitting; if the account is now
# above 8 (a concurrent submit by another lane), cancel this job (the reservation is settled by reconcile as CANCELLED)
QB=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o %i") || { echo "post-submit squeue failed: CHECK THE ACCOUNT NOW"; exit 10; }
ACC2=$(echo "$QB" | grep -c . || true)
if [ "$ACC2" -gt 8 ]; then
  echo "account at $ACC2 > 8 after submit: cancelling $JID"
  timeout 60 ssh -o ConnectTimeout=20 tufts-login "scancel $JID"
fi
echo "--- squeue after"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
