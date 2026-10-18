#!/bin/bash
# jcp-heads (copy of jcp-qualify submit.sh; edited: namespace, prefix jh_, lane cap 2): submit ONE staged attempt.
# Refuses if jh_<attempt> is queued or its remote dir exists, if the lane has
# >= 2 jobs or the account >= 8 jobs (campaign cap), or if the atomic GPU-hour reservation (DESIGN A1.11, A2.6) fails.
# The reserved hours are the sbatch --time limit read from run.sbatch (checked against RESERVATION.json).
# squeue before and after (duplicate-submit rule).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
PY=/home/tahmid/Dev/.venv/bin/python
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-heads
LOCAL="$LANE/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
exec 9>"$LANE/runs/SUBMIT.lock"; flock 9          # one lane submission at a time: cap checks + reservation + sbatch
TL=$(grep -oP '(?<=#SBATCH --time=)\S+' "$LOCAL/run.sbatch")
HOURS=$("$PY" -c "h,m,s='$TL'.split(':');print(int(h)+int(m)/60+int(s)/3600)")
RH=$("$PY" -c "import json;print(json.load(open('$LOCAL/RESERVATION.json'))['time_limit_hours'])")
"$PY" -c "import sys; sys.exit(0 if abs($HOURS-$RH)<1e-9 else 1)" || { echo "time limit $HOURS != reservation $RH"; exit 7; }
Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -h -o "%i %j %T"')
NLANE=$(echo "$Q" | awk '$2 ~ /^jh_/' | grep -c . || true)
NALL=$(echo "$Q" | grep -c . || true)
echo "--- squeue before (lane=$NLANE account=$NALL)"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "jh_$A"; then echo "jh_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 2 ] || { echo "lane cap 2 reached"; exit 4; }
if [[ "$A" == smk* ]] && echo "$Q" | awk '{print $2}' | grep -q '^jh_smk'; then echo "a lane smoke is already queued/running: one at a time"; exit 9; fi
[ "$NALL" -lt 8 ] || { echo "campaign cap 8 reached"; exit 5; }
"$PY" "$LANE/ledger.py" reserve "$A" "$HOURS" || { echo "GPU-hour reservation refused"; exit 6; }
# before sbatch is attempted a failure is unambiguous: release; after, keep the reservation (AMBIGUOUS) until squeue/sacct
# by the unique job name jh_<attempt> settles it (ledger.py settle)
trap '"$PY" "$LANE/ledger.py" release "$A"; echo "staging failed before sbatch: reservation released"' ERR
timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet"
trap '"$PY" "$LANE/ledger.py" ambiguous "$A"; echo "sbatch outcome unknown: reservation KEPT; run ledger.py settle $A"' ERR
# account cap re-checked on the login node immediately before sbatch (narrows the cross-lane race to one ssh command)
OUT=$(timeout 120 ssh -o ConnectTimeout=20 tufts-login "n=\$(squeue -u \$USER -h | wc -l); if [ \$n -ge 8 ]; then echo CAP8 \$n; exit 0; fi; cd $NS/$A && sbatch run.sbatch")
echo "$OUT"
if echo "$OUT" | grep -q '^CAP8'; then          # nothing was submitted: release the reservation and the remote dir
  trap - ERR
  "$PY" "$LANE/ledger.py" release "$A"
  timeout 120 ssh -o ConnectTimeout=20 tufts-login "rm -rf $NS/$A"
  echo "campaign cap 8 reached at sbatch time: not submitted"; exit 5
fi
JID=$(echo "$OUT" | awk '/Submitted batch job/ {print $4}')
[ -n "$JID" ]
trap - ERR
"$PY" "$LANE/ledger.py" bind "$A" "$JID"
echo "--- squeue after"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
