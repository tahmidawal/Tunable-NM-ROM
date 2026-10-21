#!/bin/bash
# Submit ONE staged attempt (COMMON-BRIEF 6, 7). Gates: (1) production attempts need runs/<smoke>/pulled/output/ACCEPT.json
# with all_passed=true (S1 acceptance) unless the attempt IS the smoke or the probe; (2) a resource certificate
# checks/resources-<attempt>.json bound to the staged config's sha256 gives the projected hours; (3) under one local lock:
# lane cap 2 (imm_ jobs in squeue), ledger reservation; then the campaign lock + account cap 8 + sbatch; bind or release.
# Usage: submit.sh <attempt> [smoke|probe]
set -euo pipefail
A="$1"; KIND="${2:-prod}"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
case "$KIND" in prod|smoke|probe) ;; *) echo "unknown kind $KIND"; exit 2;; esac
PY=/home/tahmid/Dev/.venv/bin/python
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/imp-multimesh
LOCAL="$LANE/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
TL=$(grep -oP '(?<=#SBATCH --time=)\S+' "$LOCAL/run.sbatch")
HOURS=$("$PY" -c "h,m,s='$TL'.split(':');print(int(h)+int(m)/60+int(s)/3600)")
CERT="$LANE/checks/resources-$A.json"
[ -f "$CERT" ] || { echo "no resource certificate $CERT"; exit 8; }
PROJ=$("$PY" - "$CERT" "$LOCAL" <<'PYEOF'
import json, sys, hashlib, pathlib
c = json.load(open(sys.argv[1])); loc = pathlib.Path(sys.argv[2])
r = json.load(open(loc / 'RESERVATION.json'))
assert c.get('attempt') == loc.name, 'certificate for another attempt'
if c.get('config'):
    staged = loc / 'experiments' / 'imp-multimesh' / c['config']
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == c['config_sha256'], 'certificate not bound to this config'
else:
    assert hashlib.sha256((loc / 'run.sbatch').read_bytes()).hexdigest() == c['sbatch_sha256'], 'certificate not bound'
print(c['projected_hours'])
PYEOF
)
if [ "$KIND" = prod ]; then
  ACC=$(ls "$LANE"/runs/*/pulled/output/ACCEPT.json 2>/dev/null | head -1)
  [ -n "$ACC" ] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$ACC'))['all_passed'] else 1)" || { echo "S1 acceptance missing or failed"; exit 9; }
fi
exec 9>"$LANE/runs/SUBMIT.lock"; flock 9
Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
NLANE=$(echo "$Q" | awk '$2 ~ /^imm_/' | grep -c . || true)
if echo "$Q" | awk '{print $2}' | grep -qx "imm_$A"; then echo "imm_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 2 ] || { echo "lane cap 2 reached"; exit 4; }
"$PY" "$LANE/cluster/ledger.py" reserve "$A" "$HOURS" "$PROJ" || { echo "GPU-hour reservation refused"; exit 6; }
trap '"$PY" "$LANE/cluster/ledger.py" release "$A"; echo "failed before sbatch: reservation released"' ERR
timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
timeout 600 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet"
trap - ERR
set +e
OUT=$(timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c 'n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; cd $NS/$A && sbatch --no-requeue run.sbatch'")
RC=$?
set -e
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/ {print $4}')
if [ -z "$JID" ]; then
  if [ $RC -eq 75 ] && [ -z "$OUT" ]; then          # the remote cap check refused before sbatch ran: not submitted
    "$PY" "$LANE/cluster/ledger.py" release "$A"; echo "account at cap (75): not submitted; reservation released"; exit 7
  fi
  set +e
  Q1=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -n imm_$A -o %i"); R1=$?
  Q2=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -n -X -P --name=imm_$A -S now-1hours -o JobID"); R2=$?
  set -e
  C=$(printf "%s\n%s\n" "$Q1" "$Q2" | grep -E '^[0-9]+$' | sort -u)
  if [ $R1 -eq 0 ] && [ $R2 -eq 0 ] && [ "$(echo "$C" | grep -c .)" -eq 1 ]; then
    JID="$C"; echo "recovered job id $JID by name after an ambiguous sbatch response"
  else
    "$PY" "$LANE/cluster/ledger.py" ambiguous "$A"
    echo "submission status unknown (rc=$RC, lookups $R1/$R2, candidates: $C): reservation KEPT as AMBIGUOUS"; exit 10
  fi
fi
"$PY" "$LANE/cluster/ledger.py" bind "$A" "$JID"
echo "--- squeue after"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
echo "JOBID=$JID"
