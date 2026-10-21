#!/bin/bash
# Submit ONE staged attempt: lane cap 2 (imm_ jobs), campaign cap 8 through the shared lock, ledger check against the
# smoke-derived projection (arg 2, hours), squeue before and after. Usage: submit.sh <attempt> <projected_hours>
set -euo pipefail
A="$1"; PROJ="$2"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
PY=/home/tahmid/Dev/.venv/bin/python
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/imp-multimesh
LOCAL="$LANE/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
TL=$(grep -oP '(?<=#SBATCH --time=)\S+' "$LOCAL/run.sbatch")
HOURS=$("$PY" -c "h,m,s='$TL'.split(':');print(int(h)+int(m)/60+int(s)/3600)")
"$PY" "$LANE/cluster/ledger.py" check "$A" "$PROJ"
Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
NLANE=$(echo "$Q" | awk '$2 ~ /^imm_/' | grep -c . || true)
if echo "$Q" | awk '{print $2}' | grep -qx "imm_$A"; then echo "imm_$A already queued: refusing"; exit 3; fi
[ "$NLANE" -lt 2 ] || { echo "lane cap 2 reached"; exit 4; }
timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
timeout 300 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet"
OUT=$(timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c 'n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; cd $NS/$A && sbatch --no-requeue run.sbatch'") || { rc=$?; echo "sbatch refused rc=$rc (75 = account at cap)"; exit $rc; }
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/ {print $4}')
[ -n "$JID" ]
"$PY" "$LANE/cluster/ledger.py" bind "$A" "$JID" "$HOURS"
echo "--- squeue after"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
echo "JOBID=$JID"
