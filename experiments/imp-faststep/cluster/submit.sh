#!/usr/bin/env bash
# Submit one staged imp-faststep job (COMMON-BRIEF 6/7; code audits 1-2). Usage: submit.sh <job>
# Local lane lock -> ledger admission on the staged time limit -> durable INTENT.json (reserved by the ledger until
# reconciled) -> upload to a private temporary directory -> under the campaign sbatch lock the staged
# .submit_remote.sh refuses an existing job dir, failed queue queries (70), the campaign cap (75) or the lane cap (76),
# moves the upload into place, submits, and writes the job id to <job dir>/JOBID (a rejected sbatch removes the dir).
set -euo pipefail
JOB=$1
PY=/home/tahmid/Dev/.venv/bin/python
LANE_DIR=$(cd "$(dirname "$0")/.." && pwd)
ST="$LANE_DIR/runs/$JOB/stage"
NS=/cluster/tufts/paralab/tawal01/imp-faststep
REMOTE=$NS/$JOB
TMP=$NS/.upload-$JOB-$$
[ -f "$ST/run.sbatch" ] && [ -f "$ST/.submit_remote.sh" ] || { echo "not staged: $JOB"; exit 2; }
exec 9>"$LANE_DIR/runs/.submit.lock"
flock -w 600 9 || { echo "lane submit lock busy"; exit 77; }
[ -f "$LANE_DIR/runs/$JOB/SUBMIT.json" ] && { echo "already submitted: $JOB"; exit 3; }
[ -f "$LANE_DIR/runs/$JOB/INTENT.json" ] && { echo "unresolved intent for $JOB: reconcile first (ssh cat $REMOTE/JOBID)"; exit 3; }
HOURS=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['time_limit_hours'])" "$ST/RESERVATION.json")
LEDGER_LOCKED=1 "$PY" "$LANE_DIR/cluster/ledger.py" check "$HOURS" || { echo "ledger refuses $HOURS h"; exit 78; }
echo "squeue before:"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o "%i %j %T %M %Z"'
timeout 60 ssh -o ConnectTimeout=20 tufts-login "mkdir -p $NS && test ! -e $REMOTE && test ! -e $TMP"
"$PY" -c "import json,sys,datetime; json.dump(dict(job=sys.argv[2], time_limit_hours=float(sys.argv[3]), remote=sys.argv[4], utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')), open(sys.argv[1],'w'))" \
  "$LANE_DIR/runs/$JOB/INTENT.json" "$JOB" "$HOURS" "$REMOTE"
timeout 900 rsync -a "$ST/" "tufts-login:$TMP/"
set +e
OUT=$(timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash $TMP/.submit_remote.sh $TMP $REMOTE")
RC=$?
set -e
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/{print $4}')
if [ -z "$JID" ]; then                    # reconcile through the remote receipt
  JID=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "cat $REMOTE/JOBID 2>/dev/null" || true)
fi
if [ -z "$JID" ]; then
  if [ "$RC" -eq 75 ] || [ "$RC" -eq 76 ] || [ "$RC" -eq 70 ] || [ "$RC" -eq 4 ]; then   # failed before the move: definite
    timeout 60 ssh -o ConnectTimeout=20 tufts-login "rm -rf $TMP" || true
    rm -f "$LANE_DIR/runs/$JOB/INTENT.json"   # definite non-submission
  fi
  echo "not submitted rc=$RC (INTENT kept if the outcome is ambiguous)"; exit $([ "$RC" -eq 0 ] && echo 6 || echo "$RC")
fi
[[ "$JID" =~ ^[0-9]+$ ]] || { echo "bad job id '$JID'; INTENT kept"; exit 7; }
"$PY" -c "import json,sys,datetime; json.dump(dict(job_id=sys.argv[2], remote=sys.argv[3], utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')), open(sys.argv[1],'w'), indent=1)" \
  "$LANE_DIR/runs/$JOB/SUBMIT.json" "$JID" "$REMOTE"
rm -f "$LANE_DIR/runs/$JOB/INTENT.json"
echo "squeue after:"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o "%i %j %T %M %Z"' || true
N=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o '%Z' | grep -cx '$REMOTE'" || true)
echo "jobs in $REMOTE after submit: $N"
LEDGER_LOCKED=1 "$PY" "$LANE_DIR/cluster/ledger.py" update > /dev/null || true
echo "job $JID in $REMOTE"
