#!/usr/bin/env bash
# Submit one staged imp-faststep job (COMMON-BRIEF 6/7; code audit 1). Usage: submit.sh <job>
# Under a local lane lock: ledger admission on the staged time limit; upload to a private temporary directory; then,
# under the campaign sbatch lock, atomically: refuse if the job directory exists, refuse at the campaign cap (8; exit 75)
# or the lane cap (2 ifs_ jobs; exit 76), move the upload into place and sbatch. The job id is persisted immediately.
set -euo pipefail
JOB=$1
PY=/home/tahmid/Dev/.venv/bin/python
LANE_DIR=$(cd "$(dirname "$0")/.." && pwd)
ST="$LANE_DIR/runs/$JOB/stage"
NS=/cluster/tufts/paralab/tawal01/imp-faststep
REMOTE=$NS/$JOB
TMP=$NS/.upload-$JOB-$$
[ -f "$ST/run.sbatch" ] || { echo "not staged: $JOB"; exit 2; }
exec 9>"$LANE_DIR/runs/.submit.lock"
flock -w 600 9 || { echo "lane submit lock busy"; exit 77; }
[ -f "$LANE_DIR/runs/$JOB/SUBMIT.json" ] && { echo "already submitted: $JOB"; exit 3; }
HOURS=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['time_limit_hours'])" "$ST/RESERVATION.json")
"$PY" "$LANE_DIR/cluster/ledger.py" check "$HOURS" || { echo "ledger refuses $HOURS h"; exit 78; }
echo "squeue before:"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o "%i %j %T %M %Z"'
timeout 60 ssh -o ConnectTimeout=20 tufts-login "mkdir -p $NS && test ! -e $REMOTE && test ! -e $TMP"
timeout 900 rsync -a "$ST/" "tufts-login:$TMP/"
set +e
OUT=$(timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock \
  bash -c 'test ! -e $REMOTE || exit 4; n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; \
  l=\$(squeue -u tawal01 -h -o %j | grep -c ^ifs_); [ \$l -lt 2 ] || exit 76; mv $TMP $REMOTE && cd $REMOTE && sbatch run.sbatch'")
RC=$?
set -e
echo "$OUT"
JID=$(echo "$OUT" | awk '/Submitted batch job/{print $4}')
if [ -z "$JID" ]; then
  # reconcile an ambiguous outcome: a job may have been accepted even if ssh failed
  JID=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o '%i %Z' | awk '\$2==\"$REMOTE\"{print \$1}'" || true)
fi
if [ -z "$JID" ]; then
  timeout 60 ssh -o ConnectTimeout=20 tufts-login "rm -rf $TMP" || true
  echo "not submitted rc=$RC"; exit $([ $RC -eq 0 ] && echo 6 || echo $RC)
fi
"$PY" - "$LANE_DIR/runs/$JOB/SUBMIT.json" "$JID" "$REMOTE" <<'PYX'
import json, sys, datetime
json.dump(dict(job_id=sys.argv[2], remote=sys.argv[3],
               utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')), open(sys.argv[1], 'w'), indent=1)
PYX
echo "squeue after:"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o "%i %j %T %M %Z"' || true
N=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o '%Z' | grep -cx '$REMOTE'" || true)
"$PY" - "$LANE_DIR/runs/$JOB/SUBMIT.json" "$N" <<'PYX'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d['jobs_in_dir_after'] = sys.argv[2]; json.dump(d, open(p, 'w'), indent=1)
PYX
"$PY" "$LANE_DIR/cluster/ledger.py" update > /dev/null || true
echo "job $JID in $REMOTE (jobs in dir after: $N)"
