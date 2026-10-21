#!/usr/bin/env bash
# Submit one staged imp-faststep job through the campaign sbatch lock (COMMON-BRIEF 6). Usage: submit.sh <job>
# Exit 75: account at the campaign cap (retry later). Records runs/<job>/SUBMIT.json.
set -euo pipefail
JOB=$1
LANE_DIR=$(cd "$(dirname "$0")/.." && pwd)
ST="$LANE_DIR/runs/$JOB/stage"
REMOTE=/cluster/tufts/paralab/tawal01/imp-faststep/$JOB
[ -f "$ST/run.sbatch" ] || { echo "not staged: $JOB"; exit 2; }
[ -f "$LANE_DIR/runs/$JOB/SUBMIT.json" ] && { echo "already submitted: $JOB"; exit 3; }
echo "squeue before:"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o "%i %j %T %M %Z"'
timeout 60 ssh -o ConnectTimeout=20 tufts-login "test ! -e $REMOTE" || { echo "remote dir exists: $REMOTE"; exit 4; }
timeout 600 rsync -a "$ST/" "tufts-login:$REMOTE/"
set +e
OUT=$(timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock \
  bash -c 'n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; cd $REMOTE && sbatch run.sbatch'")
RC=$?
set -e
echo "$OUT"
[ $RC -eq 0 ] || { echo "sbatch failed rc=$RC"; exit $RC; }
JID=$(echo "$OUT" | awk '/Submitted batch job/{print $4}')
echo "squeue after:"; timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o "%i %j %T %M %Z"'
N=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o '%Z' | grep -c '^$REMOTE\$'" || true)
python3 - "$LANE_DIR/runs/$JOB/SUBMIT.json" "$JID" "$REMOTE" "$N" <<'PY'
import json, sys, datetime
json.dump(dict(job_id=sys.argv[2], remote=sys.argv[3], jobs_in_dir=int(sys.argv[4] or 0),
               utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')), open(sys.argv[1], 'w'), indent=1)
PY
echo "job $JID in $REMOTE (jobs in dir: $N)"
