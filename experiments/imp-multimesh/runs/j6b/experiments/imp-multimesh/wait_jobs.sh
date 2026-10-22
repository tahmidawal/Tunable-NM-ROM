#!/bin/bash
# wait_jobs.sh <max_seconds> <jobid>... : poll sacct every 60 s (timeout on each ssh), rewrite the heartbeat, return
# when every job is terminal or the deadline passes. Prints the final states.
D=$(( $(date +%s) + $1 )); shift
L=$(cd "$(dirname "$0")" && pwd)
while :; do
  ALL=1; OUT=""
  for j in "$@"; do
    s=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $j -X -n -o State" 2>/dev/null | awk '{print $1}' | head -1)
    OUT="$OUT $j:$s"
    case "$s" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ;; *) ALL=0;; esac
  done
  python3 - "$L/HEARTBEAT.json" "$OUT" <<'PY'
import json, sys, datetime, warnings; warnings.simplefilter("ignore")
p = sys.argv[1]; d = json.load(open(p)); d['utc'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
d['notes'] = ('waiting:' + sys.argv[2])[:300]; json.dump(d, open(p, 'w'))
PY
  [ $ALL -eq 1 ] && { echo "DONE$OUT"; exit 0; }
  [ $(date +%s) -ge $D ] && { echo "DEADLINE$OUT"; exit 1; }
  timeout 60 tail -f /dev/null
done
