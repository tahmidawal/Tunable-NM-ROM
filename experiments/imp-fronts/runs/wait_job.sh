#!/bin/bash
# wait_job.sh <jobid> <deadline_s> <step>: poll sacct every 120 s, rewrite heartbeat, exit on terminal state or deadline
cd "$(dirname "$0")/.."
end=$(( $(date +%s) + $2 ))
while [ "$(date +%s)" -lt "$end" ]; do
  s=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $1 -X -n -o State" 2>/dev/null | tr -d ' ')
  case "$s" in COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF*|NODE_FAIL*) echo "$1 $s"; exit 0;; esac
  HB_SLURM=$1 ./hb.sh "$3" "background waiter pid $$" "" "" "$$" >/dev/null 2>&1
  sleep 120
done
echo "$1 DEADLINE"; exit 3
