#!/bin/bash
# wait.sh <jobid> <max_seconds>: poll sacct until terminal (COMMON-BRIEF 6); prints the state.
J=$1; D=$(( $(date +%s) + $2 ))
while :; do
  S=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $J -X -n -o State" 2>/dev/null | head -1 | tr -d ' ')
  case "$S" in COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*) echo "$S"; exit 0;; esac
  [ "$(date +%s)" -ge "$D" ] && { echo "WAIT:$S"; exit 0; }
  sleep 30
done
