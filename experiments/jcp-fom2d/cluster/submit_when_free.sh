#!/bin/bash
# Submit staged attempts in order as soon as the lane has a free slot (lane cap 3; submit.sh does every check).
# Polls every 120 s; overall deadline in hours (default 4). Usage: bash cluster/submit_when_free.sh <deadline_h> <attempt> [...]
set -u
LANE=$(cd "$(dirname "$0")/.." && pwd)
DL=$(( $(date +%s) + ${1:-4} * 3600 )); shift
for A in "$@"; do
  while true; do
    [ "$(date +%s)" -lt "$DL" ] || { echo "deadline reached before submitting $A"; exit 3; }
    N=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o %j' | grep -c '^f2d_' || true)
    if [ "$N" -lt 3 ]; then
      if bash "$LANE/cluster/submit.sh" "$A"; then echo "SUBMITTED $A $(date -u +%FT%TZ)"; break; fi
      echo "submit of $A refused at $(date -u +%FT%TZ); retrying"
    fi
    sleep 120
  done
done
echo ALL-SUBMITTED
