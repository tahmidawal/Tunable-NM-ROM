#!/bin/bash
# Wait until none of the given job ids is in squeue; heartbeat every 10 min; deadline in hours (arg 1).
LANE=$(cd "$(dirname "$0")/.." && pwd)
DL=$(( $(date +%s) + $1 * 3600 )); shift
IDS=$(IFS=,; echo "$*")
while true; do
  Q=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -h -j $IDS -o '%i %T %M'" 2>/dev/null)
  RC=$?
  if [ $RC -eq 0 ] && [ -z "$Q" ]; then echo "all done $(date -u +%FT%TZ)"; exit 0; fi
  [ "$(date +%s)" -lt "$DL" ] || { echo "deadline"; exit 3; }
  (cd "$LANE" && /home/tahmid/Dev/.venv/bin/python hb.py "waiting for C2 jobs ($(echo $Q | tr '\n' ' '))" slurm_jobs=$IDS local_pid=$$ expected=$(date -u -d @$DL +%FT%TZ))
  sleep 600
done
