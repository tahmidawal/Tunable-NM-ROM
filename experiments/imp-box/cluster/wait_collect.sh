#!/bin/bash
# usage: wait_collect.sh <jobname> <collect-name> [deadline_hours_after_start]
# Polls sacct every 5 min (timeout 60 per call). The deadline counts from the job's start (queue wait does not count;
# an overall 96 h ceiling applies). On a terminal state: ledger entry (retried), collection (verified; retried 3 x),
# status line in runs/collect.log; exits non-zero if anything failed. Run detached; record its PID in the heartbeat.
set -u
JOB=$1; NAME=$2; DL=${3:-30}
LANE=$(cd "$(dirname "$0")/.." && pwd)
LOG="$LANE/runs/collect.log"
CEIL=$(( $(date +%s) + 96 * 3600 ))
JID=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "cat /cluster/tufts/paralab/tawal01/imp-box/$JOB/submitted") || { echo "$(date -u +%FT%TZ) $JOB no job id" >> "$LOG"; exit 2; }
START=""
while [ "$(date +%s)" -lt "$CEIL" ]; do
  ST=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $JID -X -n -P -o State | head -1 | cut -d' ' -f1")
  case "$ST" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
      echo "$(date -u +%FT%TZ) $JOB $JID terminal $ST" >> "$LOG"
      CDL=$(( $(date +%s) + 2700 ))   # one overall 45-min deadline for accounting + critical evidence (polling <= 5 min)
      left() { echo $(( CDL - $(date +%s) )); }
      lok=1      # accounting first, independent of any transfer (job audit 4)
      while [ $lok -ne 0 ] && [ "$(left)" -gt 90 ]; do
        timeout 90 /home/tahmid/Dev/.venv/bin/python "$LANE/ledger.py" add "$JID" "$JOB" >> "$LOG" 2>&1 && lok=0 || sleep 30
      done
      [ $lok -eq 0 ] || echo "$(date -u +%FT%TZ) $JOB LEDGER UNRESOLVED (scheduler records kept; job dir kept)" >> "$LOG"
      crit=1     # critical evidence on its own retry loop within the same deadline; deletes nothing
      while [ $crit -ne 0 ] && [ "$(left)" -gt 120 ]; do
        timeout "$(left)" bash "$LANE/cluster/collect.sh" "$JOB" "$NAME" critical >> "$LOG" 2>&1 && crit=0 || sleep 30
      done
      echo "$(date -u +%FT%TZ) $JOB critical rc=$crit" >> "$LOG"
      [ $crit -eq 0 ] || { echo "$(date -u +%FT%TZ) $JOB CRITICAL EVIDENCE UNRESOLVED: job dir kept" >> "$LOG"; exit 1; }
      cok=1      # bulk phase (may exceed an hour); deletion only inside collect.sh after every gate
      for i in 1 2 3; do bash "$LANE/cluster/collect.sh" "$JOB" "$NAME" >> "$LOG" 2>&1 && { cok=0; break; }; sleep 300; done
      echo "$(date -u +%FT%TZ) $JOB collect rc=$cok" >> "$LOG"
      [ $cok -eq 0 ] && exit 0
      echo "$(date -u +%FT%TZ) $JOB COLLECTION UNRESOLVED: job dir kept; renew monitoring" >> "$LOG"
      exit 1;;
    RUNNING) [ -z "$START" ] && START=$(date +%s);;
  esac
  if [ -n "$START" ] && [ "$(date +%s)" -gt $(( START + DL * 3600 )) ]; then
    echo "$(date -u +%FT%TZ) $JOB still $ST after ${DL} h running: escalate" >> "$LOG"; exit 3
  fi
  sleep 300
done
echo "$(date -u +%FT%TZ) $JOB 96 h ceiling passed ($ST)" >> "$LOG"; exit 3
