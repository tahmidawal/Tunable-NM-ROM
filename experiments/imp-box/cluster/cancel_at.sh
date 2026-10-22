#!/bin/bash
# usage: cancel_at.sh <jobid> <expected_job_name> <max_elapsed_seconds>
# scancel the job once its elapsed time reaches the bound (polled every 120 s; query failures retried); verifies the
# job id is numeric and owned by tawal01 with the expected name, and verifies the cancellation before logging it.
set -u
J=$1; NAME=$2; MAX=$3
[[ "$J" =~ ^[0-9]+$ ]] || exit 2
LOG="$(dirname "$0")/../runs/release.log"
while true; do
  r=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $J -X -n -P -o User,JobName,State,ElapsedRaw" | head -1)
  if [ -z "$r" ]; then sleep 60; continue; fi
  IFS='|' read -r u n st el <<< "$r"
  [ "$u" = tawal01 ] && [ "$n" = "$NAME" ] || { echo "$(date -u +%FT%TZ) cancel_at: $J is $u/$n, not tawal01/$NAME" >> "$LOG"; exit 2; }
  case "${st%% *}" in RUNNING|PENDING) ;; *) exit 0;; esac
  if [ "${el:-0}" -ge "$MAX" ]; then
    timeout 60 ssh -o ConnectTimeout=20 tufts-login "scancel $J"; sleep 30
    s2=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $J -X -n -P -o State" | head -1)
    case "${s2%% *}" in CANCELLED*|COMPLETED|FAILED|TIMEOUT) echo "$(date -u +%FT%TZ) cancelled $J ($s2) at $el s" >> "$LOG"; exit 0;; esac
    continue
  fi
  sleep 120
done
