#!/bin/bash
# Wait until every listed Slurm job reaches a terminal sacct state (COMMON-BRIEF 15), with an overall deadline.
# usage: bash waitjobs.sh <deadline_seconds> <jobid> [<jobid> ...]; prints "<jobid> <state>" lines when done.
set -u
dl=$(( $(date +%s) + $1 )); shift
while :; do
  done_all=1
  out=""
  for j in "$@"; do
    s=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $j -X -n -o State" 2>/dev/null | head -1 | awk '{print $1}')
    case "$s" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) out="$out$j $s\n";; *) done_all=0;; esac
  done
  if [ $done_all = 1 ]; then printf "$out"; exit 0; fi
  [ "$(date +%s)" -ge "$dl" ] && { echo "DEADLINE"; printf "$out"; exit 3; }
  sleep 60
done
