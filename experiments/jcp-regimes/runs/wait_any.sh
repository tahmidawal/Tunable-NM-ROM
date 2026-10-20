#!/bin/bash
# exit when any listed Slurm job reaches a terminal sacct state, or at the deadline (seconds)
DL=$1; shift; end=$(( $(date +%s) + DL ))
while [ $(date +%s) -lt $end ]; do
  for J in "$@"; do
    s=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $J -X -n -o State" 2>/dev/null | head -1 | awk '{print $1}')
    case "$s" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) echo "TERMINAL $J $s"; exit 0;; esac
  done
  sleep 60
done
echo "DEADLINE"
