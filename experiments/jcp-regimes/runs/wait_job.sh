#!/bin/bash
# wait on a Slurm job until a terminal sacct state or the deadline (seconds); exits at most after the deadline
J=$1; DL=${2:-1050}; end=$(( $(date +%s) + DL ))
while [ $(date +%s) -lt $end ]; do
  s=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $J -X -n -o State" 2>/dev/null | head -1 | awk '{print $1}')
  case "$s" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) echo "TERMINAL $J $s"; exit 0;; esac
  sleep 60
done
echo "DEADLINE $J state=${s:-unknown}"
