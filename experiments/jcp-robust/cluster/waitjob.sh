#!/bin/bash
# Wait on Slurm jobs until all reach a terminal sacct state or MAXSEC elapses (common brief 7, 15); always exits.
#   bash cluster/waitjob.sh <maxsec> <jobid> [<jobid> ...]
MAX=$1; shift
end=$(( $(date +%s) + MAX ))
while [ $(date +%s) -lt $end ]; do
  all=1
  for j in "$@"; do
    st=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $j -X -n -o State" 2>/dev/null | head -1 | awk '{print $1}')
    case "$st" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) ;; *) all=0;; esac
  done
  [ $all = 1 ] && break
  sleep 60
done
for j in "$@"; do echo "$j $(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $j -X -n -o State,Elapsed,NodeList" 2>/dev/null | head -1)"; done
