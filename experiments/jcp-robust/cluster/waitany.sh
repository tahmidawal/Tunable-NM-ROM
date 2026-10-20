#!/bin/bash
# Wait until ANY of the given Slurm jobs reaches a terminal sacct state or MAXSEC elapses; always exits (brief 7, 15).
#   bash cluster/waitany.sh <maxsec> <jobid> [...]
MAX=$1; shift
end=$(( $(date +%s) + MAX ))
while [ $(date +%s) -lt $end ]; do
  for j in "$@"; do
    st=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $j -X -n -o State" 2>/dev/null | head -1 | awk '{print $1}')
    case "$st" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) echo "$j $st"; exit 0;; esac
  done
  sleep 60
done
echo "deadline"
