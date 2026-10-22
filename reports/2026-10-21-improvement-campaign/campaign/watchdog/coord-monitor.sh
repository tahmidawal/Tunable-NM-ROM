#!/usr/bin/env bash
# Coordinator Monitor source: new ACTION-NEEDED lines, watchdog death, and terminal states of
# parked-lane jobs listed in parked-jobs.txt ("<jobid> <lane>" per line; line removed by the coordinator).
D=/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/campaign/watchdog
n0=$(wc -l < $D/ACTION-NEEDED.log); k=0
while true; do
  n=$(wc -l < $D/ACTION-NEEDED.log)
  if [ "$n" -gt "$n0" ]; then sed -n "$((n0+1)),${n}p" $D/ACTION-NEEDED.log; n0=$n; fi
  c=$(ps -eo args | grep -c '[w]d.sh [12] '); [ "$c" -lt 2 ] && echo "WATCHDOG PROCESS COUNT $c (<2)"
  if [ $((k % 5)) -eq 0 ] && [ -s $D/parked-jobs.txt ]; then
    while read -r j lane; do
      s=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $j -X -n -o State" 2>/dev/null | head -1 | tr -d ' ')
      case "$s" in COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*) echo "PARKED JOB $j ($lane) ended: $s";; esac
    done < $D/parked-jobs.txt
  fi
  k=$((k+1)); sleep 60
done
