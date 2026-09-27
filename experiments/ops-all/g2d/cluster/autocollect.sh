#!/bin/bash
# g2d: every 5 min, pull the small records of every finished g2d job not yet pulled (runs/<job>/pull/COLLECTED marker),
# then regenerate results.json / cost_points.json. Prints one line per newly collected job.
cd "$(dirname "$0")/.."
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/g2d
while true; do
  done_jobs=$(ssh -n -o ConnectTimeout=20 tufts-login "cd $NS && grep -l run_exit= */job.out 2>/dev/null | cut -d/ -f1") || { sleep 60; continue; }
  new=0
  for j in $done_jobs; do
    [ -f runs/$j/pull/COLLECTED ] && continue
    [ -d runs/$j ] || continue
    if cluster/collect.sh $j > /dev/null 2>&1; then touch runs/$j/pull/COLLECTED; echo "$(date +%T) collected $j $(grep run_exit runs/$j/pull/job.out)"; new=1; fi
  done
  if [ $new = 1 ]; then /home/tahmid/Dev/.venv/bin/python scripts/summarize_cost.py > runs/summarize_cost.log 2>&1 || echo "summarize_cost failed"; fi
  sleep 300
done
