#!/bin/bash
# g2d queue runner: every 2 min, while fewer than 3 g2d jobs are running/pending, submit the first queue item whose
# dependencies have finished (their job.out has run_exit=). Submitted items are recorded in cluster/submitted.txt.
cd "$(dirname "$0")/.."
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/g2d
touch cluster/submitted.txt
while true; do
  left=0
  while read -r job gpu wall mem tree body deps; do
    grep -qx "$job" cluster/submitted.txt && continue
    left=$((left+1))
    N=$(ssh -n -o ConnectTimeout=20 tufts-login "squeue -u \$USER -h -o '%j %T'" | awk '$1 ~ /^g2d_/' | grep -c . || true)
    [ "$N" -lt 10 ] || break
    if [ "$deps" != "-" ]; then
      ok=1; for d in ${deps//,/ }; do ssh -n -o ConnectTimeout=20 tufts-login "grep -q run_exit= $NS/$d/job.out 2>/dev/null" || ok=0; done
      [ $ok = 1 ] || continue
    fi
    echo "$(date +%T) submitting $job"
    if cluster/submit.sh $job $gpu $wall $mem $tree $body < /dev/null > cluster/submit-$job.log 2>&1; then echo "$job" >> cluster/submitted.txt; echo "$(date +%T) submitted $job: $(grep Submitted cluster/submit-$job.log)"; else echo "$(date +%T) submit $job failed rc=$?"; tail -3 cluster/submit-$job.log; if ! grep -q Submitted cluster/submit-$job.log && ! grep -q "already queued" cluster/submit-$job.log; then rm -rf runs/$job; ssh -n tufts-login "rmdir $NS/$job 2>/dev/null; true"; fi; fi
    sleep 5
  done < cluster/queue.txt
  [ "$left" -eq 0 ] && { echo "queue empty"; exit 0; }
  sleep 120
done
