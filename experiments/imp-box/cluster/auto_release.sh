#!/bin/bash
# usage: auto_release.sh <hours> <cell> [<cell> ...]
# Submits each listed cell (already approved in the committed cluster/RELEASE.json, audited) as soon as the lane has a
# free slot (< 3 ibox- jobs), via cluster/submit.sh (which re-checks the lane and account caps under the campaign lock),
# then launches its detached wait_collect.sh. Log: runs/release.log.
set -u
H=$1; shift
LANE=$(cd "$(dirname "$0")/.." && pwd)
LOG="$LANE/runs/release.log"
END=$(( $(date +%s) + H * 3600 ))
for C in "$@"; do
  while [ "$(date +%s)" -lt "$END" ]; do
    n=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h -o %j' | grep -c '^ibox-' || true)
    if [ "${n:-3}" -lt 3 ]; then
      out=$(cd "$LANE" && bash cluster/submit.sh "$C" h200 240G 10:00:00 "cluster/bodies/$C.sh" 2>&1 | tail -3)
      echo "$(date -u +%FT%TZ) submit $C: $out" >> "$LOG"
      if timeout 60 ssh -o ConnectTimeout=20 tufts-login "test -s /cluster/tufts/paralab/tawal01/imp-box/$C/submitted"; then
        (cd "$LANE" && nohup bash cluster/wait_collect.sh "$C" "$C" 11 > /dev/null 2>&1 &)
        echo "$(date -u +%FT%TZ) waiter launched for $C" >> "$LOG"
        break
      fi
    fi
    sleep 300
  done
done
