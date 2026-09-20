#!/bin/bash
# Guarded submit: waits until the account has < 6 running/pending jobs and this lane < 2, then rsync + sbatch once.
# usage: cluster/submit.sh <attempt>      (the attempt must already be staged by cluster/stage.py)
set -euo pipefail
A=$1; HERE=$(cd "$(dirname "$0")" && pwd); NS=/cluster/tufts/paralab/tawal01/nmrombase_20260920
test -f "$HERE/stage/$A/run.sbatch"
while true; do
  Q=$(ssh tufts-login 'squeue -u tawal01 -h -o "%j %Z"') || { sleep 60; continue; }
  TOTAL=$(printf '%s\n' "$Q" | grep -c . || true); MINE=$(printf '%s\n' "$Q" | grep -c nmrombase_20260920 || true)
  if printf '%s\n' "$Q" | grep -q "$NS/$A\$"; then echo "already queued for $A"; exit 1; fi
  if [ "$TOTAL" -lt 6 ] && [ "$MINE" -lt 2 ]; then break; fi
  echo "waiting: account=$TOTAL lane=$MINE $(date +%H:%M)"; sleep 120
done
ssh tufts-login "test ! -e $NS/$A"
rsync -a "$HERE/stage/$A/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch && sleep 3 && squeue -u tawal01 -h -o '%i %j %T %Z'"
