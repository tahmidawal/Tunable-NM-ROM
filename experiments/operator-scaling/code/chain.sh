#!/bin/bash
# Run training runs sequentially in priority order. Each argument is one run: "ARM RUNG CAP [SNAPSHOT [SCHEDULE]]".
# Every run gets the protocol deadline. On the job's USR1 (wall limit near) the running trainer stops cleanly and no
# further run starts.
PY=${PY:-/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python}
[ -x "$PY" ] || { echo "no python at $PY"; exit 3; }
DEADLINE=1794754800
STOPPED=0
trap 'STOPPED=1; kill -USR1 $CHILD 2>/dev/null' USR1
for spec in "$@"; do
  set -- $spec; ARM=$1; RUNG=$2; CAP=$3; SNAP=${4:-0}; SCHED=${5:-plateau}
  [ "$STOPPED" = 1 ] && { echo "SKIP (job stopping) $spec"; continue; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "SKIP (deadline passed) $spec"; continue; }
  DEV=cuda; [ "$RUNG" -gt 4608 ] && DEV=cpu
  OUT=runs/$ARM-n$RUNG-s0; [ "$SCHED" != plateau ] && OUT=$OUT-$SCHED
  EXTRA=""; [ "$RUNG" -gt 4608 ] && EXTRA="--extra ../data/extra"
  if [ -e "$OUT/last.pt" ] && [ ! -e "$OUT/result.json" ]; then CONT=--continue-last; else CONT=""; fi
  [ -e "$OUT/result.json" ] && { echo "SKIP (done) $spec"; continue; }
  echo "START $spec $(date)"
  "$PY" code/train.py --pool ../data/pool $EXTRA --rung $RUNG --arm $ARM --seed 0 --out $OUT --max-steps $CAP \
      --snapshot-step $SNAP --deadline $DEADLINE --data-device $DEV --schedule $SCHED $CONT &
  CHILD=$!
  wait $CHILD; rc=$?
  while kill -0 $CHILD 2>/dev/null; do wait $CHILD; rc=$?; done
  echo "END $spec rc=$rc $(date)"
done
