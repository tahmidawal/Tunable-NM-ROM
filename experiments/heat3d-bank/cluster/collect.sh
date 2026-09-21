#!/bin/bash
# Pull a finished job, verify checksums, audit + summarise every panel; remove the remote dir only with --remove after all pass.
set -euo pipefail
JOB=$1; LANE=$(cd "$(dirname "$0")/.." && pwd); REMOTE=/cluster/tufts/paralab/tawal01/h3dbank_20260921/$JOB; RUN=$LANE/runs/$JOB
mkdir -p "$RUN/pull"
rsync -a --exclude tmp --exclude cache --exclude code "tufts-login:$REMOTE/" "$RUN/pull/"
(cd "$RUN/pull" && sha256sum -c OUTPUTS.sha256 > ../CHECKSUM.log && echo checksums_ok)
grep -q "jax_backend=gpu" "$RUN/pull/job.out"
PY=/home/tahmid/Dev/.venv/bin/python; ALL=1
for d in "$RUN"/pull/out/*/; do
  s=$(basename "$d"); [ -f "$d/results.json" ] || continue; mkdir -p "$RUN/$s"; ln -sfn "../pull/out/$s" "$RUN/$s/pull_out"
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $PY "$LANE/audit.py" "$d" > "$RUN/$s/audit.json"
  $PY "$LANE/summarize.py" "$RUN/$s" "$d" > /dev/null
  $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/$s/audit.json'))['passed'] else 1)" || ALL=0
done
for t in "$RUN"/pull/out/trained_*; do [ -d "$t" ] && cp "$t/training.json" "$RUN/$(basename "$t")-training.json"; done
echo "all_audits_passed=$ALL"
if [ "${2:-}" = "--remove" ]; then
  [ $ALL -eq 1 ] || { echo "not removing: an audit failed"; exit 1; }
  ssh tufts-login "rm -rf $REMOTE && test ! -e $REMOTE && echo removed $REMOTE" | tee "$RUN/REMOTE-REMOVED.log"
fi
