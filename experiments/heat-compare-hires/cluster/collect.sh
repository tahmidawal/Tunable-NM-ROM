#!/bin/bash
# Pull one finished job, verify checksums and the GPU preflight; for panel jobs run the independent audit
# and the generated summary. Remote dir is removed only with --remove AND (panel) audit passed.
#   cluster/collect.sh <job> [--remove]
set -euo pipefail
JOB=$1; LANE=$(cd "$(dirname "$0")/.." && pwd); REMOTE=/cluster/tufts/paralab/tawal01/hcmp_20260923/$JOB; RUN=$LANE/runs/$JOB
mkdir -p "$RUN/pull"
rsync -a --exclude tmp --exclude cache --exclude code "tufts-login:$REMOTE/" "$RUN/pull/"
(cd "$RUN/pull" && sha256sum -c OUTPUTS.sha256 > ../CHECKSUM.log && echo checksums_ok)
grep -q "jax_backend=gpu" "$RUN/pull/job.out"; grep -h "run_exit=" "$RUN/pull/job.out" || true
PY=/home/tahmid/Dev/.venv/bin/python
if [ -f "$RUN/pull/out/results.json" ]; then
  OPENBLAS_NUM_THREADS=8 $PY "$LANE/audit_panel.py" "$RUN/pull/out" > "$RUN/audit.json"
  $PY "$LANE/summarize.py" "$RUN/pull/out/results.json" "$RUN/audit.json" "$RUN" "$LANE/../hires-heat/runs/h2d-final04/summary.json" > "$RUN/summarize.log"
  $PY -c "import json; a=json.load(open('$RUN/audit.json')); print('audit passed', a['passed'], 'failures', a['failure_count'])"
fi
if [ "${2:-}" = "--remove" ]; then
  if [ -f "$RUN/audit.json" ]; then $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/audit.json'))['passed'] else 1)"; fi
  ssh tufts-login "rm -rf $REMOTE && test ! -e $REMOTE && echo removed $REMOTE" | tee "$RUN/REMOTE-REMOVED.log"
fi
