#!/bin/bash
# Pull a finished job, verify checksums, audit, summarise; remove the remote dir only with --remove after all pass.
set -euo pipefail
JOB=$1; LANE=$(cd "$(dirname "$0")/.." && pwd); REMOTE=/cluster/tufts/paralab/tawal01/hires_h_20260920/$JOB; RUN=$LANE/runs/$JOB
mkdir -p "$RUN/pull"
rsync -a --exclude tmp --exclude cache --exclude code "tufts-login:$REMOTE/" "$RUN/pull/"
(cd "$RUN/pull" && sha256sum -c OUTPUTS.sha256 > ../CHECKSUM.log && echo checksums_ok)
grep -q "jax_backend=gpu" "$RUN/pull/job.out"
PY=/home/tahmid/Dev/.venv/bin/python
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $PY "$LANE/audit.py" "$RUN/pull/out" > "$RUN/audit.json"
$PY "$LANE/summarize.py" "$RUN"
if [ "${2:-}" = "--remove" ]; then
  $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/audit.json'))['passed'] else 1)"
  ssh tufts-login "rm -rf $REMOTE && test ! -e $REMOTE && echo removed $REMOTE" | tee "$RUN/REMOTE-REMOVED.log"
fi
