#!/bin/bash
# Pull one job directory, verify output checksums, run the summariser; with --remove delete the remote dir after all checks.
#   cluster/collect.sh <job> [--remove]
set -euo pipefail
JOB=$1; LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/hbank_20260923; REMOTE=$NS/$JOB; RUN=$LANE/runs/$JOB
[ -d "$RUN" ] || { echo "no run dir $RUN"; exit 1; }
mkdir -p "$RUN/pull"
rsync -a --exclude cache --exclude tmp --exclude code "tufts-login:$REMOTE/" "$RUN/pull/" 2>&1 | tee "$RUN/collect.log"
(cd "$RUN/pull" && sha256sum -c OUTPUTS.sha256) > "$RUN/CHECKSUM.log" 2>&1 && CK=ok || CK=FAILED
EXIT=$(cat "$RUN/pull/out/EXIT_STATUS.txt" 2>/dev/null || echo missing)
AUD=$(/home/tahmid/Dev/.venv/bin/python -c "import json;print(json.load(open('$RUN/pull/out/audit_np.json'))['passed'])" 2>/dev/null || echo missing)
grep -q "jax_backend=gpu" "$RUN/pull/job.out" && GPU=ok || GPU=MISSING
grep -qi "captured\|out of memory\|No space left" "$RUN/pull/job.out" "$RUN/pull/job.err" && WARN=FOUND || WARN=none
echo "checksums=$CK exit=$EXIT audit=$AUD jax_backend_gpu=$GPU warnings=$WARN" | tee "$RUN/COLLECT-STATUS.txt"
/home/tahmid/Dev/.venv/bin/python "$LANE/hbk_summarize.py" "$RUN" > /dev/null && echo "summary written"
if [ "${2:-}" = "--remove" ]; then
  [ "$CK" = ok ] && [ "$EXIT" = "run_exit=0" ] && [ "$AUD" = True ] && [ "$GPU" = ok ] || { echo "checks not all passed; remote kept"; exit 1; }
  ssh tufts-login "rm -rf '$REMOTE'" && echo "removed $REMOTE $(date -u +%FT%TZ)" | tee "$RUN/REMOTE-REMOVED.log"
fi
