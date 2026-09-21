#!/bin/bash
# Pull a finished job, verify checksums/exit/panels, audit + summarise every panel; remove the remote dir only with --remove
# after every check passes (Codex findings 2, 6, 7).
set -euo pipefail
JOB=$1; [[ "$JOB" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{1,40}$ ]] || { echo "bad job name"; exit 1; }
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/h3dbank_20260921; REMOTE="$NS/$JOB"; RUN="$LANE/runs/$JOB"
ID=$(grep -oE 'Submitted batch job [0-9]+' "$RUN/SUBMIT.log" | awk '{print $4}')
if ssh tufts-login "squeue -h -j $ID" 2>/dev/null | grep -q .; then echo "job $ID still queued/running"; exit 1; fi
mkdir -p "$RUN/pull"
rsync -a --exclude tmp --exclude cache --exclude code "tufts-login:$REMOTE/" "$RUN/pull/"
(cd "$RUN/pull" && sha256sum -c OUTPUTS.sha256 > ../CHECKSUM.log && echo checksums_ok)
grep -q "jax_backend=gpu" "$RUN/pull/job.out"
ALL=1; grep -q "^run_exit=0$" "$RUN/pull/out/EXIT_STATUS.txt" || { echo "run_exit != 0"; ALL=0; }
PY=/home/tahmid/Dev/.venv/bin/python; CODE="$RUN/stage/code/heat3d-bank"   # the staged (checksummed) audit/summarize of this job
while read -r c; do
  s=${c%.json}; d="$RUN/pull/out/$s"
  if [ ! -f "$d/results.json" ]; then echo "missing panel $s"; ALL=0; continue; fi
  mkdir -p "$RUN/$s"
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $PY "$CODE/audit.py" "$d" > "$RUN/$s/audit.json"
  $PY "$CODE/summarize.py" "$RUN/$s" "$d" > /dev/null
  $PY -c "import json,sys; a=json.load(open('$RUN/$s/audit.json')); r=json.load(open('$d/results.json')); sys.exit(0 if a['passed'] and r['complete'] else 1)" || { echo "audit/complete failed: $s"; ALL=0; }
done < "$RUN/stage/EXPECTED_PANELS.txt"
for t in "$RUN"/pull/out/trained_*; do [ -d "$t" ] && cp "$t/training.json" "$RUN/$(basename "$t")-training.json"; done
echo "all_checks_passed=$ALL" | tee "$RUN/COLLECT-STATUS.txt"
if [ "${2:-}" = "--remove" ]; then
  [ $ALL -eq 1 ] || { echo "not removing: a check failed"; exit 1; }
  ssh tufts-login "case '$REMOTE' in $NS/?*) rm -rf -- '$REMOTE' && test ! -e '$REMOTE' && echo removed '$REMOTE';; *) echo refused; exit 1;; esac" | tee "$RUN/REMOTE-REMOVED.log"
fi
