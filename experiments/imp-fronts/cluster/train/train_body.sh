#!/bin/bash
# imp-fronts Phase B training job body, sourced by the per-job sbatch headers (smA, smB, trfid, trcov) from the staged
# job directory /cluster/tufts/paralab/tawal01/imp-fronts/<JOB>/ (laid out by stage_job.sh: stage_arm.sh tree +
# this file + phase_times.py + MANIFEST.sha256 + COMMIT.txt). The header sets: JOB ARM MODE(smoke|production) WALL_S.
#
# Order: manifest check -> GPU preflight (exit 42 unless jax_backend=gpu) -> [smoke only] C-B1 guard rejection on a
# corrupted copy (must exit 1 with MISMATCH) -> run_seed.sh 1 (seed 1; TRAIN2D_SMOKE=1 in smoke mode) under the
# self-timer (Amendment 4 B8: new work stops at SELF_TIMER_FRAC = 0.9 of the wall limit; on expiry EXHAUSTED.json
# records the pre-registered outcome) -> phase_times.py -> OUTPUTS.sha256. Never writes outside $TASK_ROOT.
set -euo pipefail
: "${JOB:?}" "${ARM:?}" "${MODE:?}" "${WALL_S:?}"
NS=/cluster/tufts/paralab/tawal01/imp-fronts
TASK_ROOT="$NS/$JOB"
[ "$(pwd -P)" = "$(cd "$TASK_ROOT" && pwd -P)" ] || { echo "must run from $TASK_ROOT"; exit 2; }
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output" "$TASK_ROOT/logs"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
SELF_TIMER_FRAC="${SELF_TIMER_FRAC:-0.9}"
BUDGET=$("$PY" -c "print(int($WALL_S * $SELF_TIMER_FRAC))")
T_START=$(date +%s)
echo "job=$JOB arm=$ARM mode=$MODE host=$(hostname) commit=$SOURCE_COMMIT slurm=${SLURM_JOB_ID:-none} wall_s=$WALL_S self_timer_s=$BUDGET"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
python_arm=$("$PY" -c "import json;print(json.load(open('ARM-STAGED.json'))['arm'])")
[ "$python_arm" = "$ARM" ] || { echo "staged arm $python_arm != $ARM"; exit 2; }
T2="$TASK_ROOT/experiments/imp-fronts/train2d"
OUT="$TASK_ROOT/output/train2d_s1"

if [ "$MODE" = smoke ]; then
  # C-B1 analogue (B9 SM-a): a deliberately mismatched sha256 in PROVENANCE.json must be rejected by run_seed.sh
  BAD="$TMPDIR/cb1_tree"; rm -rf "$BAD"; cp -a "$T2" "$BAD"
  "$PY" -I - "$BAD/PROVENANCE.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
e = next(f for f in p['files'] if f['path'] == 'stageA/deps/burgers2d_film.py')
e['sha256'] = e['sha256'][:-1] + ('0' if e['sha256'][-1] != '0' else '1')
json.dump(p, open(sys.argv[1], 'w'), indent=1)
EOF
  set +e
  STAGES=none bash "$BAD/run_seed.sh" 1 "$TMPDIR/cb1_out" > "$TASK_ROOT/logs/cb1.log" 2>&1
  RC=$?
  set -e
  cat "$TASK_ROOT/logs/cb1.log"
  if [ "$RC" = 1 ] && grep -q 'PROVENANCE sha256 MISMATCH' "$TASK_ROOT/logs/cb1.log"; then
    echo "C-B1 guard rejection: FIRED (rc=$RC)"; echo "{\"C_B1_guard_rejects\": true, \"rc\": $RC}" > "$TASK_ROOT/output/cb1.json"
  else
    echo "C-B1 guard rejection: DID NOT FIRE (rc=$RC)"; echo "{\"C_B1_guard_rejects\": false, \"rc\": $RC}" > "$TASK_ROOT/output/cb1.json"; exit 9
  fi
  rm -rf "$BAD" "$TMPDIR/cb1_out"
  SMOKE_ENV=(TRAIN2D_SMOKE=1)
else
  SMOKE_ENV=()
fi

LEFT=$(( BUDGET - ($(date +%s) - T_START) ))
echo "run_seed.sh seed 1 ($MODE), self-timer ${LEFT}s"
set +e
env "${SMOKE_ENV[@]}" timeout --signal=TERM --kill-after=120 "$LEFT" bash "$T2/run_seed.sh" 1 "$OUT"
RC=$?
set -e
if [ "$RC" = 124 ] || [ "$RC" = 137 ]; then
  OUTCOME=$([ "$MODE" = production ] && echo "arm NOT RUN (DESIGN Amendment 3 B8: training ceiling exhausted)" \
            || echo "smoke incomplete: Phase B not submitted (Amendment 3 B8)")
  printf '{"job": "%s", "arm": "%s", "mode": "%s", "rc": %s, "self_timer_s": %s, "outcome": "%s"}\n' \
    "$JOB" "$ARM" "$MODE" "$RC" "$LEFT" "$OUTCOME" > "$TASK_ROOT/EXHAUSTED.json"
  cat "$TASK_ROOT/EXHAUSTED.json"
fi
"$PY" "$TASK_ROOT/phase_times.py" "$OUT" "$JOB" "$ARM" "$MODE" "$WALL_S" "$BUDGET" "$TASK_ROOT/output/phase_times_$JOB.json" || true
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
[ "$RC" = 0 ] || { echo "run_seed.sh exited $RC"; exit "$RC"; }
echo ALL-DONE
