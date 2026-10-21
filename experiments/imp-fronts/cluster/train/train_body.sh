#!/bin/bash
# imp-fronts Phase B training job body, sourced by the per-job sbatch headers (smA, smB, trfid, trcov) from the staged
# job directory /cluster/tufts/paralab/tawal01/imp-fronts/<JOB>/ (laid out by stage_job.sh: stage_arm.sh tree +
# this file + phase_times.py + smoke_datagen.py + MANIFEST.sha256 + COMMIT.txt). Header sets JOB ARM MODE WALL_S
# SELF_TIMER_FRAC.
#
# production (DESIGN Amendment 11): run_seed.sh 1 (full recipe) with NO self-timer (SELF_TIMER_FRAC=1; only the hard
#   wall; 120 s before it, `timeout` stops run_seed.sh so EXHAUSTED.json ("arm NOT RUN") can be written).
# smoke (Amendment 11): SELF_TIMER_FRAC=0.9 (10 % finalisation margin). (1) C-B1 guard rejection on a corrupted copy
#   (must exit 1 with MISMATCH); (2) smoke_datagen.py: full 576 + 4,032 generation through the staged generator,
#   timed per population (cov: Newton residual stats + C-B4; fid: C-B3 + PS self-convergence); (3) chaining slice:
#   run_seed.sh's own TRAIN2D_DRYRUN mode (N=64, 16 trajectories, 20 steps; Stage A -> B -> rotation on the GPU),
#   run only with the time left (run_seed.sh's TRAIN2D_SMOKE mode does full-size extraction, ~900 s, and cannot fit
#   the 0.2 / 0.25 GPU-h ceilings). Training-phase rates come from K1's records (phase_times.py), not from the slice.
# Then phase_times.py -> output/phase_times_<JOB>.json and OUTPUTS.sha256. Never writes outside $TASK_ROOT.
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
: "${SELF_TIMER_FRAC:?}"
BUDGET=$("$PY" -c "f=$SELF_TIMER_FRAC; w=$WALL_S; print(int(w - 120) if f >= 1 else int(w * f))")
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
  LEFT=$(( BUDGET - ($(date +%s) - T_START) ))
  echo "smoke_datagen ($ARM), budget ${LEFT}s"
  set +e
  "$PY" -u "$TASK_ROOT/smoke_datagen.py" "$T2" "$ARM" "$TASK_ROOT/output/datagen_$JOB.json" --budget-s "$LEFT" \
    2>&1 | tee "$TASK_ROOT/logs/datagen.log"
  RC=${PIPESTATUS[0]}
  set -e
  LEFT=$(( BUDGET - ($(date +%s) - T_START) ))
  if [ "$RC" = 0 ] && [ "$LEFT" -gt 60 ]; then
    echo "chaining slice (run_seed.sh TRAIN2D_DRYRUN=1), self-timer ${LEFT}s"
    set +e
    TRAIN2D_DRYRUN=1 timeout --signal=TERM --kill-after=30 "$LEFT" bash "$T2/run_seed.sh" 1 "$OUT" \
      > "$TASK_ROOT/logs/chain.log" 2>&1
    CRC=$?
    set -e
    tail -5 "$TASK_ROOT/logs/chain.log"
    echo "{\"chain_rc\": $CRC, \"seed_complete\": $([ -e "$OUT/SEED-COMPLETE" ] && echo true || echo false)}" \
      > "$TASK_ROOT/output/chain.json"
  else
    echo "{\"chain_rc\": null, \"skipped\": true, \"left_s\": $LEFT}" > "$TASK_ROOT/output/chain.json"
  fi
  "$PY" "$TASK_ROOT/phase_times.py" "$TASK_ROOT/output/datagen_$JOB.json" "$JOB" "$ARM" "$MODE" "$WALL_S" "$BUDGET" \
    "$TASK_ROOT/output/phase_times_$JOB.json" || true
  cd "$TASK_ROOT"
  find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
  [ "$RC" = 0 ] || { echo "smoke_datagen exited $RC"; exit "$RC"; }
  echo ALL-DONE
  exit 0
fi

LEFT=$(( BUDGET - ($(date +%s) - T_START) ))
echo "run_seed.sh seed 1 (production), stop ${LEFT}s (hard wall minus 120 s recording reserve; no self-timer)"
set +e
timeout --signal=TERM --kill-after=60 "$LEFT" bash "$T2/run_seed.sh" 1 "$OUT"
RC=$?
set -e
if [ "$RC" = 124 ] || [ "$RC" = 137 ]; then
  printf '{"job": "%s", "arm": "%s", "rc": %s, "stop_s": %s, "outcome": "arm NOT RUN (DESIGN Amendment 11: 4.6 GPU-h training ceiling exhausted)"}\n' \
    "$JOB" "$ARM" "$RC" "$LEFT" > "$TASK_ROOT/EXHAUSTED.json"
  cat "$TASK_ROOT/EXHAUSTED.json"
fi
"$PY" "$TASK_ROOT/phase_times.py" --production "$OUT" "$JOB" "$ARM" "$MODE" "$WALL_S" "$BUDGET" \
  "$TASK_ROOT/output/phase_times_$JOB.json" || true
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
[ "$RC" = 0 ] || { echo "run_seed.sh exited $RC"; exit "$RC"; }
echo ALL-DONE
