#!/bin/bash
# imp-fronts Phase B training job body, sourced by the per-job sbatch headers (smA, smB, trfid, trcov) from the staged
# job directory /cluster/tufts/paralab/tawal01/imp-fronts/<JOB>/ (laid out by stage_job.sh: stage_arm.sh tree +
# this file + phase_times.py + smoke_datagen.py + MANIFEST.sha256 + COMMIT.txt). Header sets JOB ARM MODE WALL_S
# SELF_TIMER_FRAC.
#
# smokec (SM-c, arm (c), Amendments 19-21): C-B1, tests/test_arm_c.py on the GPU, smoke_bnd_timing.py, TRAIN2D_DRYRUN
#   chaining slice of the bnd tree, phase_times.py smokec -> output/timing_c.json; ALL-DONE only if all passed.
# production (DESIGN Amendment 11): run_seed.sh 1 (full recipe) with NO self-timer (SELF_TIMER_FRAC=1; only the hard
#   wall; 120 s before it, `timeout` stops run_seed.sh so EXHAUSTED.json ("arm NOT RUN") can be written).
# smoke (Amendment 11): SELF_TIMER_FRAC=0.9 (10 % finalisation margin). (1) C-B1 guard rejection on a corrupted copy
#   (must exit 1 with MISMATCH); (2) smoke_datagen.py: gates (fid C-B3 + self-convergence; cov C-B4), GPU burn-in,
#   incumbent vs arm generator timed on the same 576 seed-0 + 576 seed-1000 trajectories (Amendment 15); (3) chaining slice:
#   run_seed.sh's own TRAIN2D_DRYRUN mode (N=64, 16 trajectories, 20 steps; Stage A -> B -> rotation on the GPU),
#   run only with the time left (run_seed.sh's TRAIN2D_SMOKE mode does full-size extraction, ~900 s, and cannot fit
#   the smoke ceilings). Training-phase rates come from K1's records (phase_times.py), not from the slice.
#   (4) phase_times.py smoke -> output/timing.json (resources.py schema) only if every step passed. ALL-DONE only if
#   generation, gates, chaining and timing all succeeded; otherwise exit 1. Never writes outside $TASK_ROOT.
set -euo pipefail
: "${JOB:?}" "${ARM:?}" "${MODE:?}" "${WALL_S:?}"
NS=/cluster/tufts/paralab/tawal01/imp-fronts
TASK_ROOT="$NS/$JOB"
[ "$(pwd -P)" = "$(cd "$TASK_ROOT" && pwd -P)" ] || { echo "must run from $TASK_ROOT"; exit 2; }
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TASK_ROOT/output" "$TASK_ROOT/logs"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
: "${SELF_TIMER_FRAC:?}"
BUDGET=$("$PY" -c "f=$SELF_TIMER_FRAC; w=$WALL_S; print(int(w - 120) if f >= 1 else int(w * f))")
T_START=$(date +%s)
# remaining self-timer seconds; every timed invocation runs only if this is positive (audit codex-job-smC-1:
# `timeout 0` would disable the limit)
left() { echo $(( BUDGET - ($(date +%s) - T_START) )); }
echo "job=$JOB arm=$ARM mode=$MODE host=$(hostname) commit=$SOURCE_COMMIT slurm=${SLURM_JOB_ID:-none} wall_s=$WALL_S self_timer_s=$BUDGET"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
python_arm=$("$PY" -c "import json;print(json.load(open('ARM-STAGED.json'))['arm'])")
[ "$python_arm" = "$ARM" ] || { echo "staged arm $python_arm != $ARM"; exit 2; }
T2="$TASK_ROOT/experiments/imp-fronts/train2d"
OUT="$TASK_ROOT/output/train2d_s1"

if [ "$MODE" = smoke ] || [ "$MODE" = smokec ]; then
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
  STAGES=none timeout --signal=TERM --kill-after=10 300 bash "$BAD/run_seed.sh" 1 "$TMPDIR/cb1_out" > "$TASK_ROOT/logs/cb1.log" 2>&1
  RC=$?
  set -e
  cat "$TASK_ROOT/logs/cb1.log"
  if [ "$RC" = 1 ] && grep -q 'PROVENANCE sha256 MISMATCH' "$TASK_ROOT/logs/cb1.log"; then
    echo "C-B1 guard rejection: FIRED (rc=$RC)"; echo "{\"C_B1_guard_rejects\": true, \"rc\": $RC}" > "$TASK_ROOT/output/cb1.json"
  else
    echo "C-B1 guard rejection: DID NOT FIRE (rc=$RC)"; echo "{\"C_B1_guard_rejects\": false, \"rc\": $RC}" > "$TASK_ROOT/output/cb1.json"; exit 9
  fi
  rm -rf "$BAD" "$TMPDIR/cb1_out"
  if [ "$MODE" = smokec ]; then
    # SM-c (arm (c) 'bnd', DESIGN Amendments 19-21): identity tests on the GPU, paired incumbent / arm timing, chaining
    # slice of the bnd tree, projection (phase_times.py smokec -> output/timing_c.json only if everything passed)
    set +e
    LEFT=$(left); IRC=null
    if [ "$LEFT" -gt 0 ]; then
      timeout --signal=TERM --kill-after=30 "$LEFT" "$PY" -u "$TASK_ROOT/lane/tests/test_arm_c.py" --root "$TASK_ROOT/lane" \
        --k1-dir "$TASK_ROOT/lane/k1" --json "$TASK_ROOT/output/identity.json" 2>&1 | tee "$TASK_ROOT/logs/identity.log"
      IRC=${PIPESTATUS[0]}
    else echo "identity tests SKIPPED: self-timer exhausted (incomplete)"; fi
    LEFT=$(left)
    DRC=null
    if [ "$IRC" = 0 ] && [ "$LEFT" -gt 0 ]; then
      timeout --signal=TERM --kill-after=30 "$LEFT" "$PY" -u "$TASK_ROOT/smoke_bnd_timing.py" "$T2" \
        "$TASK_ROOT/output/smc_timing.json" --budget-s "$LEFT" 2>&1 | grep -v 'train2\[timing\] step' | tee "$TASK_ROOT/logs/smc_timing.log"
      DRC=${PIPESTATUS[0]}
    fi
    LEFT=$(( BUDGET - ($(date +%s) - T_START) ))
    CRC=null
    if [ "$DRC" = 0 ] && [ "$LEFT" -gt 30 ]; then
      TRAIN2D_DRYRUN=1 TRAIN2D_SMOKE=0 STAGES=A,B,R timeout --signal=TERM --kill-after=30 "$LEFT" bash "$T2/run_seed.sh" 1 "$OUT" \
        > "$TASK_ROOT/logs/chain.log" 2>&1
      CRC=$?
      tail -3 "$TASK_ROOT/logs/chain.log"
    fi
    echo "{\"chain_rc\": $CRC, \"seed_complete\": $([ -e "$OUT/SEED-COMPLETE" ] && echo true || echo false), \"identity_rc\": $IRC, \"timing_rc\": $DRC}" \
      > "$TASK_ROOT/output/chain.json"
    "$PY" "$TASK_ROOT/phase_times.py" smokec "$TASK_ROOT/output" "${SLURM_JOB_ID:-}"
    TRC=$?
    set -e
    cd "$TASK_ROOT"
    find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
    echo "SM-c status: identity_rc=$IRC timing_rc=$DRC chain_rc=$CRC projection_rc=$TRC"
    [ "$IRC" = 0 ] && [ "$DRC" = 0 ] && [ "$CRC" = 0 ] && [ "$TRC" = 0 ] || { echo "SMOKE FAILED (item uncertified)"; exit 1; }
    echo ALL-DONE
    exit 0
  fi
  LEFT=$(left)
  echo "smoke_datagen ($ARM): gates, burn-in, paired incumbent/arm timing; budget ${LEFT}s"
  set +e
  DRC=null
  if [ "$LEFT" -gt 0 ]; then
    timeout --signal=TERM --kill-after=30 "$LEFT" "$PY" -u "$TASK_ROOT/smoke_datagen.py" "$T2" "$TASK_ROOT/incumbent/burgers2d_film.py" "$ARM" \
      "$TASK_ROOT/output/datagen.json" --budget-s "$LEFT" 2>&1 | tee "$TASK_ROOT/logs/datagen.log"
    DRC=${PIPESTATUS[0]}
  else echo "smoke_datagen SKIPPED: self-timer exhausted (incomplete)"; fi
  set -e
  LEFT=$(( BUDGET - ($(date +%s) - T_START) ))
  CRC=null
  if [ "$DRC" = 0 ] && [ "$LEFT" -gt 30 ]; then
    echo "chaining slice (run_seed.sh TRAIN2D_DRYRUN=1; Amendment 15), stop ${LEFT}s"
    set +e
    TRAIN2D_DRYRUN=1 TRAIN2D_SMOKE=0 STAGES=A,B,R timeout --signal=TERM --kill-after=30 "$LEFT" bash "$T2/run_seed.sh" 1 "$OUT" \
      > "$TASK_ROOT/logs/chain.log" 2>&1
    CRC=$?
    set -e
    tail -3 "$TASK_ROOT/logs/chain.log"
  fi
  echo "{\"chain_rc\": $CRC, \"seed_complete\": $([ -e "$OUT/SEED-COMPLETE" ] && echo true || echo false), \"datagen_rc\": $DRC}" \
    > "$TASK_ROOT/output/chain.json"
  set +e
  "$PY" "$TASK_ROOT/phase_times.py" smoke "$TASK_ROOT/output" "$([ "$ARM" = fid ] && echo SM-a || echo SM-b)" "$ARM" \
    "${SLURM_JOB_ID:-}"
  TRC=$?
  set -e
  cd "$TASK_ROOT"
  find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
  echo "smoke status: datagen_rc=$DRC chain_rc=$CRC timing_rc=$TRC"
  [ "$DRC" = 0 ] && [ "$CRC" = 0 ] && [ "$TRC" = 0 ] || { echo "SMOKE FAILED (item uncertified)"; exit 1; }
  echo ALL-DONE
  exit 0
fi

# production disk evidence (job audit trcov-1): outputs ~ 0.6 GB coefficients + bank/checkpoints + logs + caches;
# require >= 50 GB available on the namespace's filesystem before starting (recorded in logs/disk.txt)
AVAIL_KB=$(df -Pk "$TASK_ROOT" | awk 'NR==2 {print $4}')
echo "disk available ${AVAIL_KB} kB under $TASK_ROOT" | tee "$TASK_ROOT/logs/disk.txt"
[ "${AVAIL_KB:-0}" -ge 52428800 ] || { printf '{"job": "%s", "arm": "%s", "outcome": "FAILED before start: < 50 GB available (%s kB)"}\n' \
  "$JOB" "$ARM" "$AVAIL_KB" > "$TASK_ROOT/FAILED.json"; cat "$TASK_ROOT/FAILED.json"; exit 8; }
LEFT=$(( BUDGET - ($(date +%s) - T_START) ))
# Production mode is forced (no inheritance of TRAIN2D_DRYRUN / TRAIN2D_SMOKE / STAGES). "No self-timer" = no
# fractional job cutoff (Amendment 11); the unchanged Stage B recipe keeps its TIME_CAP=1500 s head-fit stop, recorded
# as time_cap_fired in stageB/hfit_mid_summary.json. The cost anchor is the slowest incumbent-recipe retrain (jcp-robust
# seed 2, 15,018 s), not K1's own seed-1 run.
echo "run_seed.sh seed 1 (production), stop ${LEFT}s (hard wall minus 120 s recording reserve; no self-timer)"
[ "$LEFT" -gt 0 ] || { printf '{"job": "%s", "arm": "%s", "outcome": "FAILED before start: no time left (%s s)"}\n' "$JOB" "$ARM" "$LEFT" > "$TASK_ROOT/FAILED.json"; exit 1; }
set +e
TRAIN2D_DRYRUN=0 TRAIN2D_SMOKE=0 STAGES=A,B,R timeout --signal=TERM --kill-after=60 "$LEFT" \
  bash "$T2/run_seed.sh" 1 "$OUT" 2>&1 | tee "$TASK_ROOT/logs/run_seed.log"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" = 124 ] || [ "$RC" = 137 ]; then
  printf '{"job": "%s", "arm": "%s", "rc": %s, "stop_s": %s, "outcome": "arm NOT RUN (DESIGN Amendment 11: 4.6 GPU-h training ceiling exhausted)"}\n' \
    "$JOB" "$ARM" "$RC" "$LEFT" > "$TASK_ROOT/EXHAUSTED.json"
  cat "$TASK_ROOT/EXHAUSTED.json"
fi
if [ "$RC" != 0 ] && grep -q 'ARM-DATA-GATE-FAILED' "$TASK_ROOT/logs/run_seed.log"; then
  # Amendment 17: the cov arm's per-chunk data gate (finite + Newton rel residual <= 1e-8) fired -> arm NOT RUN
  printf '{"job": "%s", "arm": "%s", "rc": %s, "outcome": "arm NOT RUN (DESIGN Amendment 17: per-chunk data gate failed)", "detail": "%s"}\n' \
    "$JOB" "$ARM" "$RC" "$(grep -m1 -o 'ARM-DATA-GATE-FAILED[^"]*' "$TASK_ROOT/logs/run_seed.log")" > "$TASK_ROOT/EXHAUSTED.json"
  cat "$TASK_ROOT/EXHAUSTED.json"
fi
set +e
"$PY" "$TASK_ROOT/phase_times.py" --production "$OUT" "$JOB" "$ARM" "$MODE" "$WALL_S" "$BUDGET" \
  "$TASK_ROOT/output/phase_times_$JOB.json"
PRC=$?
# completion contract: rc 0, SEED-COMPLETE, a production run_record.json for seed 1, the dense mid checkpoint and the
# rotation file present and non-empty
"$PY" -I - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1]); bad = []
for f in ('SEED-COMPLETE', 'run_record.json', 'stageB/sep_hfit_dense_mid_N256_dense.pkl', 'rotation/rotation_R512.npz',
          'rotation/rotation_R512.json', 'stageA/bank.pkl'):
    if not (o / f).is_file() or (o / f).stat().st_size == 0:
        bad.append(f'missing {f}')
try:
    r = json.loads((o / 'run_record.json').read_text())
    if r.get('mode') != 'production' or r.get('seed') != 1:
        bad.append(f"run_record mode/seed {r.get('mode')}/{r.get('seed')}")
    print('time_cap_fired', (r.get('stageB_mid') or {}).get('time_cap_fired'))
except Exception as e:
    bad.append(f'run_record unreadable: {e}')
print('COMPLETION ' + ('OK' if not bad else 'FAILED: ' + '; '.join(bad)))
sys.exit(1 if bad else 0)
PYEOF
CPL=$?
set -e
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
if [ "$RC" != 0 ] || [ "$CPL" != 0 ] || [ "$PRC" != 0 ]; then
  [ -e EXHAUSTED.json ] || printf '{"job": "%s", "arm": "%s", "rc": %s, "completion_rc": %s, "phase_times_rc": %s, "outcome": "FAILED (arm NOT RUN unless rerun under the admission rule)"}\n' \
    "$JOB" "$ARM" "$RC" "$CPL" "$PRC" > FAILED.json
  cat EXHAUSTED.json FAILED.json 2>/dev/null
  exit 1
fi
echo ALL-DONE
