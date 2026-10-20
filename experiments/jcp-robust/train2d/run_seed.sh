#!/bin/bash
# jcp-robust E2 2D: the frozen dn256b recipe for one recipe seed (DESIGN.md 3.1-3.2, A1.6, A2.10).
#
#   PY=<python> bash experiments/jcp-robust/train2d/run_seed.sh <seed> <outdir>
#
# Stage A  smoothtrain.py --arm base --sigma 4 --lam 0 --nodes 256 --seed <seed>     (bank + 16,384 codes, 300k steps)
# Stage B  sep_coeff_extract.py (data stream fixed: SEED0=0) then sep_hfit_run.py "full" call (SEED0=<seed>), EMIT=mid
# Rotation make_rotation.py on the emitted dense mid checkpoint -> rotation_R512.npz/.json  (the DEPLOYMENT rotation)
#
# The script lives at <jobroot>/experiments/jcp-robust/train2d/ and reads nothing outside that directory except the
# python interpreter ($PY). smoothtrain's own least-squares rotation is renamed DIAGNOSTIC-ONLY_* and never used.
#
# Optional env:  STAGES=A,B,R (default all; a stage whose marker exists is skipped after its outputs are re-hashed)
#                TRAIN2D_DRYRUN=1  tiny CPU chaining test (N=64, 16 trajectories, 20 steps; >= R=512 states needed); NEVER for results.
#                TRAIN2D_SMOKE=1   real-size GPU smoke (DESIGN 6 smk2): every size of the recipe, only the step counts
#                                  truncated (Stage A 3,000 steps, full_last 500; head fit 2,000 steps); NEVER for results.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: PY=<python> $0 <seed> <outdir>" >&2; exit 2; }
SEED_S=$1
OUTDIR=$2
case "$SEED_S" in ''|*[!0-9]*) echo "seed must be a non-negative integer" >&2; exit 2;; esac
: "${PY:?set PY to the python interpreter (cluster: /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${STAGES:-A,B,R}"
DRY="${TRAIN2D_DRYRUN:-0}"
SMOKE="${TRAIN2D_SMOKE:-0}"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
[ ! -e "$OUTDIR/SEED-COMPLETE" ] || { echo "$OUTDIR/SEED-COMPLETE exists: refusing to rerun" >&2; exit 3; }

# Every environment name the vendored modules read at import or at run time (burgers2d_film, ms_parametric,
# ms_autodecoder, blat_common, sep_common.arch_from_env, sep_coeff_extract, sep_hfit_run, smoothtrain). The dn256b job
# set none of them globally; they are cleared here so that only the explicit per-command `env` below applies
# (in particular SEED and STEPS, which burgers2d_film reads as its DATA seed and step count).
for v in N K R SEED SEED0 STEPS N_TRAIN N_VAL N_TRAJ N_TEST TEST_SEED HIDDEN N_LAYERS AD_HIDDEN AD_LAYERS K_LAT \
         BC_MODE DECODER_ARCH N_FF FF_SCALE FF_SCALES G_HIDDEN G_LAYERS H_HIDDEN H_LAYERS H_FF H_FF_SCALE \
         MAX_SNAPS T_EARLY LOOSE EXTRA_SEED EXTRA_TRAJ GEN_CHUNK PROJ_CHUNK FEAT_CHUNK IDENT_ROWS OUT_PREFIX \
         CKPT NPZ OUT ARMS BATCH LR TIME_CAP ORACLE_ITERS ORACLE_EVERY CODEDIAG_N CODEDIAG_ITERS ENC_STEPS EMIT \
         EMIT_PATH TRAJ_FIT HOLD_FROM HOLD_N P_SUB TR_FACTOR GN_BUDGET GN_TOL GN_ITERS IC_BUDGET M_EQ EQ_POOL \
         EQ_SNAPS N_STAGES N_FREQ T_FREQ P_POINTS GT_STEPS GT_RANK EVAL_EVERY NEWTON_ITERS LIN_TOL LIN_MAXITER \
         POD_TIME_STRIDE PEAK_LR CONST_LR LAT_LR LAT_REG ADAM_INFER_STEPS LBFGS_STEPS Z_FF Z_WIDTH FILM_START \
         FILM_GROUP_SIZE FREQ_MARGIN RESIDUAL_SCALE WARP_MAX_LOG_SCALE WARP_MAX_SHIFT WEAK_ALPHA; do
  unset "$v"
done
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONDONTWRITEBYTECODE=1

log() { echo "[train2d seed=$SEED_S $(date -Is)] $*"; }

# ---- frozen recipe values (DESIGN A1.6); the dry run replaces only sizes, never the code path
if [ "$DRY" = "1" ]; then
  MODE=dryrun; A_EXTRA=(--nodes 64 --n-traj 16 --steps 20 --log-every 10 --allow-cpu)
  NB=64; EX_ENV=(N_TRAJ=16 MAX_SNAPS=512 EXTRA_TRAJ=8 N_TEST=2)
  HF_ENV=(STEPS=20 ORACLE_ITERS=5 CODEDIAG_N=64 CODEDIAG_ITERS=5 ENC_STEPS=20)
  ROT_EXTRA=(--train-mesh 64)
elif [ "$SMOKE" = "1" ]; then
  MODE=smoke; A_EXTRA=(--nodes 256 --steps 3000 --full-last 500 --log-every 500)
  NB=256; EX_ENV=(MAX_SNAPS=131072 EXTRA_TRAJ=4032 N_TEST=8)
  HF_ENV=(STEPS=2000 ORACLE_ITERS=20 CODEDIAG_N=256 CODEDIAG_ITERS=10 ENC_STEPS=500)
  ROT_EXTRA=()
else
  MODE=production; A_EXTRA=(--nodes 256)
  NB=256; EX_ENV=(MAX_SNAPS=131072 EXTRA_TRAJ=4032 N_TEST=8)
  HF_ENV=(STEPS=200000 ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000)
  ROT_EXTRA=()
fi
A="$OUTDIR/stageA"; B="$OUTDIR/stageB"; RO="$OUTDIR/rotation"
mkdir -p "$A" "$B" "$RO"

log "mode=$MODE host=$(hostname) stages=$STAGES jobroot_commit=${SOURCE_COMMIT:-unset}"
{ nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"; } \
  | tee -a "$OUTDIR/device.txt"
if [ "$DRY" != "1" ]; then
  "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" \
    | tee -a "$OUTDIR/device.txt" || exit 42
fi
# vendored files must equal their recorded sha256 (PROVENANCE.json) before anything runs
"$PY" -I - "$HERE" <<'EOF'
import hashlib, json, sys
from pathlib import Path
here = Path(sys.argv[1])
bad = [f['path'] for f in json.loads((here / 'PROVENANCE.json').read_text())['files']
       if hashlib.sha256((here / f['path']).read_bytes()).hexdigest() != f['sha256']]
print('PROVENANCE sha256 ' + ('OK' if not bad else 'MISMATCH ' + ' '.join(bad)), flush=True)
sys.exit(1 if bad else 0)
EOF

want() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
done_ok() { [ -e "$1/STAGE-COMPLETE" ] && ( cd "$1" && sha256sum -c --quiet STAGE-COMPLETE ); }
mark() { ( cd "$1" && find . -maxdepth 1 -type f ! -name STAGE-COMPLETE -printf '%P\n' | sort | xargs sha256sum > STAGE-COMPLETE ); }

# ------------------------------------------------------------------------------------------- Stage A
if want A && ! done_ok "$A"; then
  log "Stage A: smoothtrain --arm base --sigma 4 --lam 0 --seed $SEED_S ${A_EXTRA[*]}"
  t0=$SECONDS
  "$PY" -u "$HERE/stageA/smoothtrain.py" --arm base --sigma 4 --lam 0 --seed "$SEED_S" "${A_EXTRA[@]}" --out "$A" \
    2>&1 | tee "$A/stageA.log"
  # its least-squares rotation is NOT the deployment rotation (DESIGN A1.6): rename so nothing can pick it up
  mv "$A/rotation.npz" "$A/DIAGNOSTIC-ONLY_ls_rotation.npz"
  "$PY" -I - "$A/train.json" "$SEED_S" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['complete'] is True and r['seed'] == int(sys.argv[2]) and r['arm'] == 'base' and r['sigma'] == 4.0 \
    and r['lam_sob'] == 0.0, 'Stage A train.json is not a complete base/sigma4/lam0 run of this seed'
print('Stage A OK: steps_done', r['train']['steps_done'], 'recon mean', r['train']['recon_rel_l2_mean'],
      'R2a checked', r['R2a_checked'], r['train'].get('R2a'))
EOF
  echo "$((SECONDS - t0))" > "$A/seconds.txt"
  mark "$A"
fi

# ------------------------------------------------------------------------------------------- Stage B
if want B && ! done_ok "$B"; then
  done_ok "$A" || { echo "Stage A outputs missing or changed" >&2; exit 4; }
  CK="$A/bank.pkl"                       # smoothtrain writes the sep_common pickle {params, Z_tr, cfg}: used as-is
  t0=$SECONDS
  cd "$HERE/stageB/code"
  log "Stage B extraction (data stream SEED0=0)"
  env N=$NB K=16 R=512 "${EX_ENV[@]}" T_EARLY=5 SEED0=0 LOOSE=1 EXTRA_SEED=1000 GEN_CHUNK=64 PROJ_CHUNK=256 \
    IDENT_ROWS=64 CKPT="$CK" OUT_PREFIX="$B/" \
    "$PY" -u sep_coeff_extract.py 2>&1 | tee "$B/extract.log"
  NPZ="$B/sep_coeff_N${NB}_K16_R512.npz"
  [ -s "$NPZ" ] || { echo "EXTRACT FAILED (no $NPZ)" >&2; exit 5; }
  log "Stage B head fit (full call, SEED0=$SEED_S)"
  env NPZ="$NPZ" CKPT="$CK" OUT="$B/hfit_full.json" ARMS=mid,wide,k32 "${HF_ENV[@]}" BATCH=4096 LR=1e-3 \
    TIME_CAP=1500 SEED0="$SEED_S" EMIT=mid EMIT_PATH="$B/sep_hfit_dense_mid_N${NB}_dense.pkl" \
    "$PY" -u sep_hfit_run.py 2>&1 | tee "$B/hfit.log"
  cd "$OUTDIR"
  "$PY" -I - "$B" "$NB" "$SEED_S" <<'EOF'
import json, sys
from pathlib import Path
b, nb, s = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ex = json.load(open(b / f'sep_coeff_N{nb}_K16_R512.json'))
hf = json.load(open(b / 'hfit_full.json'))
assert ex['complete'] is True and hf['complete'] is True and hf['config']['seed'] == s
assert Path(hf['emitted']).name == f'sep_hfit_dense_mid_N{nb}_dense.pkl', hf['emitted']
t = hf['arms']['mid']['train']
summ = dict(seed=s, steps=t['steps'], steps_done=t['steps_done'], seconds=t['seconds'],
            time_cap_s=1500, time_cap_fired=bool(t['steps_done'] < t['steps']), final_loss=t.get('final_loss'),
            gpu=hf['config'].get('gpu'), extraction_gates=ex.get('gates'),
            note='time_cap_fired=True is a recipe deviation for this seed (DESIGN A2.10)')
(b / 'hfit_mid_summary.json').write_text(json.dumps(summ, indent=1, default=float) + '\n')
print('Stage B OK:', json.dumps({k: summ[k] for k in ('steps_done', 'steps', 'seconds', 'time_cap_fired', 'gpu')}))
EOF
  echo "$((SECONDS - t0))" > "$B/seconds.txt"
  mark "$B"
fi

# ------------------------------------------------------------------------------------------- Rotation
if want R && ! done_ok "$RO"; then
  done_ok "$B" || { echo "Stage B outputs missing or changed" >&2; exit 4; }
  t0=$SECONDS
  log "Rotation (make_rotation.py, head-based, deployed procedure)"
  cd "$HERE/rotation"
  # --directions only feeds the diagnostic `correction_directions_energy_captured` (the incumbent's correction
  # directions); it does not enter T or L. Same file as the incumbent's run (sha256 79d79458...).
  PYTHONPATH="$HERE/rotation" "$PY" -u make_rotation.py --checkpoint "$B/sep_hfit_dense_mid_N${NB}_dense.pkl" \
    --directions inputs/directions_qtd02.npz --out "$RO" "${ROT_EXTRA[@]}" 2>&1 | tee "$RO/rotation.log"
  cd "$OUTDIR"
  "$PY" -I - "$RO/rotation_R512.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print('Rotation OK: ||LT-I|| =', r['L_times_T_identity_deviation'], ' orth dev =',
      r['rotated_bank_orthonormality_deviation_at_train_mesh'], ' cond(R_G) =', r['R_G_condition_number'])
EOF
  echo "$((SECONDS - t0))" > "$RO/seconds.txt"
  mark "$RO"
fi

# ------------------------------------------------------------------------------------------- record
if done_ok "$A" && done_ok "$B" && done_ok "$RO"; then
  "$PY" -I - "$OUTDIR" "$SEED_S" "$MODE" <<'EOF'
import json, sys
from pathlib import Path
o, s, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
a = json.load(open(o / 'stageA/train.json'))
h = json.load(open(o / 'stageB/hfit_mid_summary.json'))
r = json.load(open(o / 'rotation/rotation_R512.json'))
rec = dict(seed=s, mode=mode, device=(o / 'device.txt').read_text().strip().splitlines(),
           stageA=dict(gpu=a['gpu'], steps_done=a['train']['steps_done'], seconds=a['train']['seconds'],
                       recon_rel_l2_mean=a['train']['recon_rel_l2_mean'], R2a=a['train'].get('R2a'),
                       data_R0=a['data'].get('R0_fingerprint_matches_r3a'), bank_sha256=a['bank_sha256']),
           stageB_mid=h,
           rotation=dict(L_times_T_identity_deviation=r['L_times_T_identity_deviation'],
                         orthonormality=r['rotated_bank_orthonormality_deviation_at_train_mesh'],
                         R_G_condition_number=r['R_G_condition_number'], file_sha256=r['file_sha256'],
                         checkpoint_sha256=r['checkpoint_sha256']),
           stage_seconds={k: int((o / k / 'seconds.txt').read_text()) for k in ('stageA', 'stageB', 'rotation')
                          if (o / k / 'seconds.txt').exists()})
(o / 'run_record.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
EOF
  ( cd "$OUTDIR" && find . -type f ! -name OUTPUTS.sha256 ! -name SEED-COMPLETE -printf '%P\n' | sort \
      | xargs sha256sum > OUTPUTS.sha256 )
  date -Is > "$OUTDIR/SEED-COMPLETE"
  log "SEED-COMPLETE ($OUTDIR)"
else
  log "stages done so far: $(for d in stageA stageB rotation; do done_ok "$OUTDIR/$d" && printf '%s ' $d; done); SEED-COMPLETE not written"
fi
