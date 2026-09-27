#!/bin/bash
#SBATCH --job-name=bsd_s3
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b_seeds_20260917/s3/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b_seeds_20260917/s3/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b_seeds_20260917/s3
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT attempt=s3 started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
LADDER_PATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-seeds"

# ---------------------------------------------------------------- stage A: the bank ----
# run_r3a.sbatch (job 2835788) verbatim except SEED0, TRAIN_ONLY=1 and TIME_CAP=0 (DESIGN.md D1)
mkdir -p "$TASK_ROOT/output/train"
cd "$TASK_ROOT/experiments/separable-decoder"
echo "STAGE A bank seed=3 $(date -Is)"
env ROUND=3 K=16 LR=1e-3 P_SUB=4096 WD=1e-5 EMA_DECAY=0.999 LAM_ORTH=1e-4 \
  MAX_SNAPS=16384 T_EARLY=5 FULLROWS=64 \
  N_FF=128 FF_SCALE=4.0 H_HIDDEN=256 \
  EQ_MS=64,256 EQ_CAND_CAP=65536 STEP_TOLS=1e-9,1e-6 \
  N_TEST=8 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 SEED0=3 \
  IC_TOP=12 IC_ENC_BUDGET=50 ENC_STEPS=12000 EXTRAP=1.0 \
  ORACLE_BUDGET=150 SSTEP_TS=1,2,3,5,10,25,50 SSTEP_BUDGET=120 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  MAX_NEWTON=20 BATCHED=1 \
  N=256 R=512 G_HIDDEN=1024 SNAP_NORM=0 STEPS=300000 TIME_CAP=0 FULL_LAST=10000 POOL=0 MESH_ARM=1 \
  TRAIN_ONLY=1 OUT_PREFIX="$TASK_ROOT/output/train/" \
  "$PY" sep_burgers_r3.py
BANK="$TASK_ROOT/output/train/sep_burgers_r3_N256_K16_R512.pkl"
test -s "$BANK"

# --------------------------------------------- stage B: coefficient extraction ----
# run_dn256b.sbatch (job 2837431) extraction verbatim except SEED0 (the state pick)
echo "STAGE B extract seed=3 $(date -Is)"
env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=3 LOOSE=1 \
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \
  CKPT="$BANK" OUT_PREFIX="$TASK_ROOT/output/train/" \
  "$PY" sep_coeff_extract.py
NPZ="$TASK_ROOT/output/train/sep_coeff_N256_K16_R512.npz"
test -s "$NPZ"

# ------------------------------------------------- stage C: the head refit ('mid') ----
# run_dn256b.sbatch hfit-full verbatim except SEED0, ARMS=mid only, TIME_CAP=0 (DESIGN.md D1)
echo "STAGE C hfit seed=3 $(date -Is)"
env NPZ="$NPZ" CKPT="$BANK" OUT="$TASK_ROOT/output/train/hfit_full.json" \
  ARMS=mid STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=0 \
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=3 \
  EMIT=mid EMIT_PATH="$TASK_ROOT/output/train/sep_hfit_seed3.pkl" \
  "$PY" sep_hfit_run.py
SEEDCK="$TASK_ROOT/output/train/sep_hfit_seed3.pkl"
test -s "$SEEDCK"
sha256sum "$BANK" "$NPZ" "$SEEDCK" | tee "$TASK_ROOT/output/train/TRAIN-SHA256.txt"
cd "$TASK_ROOT"

# ------------------------------------------- stage D: the seed's development ladder ----
echo "STAGE D ladder seed=3 $(date -Is)"
PYTHONPATH="$LADDER_PATH" "$PY" experiments/b-seeds/seeds_run.py \
  --config experiments/b-seeds/config-dev-seed3.json \
  --checkpoint "$SEEDCK" --out output/ladder_seed

# -------------------------------- stage E: the incumbent's development ladder, same job ----
echo "STAGE E ladder incumbent $(date -Is)"
PYTHONPATH="$LADDER_PATH" "$PY" experiments/b-seeds/seeds_run.py \
  --config experiments/b-seeds/config-dev-incumbent.json \
  --checkpoint experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl --out output/ladder_incumbent

# Stages A-E are the deliverable: checksum them NOW so a wall-clock kill during the optional
# stage F cannot void the attempt (A1.8). OUTPUTS.sha256 is rewritten after F.
echo "STAGES A-E DONE $(date -Is)" | tee output/STAGES-AE-DONE
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256

# ------------------------------ stage F: EQ certification, q <= 64 (non-fatal, optional) ----
echo "STAGE F eqcert seed=3 $(date -Is)"
set +e
PYTHONPATH="$LADDER_PATH" timeout 4h "$PY" experiments/q-ridge/q_eqcert.py \
  --config experiments/b-seeds/config-eqcert-seed3.json \
  --checkpoint "$SEEDCK" --out output/eqcert_seed
rc=$?
set -e
if [ "$rc" -ne 0 ]; then echo "EQCERT FAILED rc=$rc $(date -Is)" | tee output/EQCERT-FAILED; fi

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
