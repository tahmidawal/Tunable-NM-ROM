#!/bin/bash
#SBATCH --job-name=burgacc_dn256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:20:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/dn256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/dn256/logs/%j.err

# ROUND 5 WAVE 4 -- DENSER mu-SAMPLING at N=256, on the FROZEN r3a bank.
#
# Waves 2 and 3 located the remaining Burgers error precisely: h fits the
# training snapshots' span coefficients well (recon 7.2e-3 -> 1.4e-3 with
# capacity) but its K-dimensional image does not pass near FRESH ones (oracle
# stuck at 6-8e-3), and cutting the fit from 576 to 72 trajectories collapses
# the held-out oracle to 7.2e-2.  That is a mu-sampling-density limit, not a
# capacity limit, so this job appends trajectories from a fresh seed (the
# canonical seed-0 draw and the cohort definitions are untouched) and measures
# the learning curve in trajectory count on ONE reserved cohort.
#
# Extraction and refinement run in the SAME job because the dense npz is ~0.5 GB
# and there is no reason to move it.  The bank is the frozen r3a checkpoint
# throughout, so the span floor is one constant and oracle/span-floor is
# comparable with every wave-2/3 row.
# Sized to finish before the 2026-08-25T06:00 maintenance window.
set -uo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/dn256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 \
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \
  CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_coeff_extract.py || { echo "EXTRACT FAILED"; exit 1; }

for F in 576 1152 4096; do
  echo "=== hfit TRAJ_FIT=$F ==="
  env NPZ="$ROOT/out/sep_coeff_N256_K16_R512.npz" \
    CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" \
    OUT="$ROOT/out/hfit_dn256_f$F.json" \
    ARMS=wide,k24 STEPS=120000 BATCH=4096 LR=1e-3 TIME_CAP=900 \
    ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 \
    SEED0=0 TRAJ_FIT=$F HOLD_FROM=4096 HOLD_N=512 \
    "$PY" sep_hfit_run.py || echo "hfit F=$F FAILED"
done

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
