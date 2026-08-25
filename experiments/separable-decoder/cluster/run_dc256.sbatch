#!/bin/bash
#SBATCH --job-name=burgacc_dc256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:50:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/dc256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/dc256/logs/%j.err

# ROUND 5 WAVE 7 -- CAPACITY AT HIGH mu-DENSITY.
#
# The density ladder SATURATES for a fixed h: held-out oracle 1.375e-2 ->
# 8.017e-3 -> 6.013e-3 at 576 -> 1152 -> 4096 trajectories (exponent 0.78 over
# the first doubling, 0.22 over the last), and the same arm's TRAINING
# reconstruction degrades from 2.22e-3 to 5.85e-3 as it is asked to fit 7x more
# states.  That is a capacity limit reasserting itself, exactly as the low-data
# arms predicted in reverse (at 72 trajectories the SMALL h generalised better
# than the wide one).  So capacity and density have to be scaled together, and
# this job sweeps capacity and K at the high-density end.
#
# Reads the dense coefficient file produced by job dn256 by COPYING it into
# this job's own directory first -- no two jobs ever share a directory.
set -uo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/dc256
SRC=/cluster/tufts/paralab/tawal01/burgacc/dn256/out
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
cp "$SRC/sep_coeff_N256_K16_R512.npz" "$SRC/sep_coeff_N256_K16_R512.json" "$ROOT/in/"
echo "copied dense coefficients into this job's own dir"

cd "$ROOT/code"
env NPZ="$ROOT/in/sep_coeff_N256_K16_R512.npz" \
  CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" \
  OUT="$ROOT/out/hfit_dc256.json" \
  ARMS=huge,k24,k32,k48 STEPS=120000 BATCH=4096 LR=1e-3 TIME_CAP=700 \
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 \
  SEED0=0 TRAJ_FIT=4096 HOLD_FROM=4096 HOLD_N=512 \
  "$PY" sep_hfit_run.py || echo "HFIT FAILED"

cd "$ROOT/out" && sha256sum * > RESULTS.sha256
echo ALL-DONE
