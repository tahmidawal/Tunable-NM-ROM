#!/bin/bash
#SBATCH --job-name=n256push_r1b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r1_burgers/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r1_burgers/logs/%j.err

# N=256 push ROUND 1, Burgers: IC repair (high-budget fit + u0->z0 encoder),
# adaptive trust region + restart-on-stall, safeguarded 2-step latent
# warm-start extrapolation, EQ arms (m512 / M128 / tail-reweight), state
# oracles + single-step weak-opt ladder diagnostics.  PUSH-PLAN.md round 1.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r1_burgers
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
env N=256 K=16 R=64 STEPS=60000 N_TEST=8 REPS=5 WARM=2 \
  MAX_SNAPS=8192 TR_FACTOR=0.01 SEED0=0 \
  M_CTRL=64 MQ_CTRL=256 M_BIG=128 MQ_BIG=512 \
  IC_TOP=12 IC_TOP_HI=48 IC_BUDGET_HI=300 IC_ENC_BUDGET=50 ENC_STEPS=8000 \
  EXTRAP=1.0 ADAPT_GROW=2.0 ADAPT_SHRINK=0.5 DMIN_FACTOR=1e-4 DMAX_FACTOR=1.0 \
  N_STEP_RESTARTS=2 STEP_RESTART_SIG=0.05 TAIL_CAP=3e-2 TAIL_ROUNDS=3 \
  ORACLE_BUDGET=150 SSTEP_TS=1,10,25,50 SSTEP_BUDGET=120 \
  NEWTON_TOLS=1e-2,1e-3,1e-5,1e-8 LIN_FACTOR=1e-2 MAX_NEWTON=20 \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_burgers_r1.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
