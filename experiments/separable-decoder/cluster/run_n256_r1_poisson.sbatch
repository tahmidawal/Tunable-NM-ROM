#!/bin/bash
#SBATCH --job-name=n256push_r1p
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r1_poisson/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r1_poisson/logs/%j.err

# N=256 push ROUND 1, Poisson: solver-termination repair (budget/restarts),
# EQ arms (m512 / M128 / tail-reweight), f_m->z encoder init, adaptive
# quadrature diagnostic, full error ladder.  PUSH-PLAN.md round 1.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r1_poisson
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
env N=256 K=16 R=64 STEPS=200000 N_TEST=16 REPS=9 WARM=2 \
  GN_ITERS=60 BUDGET_HI=300 N_RESTARTS=6 RESTART_SIG=0.3 \
  TAUS=1e-3,1e-2 TR_FACTOR=1.0 SEED0=0 FRESH_SEED=777 \
  FOM_LADDER=5e-1,3e-1,1e-1,3e-2,1e-2,1e-3,1e-6 \
  M_CTRL=64 MQ_CTRL=256 M_BIG=128 MQ_BIG=512 \
  TAIL_CAP=3e-2 TAIL_ROUNDS=3 ENC_STEPS=8000 \
  ORACLE_BUDGET=250 WOPT_BUDGET=600 ADQ_N=8 ADQ_PERT=3 \
  OUT="$ROOT/out/sep_poisson_r1.json" \
  CKPT="$ROOT/out/sep_poisson_r1_N256_K16_R64.pkl" \
  "$PY" sep_poisson_r1.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
