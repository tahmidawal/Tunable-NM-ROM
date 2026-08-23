#!/bin/bash
#SBATCH --job-name=n256push_r2a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r2a_poisson/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r2a_poisson/logs/%j.err

# N=256 push ROUND 2a, Poisson: R=512 per the POD-floor verdict (dense-family
# fresh floor 4.8e-4 mean at R=512), S=512+2048 snapshots, MULTI-SCALE Fourier
# features, wider tracks, v2 trainer (point-subsampled AdamW+EMA), objective
# certification at M in {64,128,256} + tail.  r2b is the single-scale control.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r2a_poisson
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
env N=256 K=16 R=512 STEPS=300000 LR=1e-3 P_SUB=16384 WD=1e-5 \
  EMA_DECAY=0.999 FULL_LAST=20000 TIME_CAP=13500 LAM_ORTH=1e-4 \
  N_EXTRA=2048 EXTRA_SEED=4242 \
  N_FF=128 FF_SCALES=1,4,16 G_HIDDEN=256 H_HIDDEN=256 \
  N_TEST=16 REPS=9 WARM=2 GN_ITERS=60 BUDGET_HI=300 N_RESTARTS=6 \
  TAUS=1e-3,1e-2 TR_FACTOR=1.0 SEED0=0 FRESH_SEED=777 \
  FOM_LADDER=5e-1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-6 \
  EQ_MS=64,128,256 TAIL_CAP=3e-2 TAIL_ROUNDS=2 ENC_STEPS=12000 \
  ORACLE_BUDGET=300 WOPT_BUDGET=600 \
  OUT="$ROOT/out/sep_poisson_r2a.json" \
  CKPT="$ROOT/out/sep_poisson_r2a_N256_K16_R512.pkl" \
  "$PY" sep_poisson_r2.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
