#!/bin/bash
#SBATCH --job-name=qf_n256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b1dqf/qf_n256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b1dqf/qf_n256/logs/%j.err

# QUADRATURE-FREE EXACT WEAK RESIDUAL, POISSON N=256 (exp/2026-08-27-b1d-poissonqf,
# part B).  Frozen decoder sep_poisson_N256_K16_R64.pkl (n256_j1, job 2825729);
# NO training.  sep_poisson_qf.py compares FULL (all interior nodes, exact
# weights) vs incumbent NNLS-EQ (m=256, refit in-job) vs QF
# (r(z)=Wl*(B h(z))-f_m, B precomputed) through the same lm_tau_poisson solver;
# gates Q/S/F/E must pass or the job aborts.  M=64, m=256, gn_iters=60,
# tr_factor=1, fresh_seed=777 all default from the checkpoint's own cfg.
# Stage first:  ./stage_qf.sh qf_n256 ../runs/inherited_qf/sep_poisson_N256_K16_R64.pkl
# then rsync stage/qf_n256/ to $ROOT below before sbatch.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b1dqf/qf_n256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_poisson_N256_K16_R64.pkl"

cd "$ROOT/code"
env CKPT="$ROOT/in/sep_poisson_N256_K16_R64.pkl" N=256 \
  N_SRC=16 REPS=12 BURN=3 SEED0=0 GATE_NZ=32 \
  OUT_TAG=qf_N256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_poisson_qf.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
