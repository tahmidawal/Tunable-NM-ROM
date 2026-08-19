#!/bin/bash
#SBATCH -J ctol_hybp_pairfinal1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybp1024/pairfinal1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybp1024/pairfinal1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybp1024/pairfinal1/code/poisson-hybrid-1024"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=92447ead81a9c37609d1f0127de98300e5a927a3
echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=92447ead81a9c37609d1f0127de98300e5a927a3 dirty_hash=e3b0c44298fc cell=pairfinal1"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export PKL=input/auto.pkl
export PARAM_PKL=input/param.pkl
export GROUP_PKL=input/groupfilm.pkl

export SMOKE=0 NS=512,1024 FOM_TAUS=1e-6,1e-8,1e-10 N_TEST=16 N_TIME=8 TIME_REPS=12 TIME_WARM=2 BURN_S=3 TEST_SEED=20260820 M=64 MQ=256 GN_ITERS=60 LM_ROM_TAU=0.01 TR_SCALE=1.0 PAIRWISE_DIAGNOSTIC=0 ARMS=lmtrmean_c64_q0 NATIVE_ARMS= BALANCED_PAIR_ARM=lmtrmean_c64_q0 BOOTSTRAP_REPS=10000
$PY feasibility.py "/cluster/tufts/paralab/tawal01/hybp1024/pairfinal1/out/pairfinal1.json"
echo ALL-DONE
