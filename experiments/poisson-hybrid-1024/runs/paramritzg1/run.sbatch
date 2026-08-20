#!/bin/bash
#SBATCH -J ctol_hybp_paramritzg1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybp1024/paramritzg1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybp1024/paramritzg1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybp1024/paramritzg1/code/poisson-hybrid-1024"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=46127cd773206707fcdbc5cdfa8196d71e814b6c
echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=46127cd773206707fcdbc5cdfa8196d71e814b6c dirty_hash=e3b0c44298fc cell=paramritzg1"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export PKL=input/auto.pkl
export PARAM_PKL=input/param.pkl
export GROUP_PKL=input/groupfilm.pkl

export SMOKE=0 NS=64,256 FOM_TAUS=1e-6 N_TEST=16 N_TIME=8 TIME_REPS=6 TIME_WARM=2 BURN_S=3 TEST_SEED=20260821 EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 ARMS=param1_c64_q0,paramritz1_m24_c64_q0,spectral_q256 NATIVE_ARMS= PAIRWISE_DIAGNOSTIC=1 BOOTSTRAP_REPS=10000 PARAM_LM_DELTA=0.35
$PY feasibility.py "/cluster/tufts/paralab/tawal01/hybp1024/paramritzg1/out/paramritzg1.json"
echo ALL-DONE
