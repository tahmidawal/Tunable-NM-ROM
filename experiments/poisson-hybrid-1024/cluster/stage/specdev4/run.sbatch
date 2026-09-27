#!/bin/bash
#SBATCH -J ctol_hybp_specdev4
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybp1024/specdev4/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybp1024/specdev4/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybp1024/specdev4/code/poisson-hybrid-1024"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=b7a5263fba818febf1dee59390a5caf101db9824
echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=b7a5263fba818febf1dee59390a5caf101db9824 dirty_hash=e3b0c44298fc cell=specdev4"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export PKL=input/auto.pkl
export PARAM_PKL=input/param.pkl
export GROUP_PKL=input/groupfilm.pkl

export SMOKE=0 NS=1024 FOM_TAUS=1e-6 N_TEST=16 N_TIME=6 TIME_REPS=7 TIME_WARM=2 BURN_S=3 TEST_SEED=0 PAIRWISE_DIAGNOSTIC=0 ARMS=spectral_q192,spectral_q256,spectral_q384,spectral_q512,spectral_q768,spectral_q1024 NATIVE_ARMS=spectral_q256,spectral_q512 BOOTSTRAP_REPS=10000
$PY feasibility.py "/cluster/tufts/paralab/tawal01/hybp1024/specdev4/out/specdev4.json"
echo ALL-DONE
