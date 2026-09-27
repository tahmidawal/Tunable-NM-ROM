#!/bin/bash
#SBATCH -J ctol_hybp_specdev2
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybp1024/specdev2/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybp1024/specdev2/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybp1024/specdev2/code/poisson-hybrid-1024"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=2e4ac0d56b2e49c6094afba60a0c524091d24d95
echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=2e4ac0d56b2e49c6094afba60a0c524091d24d95 dirty_hash=e3b0c44298fc cell=specdev2"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export PKL=input/auto.pkl
export PARAM_PKL=input/param.pkl
export GROUP_PKL=input/groupfilm.pkl

export SMOKE=0 NS=256,512,1024 FOM_TAUS=1e-6 N_TEST=16 N_TIME=6 TIME_REPS=7 TIME_WARM=2 BURN_S=3 TEST_SEED=0 PAIRWISE_DIAGNOSTIC=0 ARMS=spectral_q32,spectral_q64,spectral_q96,spectral_q128 NATIVE_ARMS=spectral_q64,spectral_q128 BOOTSTRAP_REPS=10000
$PY feasibility.py "/cluster/tufts/paralab/tawal01/hybp1024/specdev2/out/specdev2.json"
echo ALL-DONE
