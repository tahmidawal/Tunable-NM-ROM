#!/bin/bash
#SBATCH -J ctol_hybp_dev1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybp1024/dev1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybp1024/dev1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybp1024/dev1/code/poisson-hybrid-1024"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=6346c25a82c0c3d4e680af04fa68653ff16a698b
echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=6346c25a82c0c3d4e680af04fa68653ff16a698b dirty_hash=e3b0c44298fc cell=dev1"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export PKL=input/auto.pkl
export PARAM_PKL=input/param.pkl
export SMOKE=0 NS=256,512,1024 FOM_TAUS=1e-6 N_TEST=16 N_TIME=4 TIME_REPS=5 TIME_WARM=2 BURN_S=3 M=64 MQ=256 GN_ITERS=60 LM_ROM_TAU=0.01 TR_SCALE=1 TEST_SEED=0 ARMS=lmmean_cfull_q0,lmmean_c64_q0,lmtrmean_c64_q0,lmnearest_c64_q0,param1_c64_q0,paramall_c64_q0,param1_c64_q8,spectral_q8,spectral_q16
$PY feasibility.py "/cluster/tufts/paralab/tawal01/hybp1024/dev1/out/dev1.json"
echo ALL-DONE
