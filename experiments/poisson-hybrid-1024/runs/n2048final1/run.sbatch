#!/bin/bash
#SBATCH -J ctol_hybp2_n2048final1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybp2048/n2048final1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybp2048/n2048final1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybp2048/n2048final1/code/poisson-hybrid-1024"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=a98b1e8d77f27fd76c6bd008d38e6064a591f6fb
echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "commit=a98b1e8d77f27fd76c6bd008d38e6064a591f6fb dirty_hash=e3b0c44298fc cell=n2048final1 namespace=hybp2048"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export PKL=input/auto.pkl
export PARAM_PKL=input/param.pkl
export GROUP_PKL=input/groupfilm.pkl
export POISSON_2048_KIND=final SMOKE=0 NS=2048 FOM_TAUS=1e-6,1e-8,1e-10 N_TEST=16 N_TIME=8 TIME_REPS=12 TIME_WARM=3 BURN_S=3 TEST_SEED=20260826 M=64 MQ=256 GN_ITERS=60 LM_ROM_TAU=0.01 TR_SCALE=1 CG_MAXITER=50000 EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_CAND_OFF=4096 ARMS=lmtrmean_c64_q0,spectral_q1024,spectral_q2048 NATIVE_ARMS= PAIRWISE_DIAGNOSTIC=0 BALANCED_PAIR_ARM=lmtrmean_c64_q0 BALANCED_CONTROL_REPS=10 BOOTSTRAP_REPS=10000
$PY feasibility.py "/cluster/tufts/paralab/tawal01/hybp2048/n2048final1/out/n2048final1.json"
echo ALL-DONE
