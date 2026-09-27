#!/bin/bash
#SBATCH -J wsp_cons
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /cluster/tufts/paralab/tawal01/wsfom/wsp_cons/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/wsfom/wsp_cons/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/wsfom/wsp_cons/code/rom-warmstart-fom"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=63de48bdd5dd7c4a89494a05d0059bf48e27fe5e  dirty=e3b0c44298fc  cell=wsp_cons"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export PKL=in/autodec_K8_N64_hbc_stages.pkl N=64 M=64 MQ=256 EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 INIT=mean N_TEST=16 N_TIME=8 GN_ITERS=60 TIME_REPS=7 TIME_WARM=2 ROM_TAUS=0.5,1e-1,1e-2,1e-3,0 FOM_TAUS=1e-6,1e-8,1e-10 NS=32,64,128,256,512 RUN_ROLE=consolidated
$PY wsf_poisson.py ../../out/wsp_cons.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
