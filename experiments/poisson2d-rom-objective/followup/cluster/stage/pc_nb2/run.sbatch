#!/bin/bash
#SBATCH -J pc_nb2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/pobj2/pc_nb2/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/pobj2/pc_nb2/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/pobj2/pc_nb2"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit=d26d27e  dirty=3a08825d9de9  cell=pc_nb2"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
cd code
NB=2 KS=2,4,6,8,12,16,24,32 M=64 MQ=256 N_TEST=16 GN_ITERS=60 POD_KS=2,4,6,8,12,16,24,32,48,64 EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 $PY -u followup/fu_family.py ../out
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
