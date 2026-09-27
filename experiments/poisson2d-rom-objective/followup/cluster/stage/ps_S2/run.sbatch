#!/bin/bash
#SBATCH -J ps_S2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/pobj2/ps_S2/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/pobj2/ps_S2/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/pobj2/ps_S2"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit=5547a96  dirty=e3b0c44298fc  cell=ps_S2"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
cd code
K_LAT=8 HARD_BC=1 TRAIN_SEED=2 $PY -u followup/fu_train.py ../out && PKL=../out/autodec_K8_N64_hbc_S2_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=weak_a1_M64 MS=256 SCHEMES=full,nnls,nnlsoff INITS=nearest,mean EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1 $PY -u pro_colloc.py ../out/rom_S2.json && PKL=../out/autodec_K8_N64_hbc_S2_stages.pkl NS=1 GN_ITERS=60 OBJECTIVES=fd MS=256 SCHEMES=full INITS=nearest $PY -u pro_colloc.py ../out/fd_S2.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
