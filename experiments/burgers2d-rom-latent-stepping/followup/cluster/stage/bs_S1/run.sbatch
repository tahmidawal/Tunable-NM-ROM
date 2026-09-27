#!/bin/bash
#SBATCH -J bs_S1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/blat2/bs_S1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/blat2/bs_S1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/blat2/bs_S1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=bc77aa8804f62c2d7573a7a87c689a0d51cbb7df"
echo "git_dirty=e3b0c44298fc (dirty marker: 'e3b0c442' when clean)"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export N=64 AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30 K_LAT=8 TRAIN_SEED=1 POD_KS=8,16,32,64 POD_VARIANTS=lspg:full:fd VARIANTS=lspg:full:fd,lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64
$PY -u blat_train_ad.py ../out && $PY -u blat_rom.py ../out/blat_ad_N64_K8_S1.pkl ../out
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
