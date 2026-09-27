#!/bin/bash
#SBATCH -J bm_M
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/blat2/bm_M/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/blat2/bm_M/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/blat2/bm_M/code"
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
export N=64 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30 K_LAT=8 POD_KS=8 POD_VARIANTS=lspg:full:fd VARIANTS=lspg:full:weak16,lspg:eq64:weak16,lspg:full:weak32,lspg:eq128:weak32,lspg:full:weak64,lspg:eq256:weak64,lspg:full:weak128,lspg:eq512:weak128,lspg:full:weak256,lspg:eq1024:weak256
$PY -u blat_rom.py in/blat_ad_N64_K8.pkl ../out
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
