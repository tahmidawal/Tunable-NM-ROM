#!/bin/bash
#SBATCH -J wad_n64_k16
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 20:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/wlat/wad_n64_k16/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/wlat/wad_n64_k16/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/wlat/wad_n64_k16/code"
export PYTHONPATH="/cluster/tufts/paralab/tawal01/wlat/wad_n64_k16/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=ad575708e8cb064d6997d5527a7979673a3479b4"
echo "cell=wad_n64_k16"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export N=64 K_LAT=16 ROM_SUBSTEPS=20 AD_STEPS=80000 AD_BATCH=128 P_SUB=2048 N_TEST=16 GN_BUDGET=30 IC_BUDGET=100 FLOOR_BUDGET=60 EQ_SNAPS=64 EQ_POOL=4096 POD_KS=6,8,16,32,64 VARIANTS=lspg:eq256:weak64,galerkin:eq256:weak64,lspg:eq576:weak144,lspg:eq1024:weak256,galerkin:eq1024:weak256,lspg:eqoff256:weak64,lspg:full:weak64,galerkin:full:weak64,lspg:full:weaku64,lspg:full:weakl64,lspg:full:weak144,lspg:full:weak256,lspg:full:fd,galerkin:full:fd,lspg:rand512:fd,lspg:offgrid512:fd POD_VARIANTS=lspg:full:fd,lspg:full:weak256,lspg:eq1024:weak256
$PY -u wlat_verify.py ../out && $PY -u wlat_train_ad.py ../out && $PY -u wlat_rom.py ../out/wlat_ad_N64_K16.pkl ../out
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
