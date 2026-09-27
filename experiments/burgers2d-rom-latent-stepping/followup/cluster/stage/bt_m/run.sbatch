#!/bin/bash
#SBATCH -J bt_m
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/blat2/bt_m/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/blat2/bt_m/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/blat2/bt_m/code"
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
export MODE=k N=64 K_LAT=8 N_TEST=16 PKLS=in/blat_ad_N64_K8.pkl POD_KS= TIME_REPS=7 VARIANTS=lspg:full:weak64,lspg:eq64:weak64,lspg:eq128:weak64,lspg:eq256:weak64,lspg:eq512:weak64,lspg:eq1024:weak64,lspg:full:weakc64,lspg:eqoff64:weakc64,lspg:eqoff128:weakc64,lspg:eqoff256:weakc64,lspg:eqoff512:weakc64,lspg:eqoff1024:weakc64,lspg:full:weak16,lspg:eq64:weak16,lspg:full:weak32,lspg:eq128:weak32,lspg:full:weak64,lspg:eq256:weak64,lspg:full:weak128,lspg:eq512:weak128,lspg:full:weak256,lspg:eq1024:weak256
$PY -u followup/fu_timing.py ../out/timing_m.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
