#!/bin/bash
#SBATCH -J bt_n
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 8:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/blat2/bt_n/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/blat2/bt_n/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/blat2/bt_n/code"
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
export MODE=n N=64 K_LAT=8 N_TEST=16 PKL=in/blat_ad_N64_K8.pkl NS=32,64,128,256 TIME_REPS=7 VARIANTS=lspg:eq256:weak64,lspg:eq512:weak64,lspg:full:weak64,lspg:eqoff512:weakc64
$PY -u followup/fu_timing.py ../out/timing_n.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
