#!/bin/bash
#SBATCH -J bt_k
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/blat2/bt_k/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/blat2/bt_k/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/blat2/bt_k/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=3561b8fd551664bbfd2931cf5b68ce1e8843f70e"
echo "git_dirty=4d224625b01f (dirty marker: 'e3b0c442' when clean)"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export MODE=k N=64 K_LAT=8 N_TEST=16 PKLS=in/blat_ad_N64_K2.pkl,in/blat_ad_N64_K4.pkl,in/blat_ad_N64_K6.pkl,in/blat_ad_N64_K8.pkl,in/blat_ad_N64_K12.pkl,in/blat_ad_N64_K16.pkl,in/blat_ad_N64_K24.pkl,in/blat_ad_N64_K32.pkl POD_KS=2,4,6,8,12,16,24,32,64 TIME_REPS=7 VARIANTS=lspg:eq256:weak64,lspg:full:weak64 POD_VARIANT=lspg:full:fd
$PY -u followup/fu_timing.py ../out/timing_k.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
