#!/bin/bash
#SBATCH -J ctol_fom_p
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/ctol/ctol_fom_p/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/ctol/ctol_fom_p/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/ctol/ctol_fom_p/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export CTOL_COMMIT=68ac248
export CTOL_SRC_COMMITS="ctol=68ac248/4819beccd106,2026-08-16-poisson2d-rom-objective=d68d8cb/e3b0c442,2026-08-16-burgers2d-rom-latent-stepping=c7e9c2e/e3b0c442,2026-08-14-multistage-precision=b389467/e3b0c442,2026-08-14-burgers2d-coord-rom=b327915/e3b0c442"
echo "host=$(hostname)  node=$SLURMD_NODENAME  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=68ac248  dirty=4819beccd106 (dirty marker 'e3b0c442' == clean)  cell=ctol_fom_p  job=$SLURM_JOB_ID"
echo "src_bundle=ctol=68ac248/4819beccd106,2026-08-16-poisson2d-rom-objective=d68d8cb/e3b0c442,2026-08-16-burgers2d-rom-latent-stepping=c7e9c2e/e3b0c442,2026-08-14-multistage-precision=b389467/e3b0c442,2026-08-14-burgers2d-coord-rom=b327915/e3b0c442"
cp "/cluster/tufts/paralab/tawal01/ctol/ctol_fom_p/MANIFEST.sha256" "/cluster/tufts/paralab/tawal01/ctol/ctol_fom_p/out/MANIFEST.sha256" 2>/dev/null || true
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export KS=2,4,6,8,12,16,24,32 TAUS=1e-1,1e-2,1e-3 M=64 MQ=256 M_BIG=256 K_BIG=32 MQ_4M=1024 DO_SUPP=1 POOL_CONTROL=1 N_TEST=16 GN_ITERS=60 TIME_REPS=7 TIME_WARM=2 PKL_DIR=../ckpt NS=32,64,128,256,512 FOM_ONLY=1 DO_SUPP=0 POOL_CONTROL=0 CAP_CONTROL=0 DO_CEILING=0
$PY -u ctol_poisson.py ../out/ctol_poisson_fomladder.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
