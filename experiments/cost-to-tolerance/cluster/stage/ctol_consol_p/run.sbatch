#!/bin/bash
#SBATCH -J ctol_consol_p
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/ctol/ctol_consol_p/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/ctol/ctol_consol_p/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/ctol/ctol_consol_p/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export CTOL_COMMIT=b61ba89
export CTOL_SRC_COMMITS="ctol=b61ba89/78235ef9e9b8,2026-08-16-poisson2d-rom-objective=d68d8cb/e3b0c442,2026-08-16-burgers2d-rom-latent-stepping=c7e9c2e/e3b0c442,2026-08-14-multistage-precision=b389467/e3b0c442,2026-08-14-burgers2d-coord-rom=b327915/e3b0c442"
echo "host=$(hostname)  node=$SLURMD_NODENAME  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=b61ba89  dirty=78235ef9e9b8 (dirty marker 'e3b0c442' == clean)  cell=ctol_consol_p  job=$SLURM_JOB_ID"
echo "src_bundle=ctol=b61ba89/78235ef9e9b8,2026-08-16-poisson2d-rom-objective=d68d8cb/e3b0c442,2026-08-16-burgers2d-rom-latent-stepping=c7e9c2e/e3b0c442,2026-08-14-multistage-precision=b389467/e3b0c442,2026-08-14-burgers2d-coord-rom=b327915/e3b0c442"
cp "/cluster/tufts/paralab/tawal01/ctol/ctol_consol_p/MANIFEST.sha256" "/cluster/tufts/paralab/tawal01/ctol/ctol_consol_p/out/MANIFEST.sha256" 2>/dev/null || true
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export KS=2,4,6,8,12,16,24,32 TAUS=1e-1,1e-2,1e-3 M=64 MQ=256 M_BIG=256 K_BIG=32 MQ_4M=1024 DO_SUPP=1 POOL_CONTROL=1 N_TEST=16 GN_ITERS=60 TIME_REPS=7 TIME_WARM=2 PKL_DIR=../ckpt NS=32,64,128,256,512 DO_SUPP=0 POOL_CONTROL=0 DO_POD_DIRECT=0 CONFIGS=../consolidate_configs.json ARM_TAG=consolidated
$PY -u ctol_poisson.py ../out/ctol_poisson_consolidated.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
