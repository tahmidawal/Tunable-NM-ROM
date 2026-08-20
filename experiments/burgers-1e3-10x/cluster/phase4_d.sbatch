#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
: "${B10_COMMIT:?}"; : "${B10_CELL:?}"; : "${B10_REMOTE:?}"
test "$B10_REMOTE" = "/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$B10_CELL"
cd "$B10_REMOTE"
echo "stage_manifest_file_sha256=$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)"
sha256sum -c MANIFEST.sha256
cd code
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export B10_ORACLE_WORKERS=8
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$B10_COMMIT cell=$B10_CELL oracle_workers=$B10_ORACLE_WORKERS"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax,sys
b=jax.default_backend(); print(f"jax_backend={b}",jax.devices()[0],flush=True)
sys.exit(0 if b=="gpu" else 42)
PRE
$PY b10_phase4_d.py \
  --output-json ../out/phase4_d.json --output-npz ../out/phase4_d.npz \
  --s0-json deps/s0/s0.json --s0-npz deps/s0/s0.npz --s0-audit deps/s0/AUDIT.json \
  --p3-json deps/p3/phase3_d.json --p3-npz deps/p3/phase3_d.npz --p3-audit deps/p3/AUDIT.json
echo ALL-DONE
