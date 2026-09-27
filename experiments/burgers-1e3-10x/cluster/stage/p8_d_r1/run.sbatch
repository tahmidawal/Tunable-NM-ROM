#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
: "${B10_COMMIT:?}" "${B10_CELL:?}" "${B10_REMOTE:?}"
[[ "$B10_CELL" == p8_d_r1 ]]
test "$B10_REMOTE" = "/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$B10_CELL"
cd "$B10_REMOTE"
./verify_manifest_file_set.sh . MANIFEST.sha256
sha256sum -c MANIFEST.sha256
cd code
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$B10_COMMIT cell=$B10_CELL phase=P8-D"
"$PY" - <<'PRE'
import jax,sys
b=jax.default_backend(); print(f"jax_backend={b}",jax.devices()[0],flush=True); sys.exit(0 if b=="gpu" else 42)
PRE
"$PY" b10_phase8_d.py \
  --p4-json deps/p4/phase4_d.json --p4-npz deps/p4/phase4_d.npz \
  --p4-audit deps/p4/AUDIT.json --p4-manifest deps/p4/MANIFEST.sha256 \
  --p5-json deps/p5/phase5_d.json --p5-npz deps/p5/phase5_d.npz \
  --p5-audit deps/p5/AUDIT.json --p5-manifest deps/p5/MANIFEST.sha256 \
  --p6-json deps/p6/phase6_d.json --p6-npz deps/p6/phase6_d.npz \
  --p6-audit deps/p6/AUDIT.json --p6-manifest deps/p6/MANIFEST.sha256 \
  --p7-json deps/p7/phase7_train.json --p7-npz deps/p7/phase7_train.npz \
  --p7-checkpoint deps/p7/checkpoint.pkl --p7-audit deps/p7/AUDIT.json \
  --p7-manifest deps/p7/MANIFEST.sha256 --target-dir deps/p5/targets \
  --prereg PHASE-8-PRE-REGISTRATION.md \
  --output-json ../out/phase8_d.json --output-npz ../out/phase8_d.npz \
  --progress-json ../out/PROGRESS.json --work-checkpoint ../out/work_checkpoint.pkl
echo ALL-DONE
