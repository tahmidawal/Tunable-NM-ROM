#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
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
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$B10_COMMIT cell=$B10_CELL"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax,sys
b=jax.default_backend(); print(f"jax_backend={b}",jax.devices()[0],flush=True)
sys.exit(0 if b=="gpu" else 42)
PRE
$PY b10_phase6_d.py \
  --output-json ../out/phase6_d.json --output-npz ../out/phase6_d.npz \
  --p5-json deps/p5/phase5_d.json --p5-npz deps/p5/phase5_d.npz \
  --p5-audit deps/p5/AUDIT.json --p5-manifest deps/p5/MANIFEST.sha256 \
  --prereg deps/phase6/PHASE-6-PRE-REGISTRATION.md
$PY b10_audit_phase6_d.py ../out/phase6_d.json ../out/phase6_d.npz ../out/AUDIT.json \
  --expected-commit "$B10_COMMIT" --expected-job "$SLURM_JOB_ID" \
  --manifest ../MANIFEST.sha256 \
  --p5-json deps/p5/phase5_d.json --p5-npz deps/p5/phase5_d.npz \
  --p5-audit deps/p5/AUDIT.json --p5-manifest deps/p5/MANIFEST.sha256 \
  --prereg deps/phase6/PHASE-6-PRE-REGISTRATION.md
cd ..
find out -type f -exec sha256sum {} \; | sort > RESULTS.sha256
echo ALL-DONE
