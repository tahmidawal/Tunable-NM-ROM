#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=96G
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
$PY b10_phase5_d.py \
  --output-json ../out/phase5_d.json --output-npz ../out/phase5_d.npz \
  --target-dir ../out/targets \
  --p4-json deps/p4/phase4_d.json --p4-npz deps/p4/phase4_d.npz \
  --p4-audit deps/p4/AUDIT.json --p4-manifest deps/p4/MANIFEST.sha256 \
  --rank-json deps/phase5/phase5_rank_diagnostic.json \
  --rank-script b10_phase5_rank_diagnostic.py \
  --rank-checkpoint deps/phase5/PHASE-5-RANK-CHECKPOINT.md \
  --prereg deps/phase5/PHASE-5-PRE-REGISTRATION.md
$PY b10_audit_phase5_d.py ../out/phase5_d.json ../out/phase5_d.npz ../out/AUDIT.json \
  --expected-commit "$B10_COMMIT" --expected-job "$SLURM_JOB_ID" \
  --manifest ../MANIFEST.sha256 \
  --p4-json deps/p4/phase4_d.json --p4-npz deps/p4/phase4_d.npz \
  --p4-audit deps/p4/AUDIT.json --p4-manifest deps/p4/MANIFEST.sha256 \
  --rank-json deps/phase5/phase5_rank_diagnostic.json \
  --rank-script b10_phase5_rank_diagnostic.py \
  --rank-checkpoint deps/phase5/PHASE-5-RANK-CHECKPOINT.md \
  --prereg deps/phase5/PHASE-5-PRE-REGISTRATION.md
cd ..
find out -type f -exec sha256sum {} \; | sort > RESULTS.sha256
echo ALL-DONE
