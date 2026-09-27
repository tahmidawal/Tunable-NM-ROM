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
[[ "$B10_CELL" == p9_terminal_recovery_audit_r1 ]]
test "$B10_REMOTE" = "/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$B10_CELL"
cd "$B10_REMOTE"
./verify_manifest_file_set.sh . MANIFEST.sha256 "./logs/$SLURM_JOB_ID.out" "./logs/$SLURM_JOB_ID.err"
sha256sum -c MANIFEST.sha256
cd code
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$B10_COMMIT cell=$B10_CELL audit_only=true optimizer_updates=0"
"$PY" - <<'PRE'
import jax,sys
b=jax.default_backend(); print(f"jax_backend={b}",jax.devices()[0],flush=True); sys.exit(0 if b=="gpu" else 42)
PRE
"$PY" b10_audit_phase9_terminal_retraction.py \
  --source-json deps/recovery/out/phase9_terminal_recovery.json \
  --source-npz deps/recovery/out/phase9_terminal_recovery.npz \
  --checkpoint deps/recovery/out/terminal_checkpoint.pkl \
  --prior-failed-audit deps/recovery/out/AUDIT.json --recovery-dir deps/recovery \
  --failed-t1-dir deps/failed --manifest ../MANIFEST.sha256 \
  --prereg PHASE-9-TERMINAL-RECOVERY-PRE-REGISTRATION.md \
  --p5-json deps/p5/phase5_d.json --p7-npz deps/p7/phase7_train.npz \
  --target-dir deps/p5/targets --expected-commit "$B10_COMMIT" \
  --expected-job "$SLURM_JOB_ID" --slurm-out "../logs/$SLURM_JOB_ID.out" \
  --slurm-err "../logs/$SLURM_JOB_ID.err" --output ../out/AUDIT.json \
  --audit-work-npz ../out/AUDIT-WORK.npz
echo ALL-DONE
