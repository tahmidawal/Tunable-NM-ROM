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
[[ "$B10_CELL" == p12_g2_s11_r1 ]]
test "$B10_REMOTE" = "/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$B10_CELL"
cd "$B10_REMOTE"
./verify_manifest_file_set.sh . MANIFEST.sha256 "./logs/$SLURM_JOB_ID.out" "./logs/$SLURM_JOB_ID.err"
sha256sum -c MANIFEST.sha256
cd code
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_TIMELIMIT_SECONDS=57600 B10_P11_DIR="$PWD/deps/p11"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$B10_COMMIT cell=$B10_CELL phase=Phase12"
"$PY" - <<'PRE'
import jax,sys
b=jax.default_backend(); print(f"jax_backend={b}",jax.devices()[0],flush=True); sys.exit(0 if b=="gpu" else 42)
PRE
common=(
  --p4-json deps/p4/phase4_d.json --p4-npz deps/p4/phase4_d.npz --p4-audit deps/p4/AUDIT.json --p4-manifest deps/p4/MANIFEST.sha256
  --p5-json deps/p5/phase5_d.json --p5-npz deps/p5/phase5_d.npz --p5-audit deps/p5/AUDIT.json --p5-manifest deps/p5/MANIFEST.sha256
  --p6-json deps/p6/phase6_d.json --p6-npz deps/p6/phase6_d.npz --p6-audit deps/p6/AUDIT.json --p6-manifest deps/p6/MANIFEST.sha256
  --p7-json deps/p7/phase7_train.json --p7-npz deps/p7/phase7_train.npz --p7-checkpoint deps/p7/checkpoint.pkl --p7-audit deps/p7/AUDIT.json --p7-manifest deps/p7/MANIFEST.sha256
  --p8-json deps/p8/phase8_d.json --p8-npz deps/p8/phase8_d.npz --p8-audit deps/p8/AUDIT.json --p8-manifest deps/p8/MANIFEST.sha256
  --p9-json deps/p9/phase9_terminal_recovery.json --p9-npz deps/p9/phase9_terminal_recovery.npz --p9-checkpoint deps/p9/checkpoint.pkl
  --p9-audit deps/p9/AUDIT.json --p9-audit-work deps/p9/AUDIT-WORK.npz --p9-work-checkpoint deps/p9/work_checkpoint.pkl --p9-manifest deps/p9/MANIFEST.sha256
  --p10-json deps/p10/phase10_d.json --p10-npz deps/p10/phase10_d.npz --p10-work-checkpoint deps/p10/work_checkpoint.pkl
  --p10-audit deps/p10/AUDIT.json --p10-manifest deps/p10/MANIFEST.sha256 --p10-audit-manifest deps/p10/AUDIT-MANIFEST.sha256
)
"$PY" b10_phase12_g2.py "${common[@]}" --target-dir deps/p5/targets \
  --prereg PHASE-12-PRE-REGISTRATION.md --output-json ../out/phase12_g2.json \
  --output-npz ../out/phase12_g2.npz --checkpoint ../out/checkpoint.pkl \
  --progress-json ../out/PROGRESS.json --work-checkpoint ../out/work_checkpoint.pkl
"$PY" b10_audit_phase12_g2.py --source-json ../out/phase12_g2.json \
  --source-npz ../out/phase12_g2.npz --checkpoint ../out/checkpoint.pkl \
  --manifest ../MANIFEST.sha256 --prereg PHASE-12-PRE-REGISTRATION.md \
  --p5-json deps/p5/phase5_d.json --target-dir deps/p5/targets \
  --expected-commit "$B10_COMMIT" --expected-job "$SLURM_JOB_ID" \
  --slurm-out "../logs/$SLURM_JOB_ID.out" --slurm-err "../logs/$SLURM_JOB_ID.err" \
  --output ../out/AUDIT.json
echo ALL-DONE
