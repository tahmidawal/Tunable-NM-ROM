#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?reviewed batch allocation required}"
JOB_DIR=$(pwd -P)
[ "$JOB_DIR" = /cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-diagnostic05 ] || exit 64
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export ROM_ALLOCATION_START_UNIX_SECONDS=$(/usr/bin/date +%s)
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" CUDA_CACHE_PATH="$JOB_DIR/cache/cuda"
export MPLCONFIGDIR="$JOB_DIR/cache/matplotlib"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" results
JOB_PHASE=source_verification
finish() {
  task_status=$?
  trap - EXIT
  set +e
  "$PY" - "$task_status" "$JOB_PHASE" <<'PY'
import json,sys
from pathlib import Path
p=Path('results/diagnostic/DIAGNOSTIC.json');r=json.loads(p.read_text()) if p.exists() else {}
Path('results/batch_summary.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'last_stage':sys.argv[2],
'diagnostic_completed':r.get('status')=='completed_diagnostic_pending_independent_review',
'physical_admission_created':False,'automatic_retry':False},indent=2)+'\n')
PY
  exit "$task_status"
}
trap finish EXIT
trap 'JOB_PHASE=allocation_deadline; exit 143' TERM
sha256sum -c SOURCE-SHA256SUMS
cat SOURCE_COMMIT.txt
"$PY" -m accuracy_v2.diagnostic05_launch --plan PLAN.json --release DIAGNOSTIC-RELEASE.json --inventory RUNTIME-INVENTORY.json --verify-only
JOB_PHASE=gpu_preflight
"$PY" - <<'PY'
import json,sys,platform,hashlib,jax,jaxlib,optax,numpy,scipy
from pathlib import Path
r={'python':platform.python_version(),'jax':jax.__version__,'jaxlib':jaxlib.__version__,'optax':optax.__version__,
'numpy':numpy.__version__,'scipy':scipy.__version__,'jax_backend':jax.default_backend(),
'jax_x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision),
'python_executable_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
Path('results/runtime.json').write_text(json.dumps(r,indent=2)+'\n')
print('jax_backend='+r['jax_backend'],flush=True);print(json.dumps(r),flush=True)
sys.exit(0 if r['jax_backend']=='gpu' and r['jax_x64'] and r['matmul_precision']=='highest' else 42)
PY
lscpu
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
JOB_PHASE=successor_neural_weak_LS
"$PY" -m accuracy_v2.diagnostic05_launch --plan PLAN.json --release DIAGNOSTIC-RELEASE.json --inventory RUNTIME-INVENTORY.json
JOB_PHASE=completed
