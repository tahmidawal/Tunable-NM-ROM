#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?reviewed batch allocation required}"
JOB_DIR=$(pwd -P)
[ "$JOB_DIR" = /cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/fom-calibration02 ] || exit 64
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" PIP_CACHE_DIR="$JOB_DIR/cache/pip" MPLCONFIGDIR="$JOB_DIR/cache/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" results
JOB_PHASE=source_verification
finish() {
  task_status=$?
  trap - EXIT
  set +e
  "$PY" - "$task_status" "$JOB_PHASE" <<'PY'
import json,sys
from pathlib import Path
p=Path('results/calibration/WORKLOAD-COMPLETENESS.json')
r=json.loads(p.read_text()) if p.exists() else {}
Path('results/batch_summary.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'last_stage':sys.argv[2],
'calibration_completed':r.get('complete_planned_workload') is True,'physical_admission':False,'training_released':False},indent=2)+'\n')
PY
  exit "$task_status"
}
trap finish EXIT
trap 'JOB_PHASE=allocation_deadline; exit 143' TERM
export ROM_ALLOCATION_START_UNIX_SECONDS=$("$PY" -c "import time; print(time.time())")
"$PY" -c "import shutil; assert shutil.disk_usage('.').free >= 20*2**30, 'ten-GiB reserve plus ten-GiB workspace/archive envelope'"
sha256sum -c SOURCE-SHA256SUMS
test -s SOURCE_COMMIT.txt
cat SOURCE_COMMIT.txt
"$PY" calibration_launch2d.py --plan CALIBRATION-PLAN.json --release CALIBRATION-RELEASE.json --inventory RUNTIME-INVENTORY.json --verify-only
JOB_PHASE=gpu_preflight
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); print('jax_devices='+str(jax.devices())); print('jax_x64='+str(jax.config.x64_enabled)); print('matmul_precision='+str(jax.config.jax_default_matmul_precision)); sys.exit(0 if b=='gpu' and jax.config.x64_enabled and str(jax.config.jax_default_matmul_precision)=='highest' else 42)"
lscpu
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
JOB_PHASE=dependencies
"$PY" dependencies.py --wheel-dir wheels --install --target "$JOB_DIR/deps"
DEPS_SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base':'$JOB_DIR/deps','platbase':'$JOB_DIR/deps'}))")
export PYTHONPATH="$DEPS_SITE${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$JOB_DIR/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
JOB_PHASE=dependency_import
"$PY" - <<'PY'
import json,pathlib,ngsolve,netgen,numpy,scipy,jax
prefix=pathlib.Path('deps').resolve()
for module in (ngsolve,netgen):
 assert pathlib.Path(module.__file__).resolve().is_relative_to(prefix)
r={'ngsolve':ngsolve.__version__,'netgen_path':netgen.__file__,'numpy':numpy.__version__,'scipy':scipy.__version__,'jax':jax.__version__,
'jax_backend':jax.default_backend(),'jax_x64':jax.config.x64_enabled,'matmul_precision':str(jax.config.jax_default_matmul_precision)}
assert r['jax_backend']=='gpu' and r['jax_x64'] and r['matmul_precision']=='highest'
pathlib.Path('results/dependency_runtime.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True)
PY
JOB_PHASE=common_target_screening
"$PY" calibration_launch2d.py --plan CALIBRATION-PLAN.json --release CALIBRATION-RELEASE.json --inventory RUNTIME-INVENTORY.json
JOB_PHASE=completed
