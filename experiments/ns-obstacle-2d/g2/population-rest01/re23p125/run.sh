#!/usr/bin/env bash
#SBATCH --job-name=ctol_nsobs2d_population-re23p125-01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export NSOBS_ALLOCATION_STARTED=$(date +%s)
: "${SLURM_JOB_ID:?must be submitted as reviewed batch job}"
JOB_DIR=$(pwd -P)
case "$JOB_DIR" in /cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/*) ;; *) exit 64;; esac
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache"
export MPLCONFIGDIR="$JOB_DIR/cache/matplotlib" PIP_CACHE_DIR="$JOB_DIR/cache/pip"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$JOB_DIR/results"
JOB_PHASE=source_verification
finish() {
  task_status=$?
  trap - EXIT
  set +e
  "$PY" g2/population-rest01/re23p125/summary.py results --exit-code "$task_status" --failed-stage "$JOB_PHASE"
  summary_status=$?
  if [ "$task_status" -eq 0 ] && [ "$summary_status" -ne 0 ]; then task_status=$summary_status; fi
  exit "$task_status"
}
trap finish EXIT
run_stage() {
  JOB_PHASE=$1
  shift
  set +e
  "$@"
  stage_status=$?
  set -e
  printf '{"stage":"%s","exit_code":%s}\n' "$JOB_PHASE" "$stage_status" >> results/stage_exit_codes.jsonl
  return "$stage_status"
}
run_stage source_verification sha256sum -c SOURCE-SHA256SUMS
run_stage source_commit_present test -s SOURCE_COMMIT.txt
cat SOURCE_COMMIT.txt
run_stage gpu_preflight "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); print('jax_version='+jax.__version__); print('jax_devices='+str(jax.devices())); print('jax_x64='+str(jax.config.x64_enabled)); sys.exit(0 if b=='gpu' and jax.config.x64_enabled else 42)"
printf 'cpu_model='
lscpu | sed -n 's/^Model name:[[:space:]]*//p'
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
run_stage dependencies "$PY" dependencies.py --wheel-dir wheels --install --target "$JOB_DIR/deps"
DEPS_SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base':'$JOB_DIR/deps','platbase':'$JOB_DIR/deps'}))")
export PYTHONPATH="$DEPS_SITE${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$JOB_DIR/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_stage dependency_import "$PY" - <<'PY'
import json, pathlib, sys
import ngsolve, netgen, numpy, jax
prefix=pathlib.Path('deps').resolve()
for module in [ngsolve,netgen]:
    assert pathlib.Path(module.__file__).resolve().is_relative_to(prefix), module.__file__
report={'ngsolve':ngsolve.__version__,'ngsolve_path':ngsolve.__file__,
        'netgen_path':netgen.__file__,'numpy':numpy.__version__,'jax':jax.__version__,
        'jax_backend':jax.default_backend(),'jax_x64':jax.config.x64_enabled,
        'matmul_precision':jax.config.jax_default_matmul_precision}
pathlib.Path('results/dependency_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
sys.exit(0 if report['jax_backend']=='gpu' and report['jax_x64']
         and report['matmul_precision']=='highest' else 42)
PY
run_stage immutable_input_metadata "$PY" g2/validate_population_inputs.py --plan g2/population01/PLAN.json --group re23p125 --out results/immutable_input_preflight.json
run_stage api_and_inverse_preflight "$PY" preflight.py --out results/preflight.json
run_stage population "$PY" g2/population_run.py --plan g2/population01/PLAN.json --group re23p125 --release g2/population-rest01/re23p125/RUNTIME-BOUND.json
JOB_PHASE=completed
