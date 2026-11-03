#!/usr/bin/env bash
# One G5 reference allocation. Usage (from the staged job directory, via sbatch):
#   sbatch --job-name=ctol_cylu_<tag> --time=HH:MM:SS cluster/run.sh <config1.json> [<config2.json> ...]
# Each config runs transient.py into results/<config stem>/ sequentially.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?must be submitted as a batch job}"
JOB_DIR=$(pwd -P)
case "$JOB_DIR" in /cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/*) ;; *) echo "bad job dir"; exit 64;; esac
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" MPLCONFIGDIR="$JOB_DIR/cache/mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$JOB_DIR/results"
stage() { local name=$1; shift; set +e; "$@"; local s=$?; set -e
  printf '{"stage":"%s","exit_code":%s,"utc":"%s"}\n' "$name" "$s" "$(date -u +%FT%TZ)" >> results/stage_exit_codes.jsonl
  return $s; }
stage source_verification sha256sum -c SOURCE-SHA256SUMS
cat SOURCE_COMMIT.txt
stage gpu_preflight "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); print('jax_x64='+str(jax.config.x64_enabled)); sys.exit(0 if b=='gpu' and jax.config.x64_enabled else 42)"
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
hostname
stage dependencies "$PY" install_deps.py --wheel-dir "$JOB_DIR/wheels" --target "$JOB_DIR/deps"
DEPS_SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base':'$JOB_DIR/deps','platbase':'$JOB_DIR/deps'}))")
export PYTHONPATH="$DEPS_SITE${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$JOB_DIR/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
stage dependency_import "$PY" -c "import ngsolve, netgen, jax, json; print(json.dumps({'ngsolve': ngsolve.__version__, 'jax_backend': jax.default_backend(), 'precision': jax.config.jax_default_matmul_precision}))"
for cfg in "$@"; do
  name=$(basename "$cfg" .json)
  stage "run_$name" "$PY" transient.py --config "$cfg" --out "results/$name" || true
done
echo "all_configs_attempted"
