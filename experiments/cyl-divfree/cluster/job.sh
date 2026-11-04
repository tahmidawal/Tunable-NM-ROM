#!/usr/bin/env bash
# Generic lane batch body. Submitted from its own job directory:
#   sbatch -p gpu --gres=gpu:1 [--constraint ...] -t HH:MM:SS --mem 64G -c 16 -J cdf_<tag> \
#          -o slurm-%j.out -e slurm-%j.err job.sh <train|eval> [extra args...]
# Code was scp'd to ./code (experiments/cyl-divfree + experiments/ns-obstacle-rom, verified by CODE-SHA256SUMS).
set -euo pipefail
: "${SLURM_JOB_ID:?batch allocation required}"
NS=/cluster/tufts/paralab/tawal01/cyl-divfree-20261103
JOB_DIR=$(pwd -P); case "$JOB_DIR" in $NS/*) ;; *) echo "job dir outside namespace"; exit 64;; esac
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
OBS=/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027
export ROM_ALLOCATION_START_UNIX_SECONDS=$(date +%s)
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" MPLCONFIGDIR="$JOB_DIR/cache/mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
(cd code && sha256sum -c --quiet CODE-SHA256SUMS) && echo "code checksums OK"
df -h /cluster/tufts/paralab | tail -1
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', jax.__version__, jax.devices()); sys.exit(0 if b=='gpu' else 42)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
MODE=$1; shift
PREP="--prepared $OBS/bank-fit01/inputs/prepared --release $OBS/bank-fit01/inputs/consumer/PREPARATION-CONSUMER-RELEASE.json"
RUN09=$OBS/neural-pod-guided-scanned-coordinate01/results/fit/spatial-bank-raw.npz
case $MODE in
  train) $PY code/experiments/cyl-divfree/train_divfree.py $PREP --run09-bank $RUN09 --out results/train "$@" ;;
  eval)  $PY code/experiments/cyl-divfree/evaluate_divfree.py $PREP \
           --probes $OBS/fine-preparation01/results/preparation/probes.npz \
           --development $OBS/development-materialize02/results/population/development.json \
           --run09-bank $RUN09 --out results/eval "$@" ;;
  *) echo "unknown mode"; exit 2 ;;
esac
echo "job finished mode=$MODE"
