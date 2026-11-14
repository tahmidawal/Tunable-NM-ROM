#!/bin/bash -l
# One GPU allocation of the linear-query-speed lane: preflight, then one process per mesh.
# Usage: sbatch ... cluster/run_gpu.sh <select|test|paper> "<meshes>" [frozen.json]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -eo pipefail
: "${SLURM_JOB_ID:?must be submitted as a batch job}"
NS=/cluster/tufts/paralab/tawal01/linear-query-speed-20261114
JOB_DIR=$(pwd -P)
case "$JOB_DIR" in $NS/jobs/*) ;; *) echo "bad job dir"; exit 64;; esac
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" MPLCONFIGDIR="$JOB_DIR/cache/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JOB_DIR/results"
sha256sum -c --quiet SOURCE-SHA256SUMS && echo "sources verified"
export SOURCE_COMMIT=$(cat SOURCE_COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"; date
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
MODE=$1; MESHES=$2; FROZEN=${3:-}
for n in $MESHES; do
  CFG=paper_code/config-$n.json; [ -f "$CFG" ] || CFG=config-$n.json
  echo "=== mesh $n ($CFG) ==="
  if [ "$MODE" = paper ]; then
    # the paper's whole driver, unmodified (all arms and the CG reference solver)
    (cd paper_code && "$PY" p2t_solve.py --config "config-$n.json" --out "../results/paper-$n" \
       && "$PY" pbk_audit_np.py "../results/paper-$n" --subsample 16 --delete-fields) || echo "MESH-FAILED $n exit=$?"
  else
    "$PY" lqs_p2d.py --config "$CFG" --intervals "$n" --mode "$MODE" ${FROZEN:+--frozen "$FROZEN"} --out "results/p2d-$n" \
      || echo "MESH-FAILED $n exit=$?"
  fi
done
rm -rf cache tmp
date
echo "task_completed"
