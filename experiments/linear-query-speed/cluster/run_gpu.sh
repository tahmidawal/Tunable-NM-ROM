#!/bin/bash -l
# One GPU allocation of the linear-query-speed lane: preflight, then one process per mesh.
# Usage: sbatch ... cluster/run_gpu.sh auto "<selection meshes>" "<test meshes>" ["<paper-driver meshes>"]
#        sbatch ... cluster/run_gpu.sh testfrozen <frozen file> "<test meshes>" ["<extra driver arguments>"]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
MODE=$1
cfg_of() { if [ -f "paper_code/config-$1.json" ]; then echo "paper_code/config-$1.json"; else echo "config-$1.json"; fi; }
if [ "$MODE" = auto ]; then
  # auto "<selection meshes>" "<test meshes>" ["<paper-driver meshes>"]
  SEL=$2; TEST=$3; PAPER=${4:-}
  for n in $SEL; do
    echo "=== select mesh $n ==="
    "$PY" lqs_p2d.py --config "$(cfg_of $n)" --intervals "$n" --mode select --out "results/select-p2d-$n" \
      || echo "MESH-FAILED select $n exit=$?"
  done
  # mechanical freeze from development-only outputs; the test cohort is not touched before this line
  "$PY" lqs_select.py --results $(for n in $SEL; do echo "results/select-p2d-$n/result.json"; done) --out results/frozen.json
  sha256sum results/frozen.json
  for n in $TEST; do
    echo "=== test mesh $n ==="
    "$PY" lqs_p2d.py --config "$(cfg_of $n)" --intervals "$n" --mode test --frozen results/frozen.json --out "results/p2d-$n" \
      || echo "MESH-FAILED test $n exit=$?"
  done
  for n in $PAPER; do
    echo "=== paper driver mesh $n ==="
    # the paper's whole driver, unmodified (all arms and the CG reference solver)
    (cd paper_code && "$PY" p2t_solve.py --config "config-$n.json" --out "../results/paper-$n" \
       && "$PY" pbk_audit_np.py "../results/paper-$n" --subsample 16 --delete-fields) || echo "MESH-FAILED paper $n exit=$?"
  done
elif [ "$MODE" = testfrozen ]; then
  # testfrozen <frozen file committed in the lane> "<test meshes>" [extra driver arguments]
  FROZEN=$2; TEST=$3; EXTRA=${4:-}
  sha256sum "$FROZEN"; cp "$FROZEN" results/frozen.json
  for n in $TEST; do
    echo "=== test mesh $n ==="
    "$PY" lqs_p2d.py --config "$(cfg_of $n)" --intervals "$n" --mode test --frozen results/frozen.json --out "results/p2d-$n" $EXTRA \
      || echo "MESH-FAILED test $n exit=$?"
  done
else
  echo "unknown mode $MODE"; exit 65
fi
rm -rf cache tmp
date
echo "task_completed"
