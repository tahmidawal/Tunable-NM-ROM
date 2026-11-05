#!/bin/bash
# Usage (from submit.sh): sbatch ... job.sbatch <config>
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:TERM@300
set -euo pipefail
CONFIG="$1"
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
on_term() { echo "SIGTERM: time limit near; writing manifest"; find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; echo "RUN_EXIT=TERM"; exit 143; }
trap on_term TERM
echo "host=$(hostname) source_commit=$(cat COMMIT.txt) job=$SLURM_JOB_ID config=$CONFIG"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
echo "JAX_DEFAULT_MATMUL_PRECISION=$JAX_DEFAULT_MATMUL_PRECISION"
export SOURCE_COMMIT="$(cat COMMIT.txt)"
set +e
"$PY" experiments/ns3d-offmesh-quadrature/run_mesh.py --config "$CONFIG" --out output & wait $!; RC=$?
set -e
echo "RUN_EXIT=$RC"
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
exit "$RC"
