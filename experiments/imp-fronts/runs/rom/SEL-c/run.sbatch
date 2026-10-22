#!/bin/bash
#SBATCH --job-name=imf_SEL-c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:24:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/SEL-c/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/SEL-c/logs/%j.err
# imp-fronts SEL-c: ceiling 0.4 GPU-h on 1 GPU = Slurm wall limit (DESIGN Amendment 4 B8). gpu partition only.
set -euo pipefail
T0=$(date +%s)
export TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-fronts/SEL-c
WALL_S=1440
STOP_S=$(( WALL_S * 9 / 10 ))                    # self-timer: no new work after 90 % of the wall
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda" JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/cache/jax"   # every write in the job dir
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR" "$TASK_ROOT/output" "$TASK_ROOT/logs"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) job=$SLURM_JOB_ID source_commit=$SOURCE_COMMIT precision=$JAX_DEFAULT_MATMUL_PRECISION"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b == 'gpu' else 42)"
source experiments/imp-fronts/rom/env.sh
cd experiments/imp-fronts/rom
left() { echo $(( STOP_S - ($(date +%s) - T0) )); }
L=$(left); [ "$L" -gt 60 ] || { echo "no time left"; exit 1; }
echo "STEP r2acc configs/e2sc.json timeout=${L}s $(date -u +%FT%TZ)"
set +e
timeout --signal=TERM "$L" "$PY" archrun.py r2acc.py --config configs/e2sc.json --out "$TASK_ROOT/output/r2acc_e2sc"
RC=$?
set -e
[ "$RC" = 124 ] && echo "SELF-TIMER: r2acc stopped at 90 % of the wall; result.json incomplete -> no certified rule (m* FAIL)"
[ "$RC" = 0 ] || [ "$RC" = 124 ] || { echo "r2acc failed rc=$RC"; exit "$RC"; }
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
