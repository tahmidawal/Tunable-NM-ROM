#!/bin/bash
#SBATCH --job-name=imf_POL-e10s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:48:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-fronts/POL-e10s/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-fronts/POL-e10s/logs/%j.err
# imp-fronts POL-e10s: ceiling 0.8 GPU-h on 1 GPU = Slurm wall limit (DESIGN Amendment 4 B8). gpu partition only.
set -euo pipefail
T0=$(date +%s)
export TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-fronts/POL-e10s
WALL_S=2880
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
export IMPF_TESTONCE_DIR=/cluster/tufts/paralab/tawal01/imp-fronts/TESTONCE        # shared, atomic test-once claims (rom/testonce.py)
mkdir -p "$IMPF_TESTONCE_DIR"
echo "STEP fpolicy configs/pole10s.json $(date -u +%FT%TZ)"
# outer timeout at 95 % of the wall (fpolicy's own self-timer stops new solves at 90 %); a kill -> fcollect.py locally
timeout --signal=TERM --kill-after=30 $(( WALL_S * 19 / 20 - ($(date +%s) - T0) ))   "$PY" fpolicy.py --config configs/pole10s.json --out "$TASK_ROOT/output/fpolicy_pole10s" --t0-epoch "$T0"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
