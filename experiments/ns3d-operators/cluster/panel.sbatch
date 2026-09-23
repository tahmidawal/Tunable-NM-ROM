#!/bin/bash
# Usage (from submit_panel.sh): sbatch ... panel.sbatch <config>
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
set -euo pipefail
CONFIG="$1"; JFRAC="${2:-0.5}"
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_MEM_FRACTION=$JFRAC
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt) job=$SLURM_JOB_ID config=$CONFIG"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export SOURCE_COMMIT="$(cat COMMIT.txt)"
set +e
"$PY" experiments/ns3d-operators/panel.py --config "experiments/ns3d-operators/$CONFIG" --out output
RC=$?
VRC=-1
if [[ "$RC" -eq 0 ]]; then "$PY" experiments/ns3d-operators/audit_panel.py --out output; VRC=$?; fi
set -e
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo "RUN_EXIT=$RC AUDIT_EXIT=$VRC"
exit "$RC"
