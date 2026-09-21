#!/bin/bash
#SBATCH --job-name=ns3dgrok_diag04
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag04/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag04/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag04
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/ns3d:$TASK_ROOT/experiments/ns2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/ns3d-grok"
export SOURCE_COMMIT="$(cat COMMIT.txt)"
set +e
"$PY" experiments/ns3d-grok/diag04.py --config experiments/ns3d-grok/configs/diag04.json --out output
NS3D_EXIT=$?
set -e
if [[ "$NS3D_EXIT" -eq 0 ]]; then
  "$PY" experiments/ns3d-grok/verify_diag04.py --out output
fi
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo "DIAG_EXIT=$NS3D_EXIT"
exit "$NS3D_EXIT"
