#!/bin/bash
#SBATCH --job-name=lv_lvg01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=6:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b_lowvisc_20260917/lvg01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b_lowvisc_20260917/lvg01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b_lowvisc_20260917/lvg01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT attempt=lvg01 started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
LANE_PATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/b-lowvisc"

# ---------------------------------------------------------------- the gate job ----
# Both viscosity families, one allocation, one GPU, no trained model (DESIGN.md section 5).
echo "GATE $(date -Is)"
PYTHONPATH="$LANE_PATH" "$PY" experiments/b-lowvisc/gate.py \
  --config experiments/b-lowvisc/config-gate.json --out output/gate

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
