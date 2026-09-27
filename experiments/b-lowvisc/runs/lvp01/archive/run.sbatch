#!/bin/bash
#SBATCH --job-name=lv_lvp01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b_lowvisc_20260917/lvp01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b_lowvisc_20260917/lvp01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b_lowvisc_20260917/lvp01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT attempt=lvp01 started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
LANE_PATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/b-lowvisc"

# bpn301's memory fraction on an 80 GB card; the panel holds every reduced query resident.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-lowvisc:$TASK_ROOT/experiments/b-lowvisc/deps/b-panel-speed"
CKPT="$TASK_ROOT/experiments/b-lowvisc/checkpoints/sep_hfit_lowvisc0.pkl"
echo "checkpoint $(sha256sum "$CKPT")"

# ------------------------------- stage 1: the checkpoint's own correction directions ----
echo "DIRECTIONS $(date -Is)"
"$PY" experiments/b-lowvisc/lv_directions.py \
  --config experiments/b-lowvisc/config-panel.json --checkpoint "$CKPT" --out output/directions

# ---------------------------------------------------------- stage 2: the panel ----
echo "PANEL $(date -Is)"
"$PY" experiments/b-lowvisc/lv_panel.py \
  --config experiments/b-lowvisc/config-panel.json --checkpoint "$CKPT" \
  --inputs output/directions --out output/panel

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
