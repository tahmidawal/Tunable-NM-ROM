#!/bin/bash
#SBATCH --job-name=bpn_bpn101
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b_panel_20260917/bpn101/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b_panel_20260917/bpn101/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b_panel_20260917/bpn101
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-panel:$TASK_ROOT/experiments/b-panel/speed"
"$PY" experiments/b-panel/panel.py \
  --config experiments/b-panel/config-256.json \
  --checkpoint experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \
  --inputs experiments/b-panel/inputs \
  --fno-train-index /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train/index.json \
  --out output

echo "FNO PHASE"
export PYTHONPATH="$TASK_ROOT/fnocode:$TASK_ROOT/experiments/b-ladder-top"
set +e
"$PY" experiments/b-ladder-top/fno_panel.py \
  --checkpoint "$TASK_ROOT/fnockpt/best.pt" \
  --index output/fno-cohort/index.json \
  --out output --name fno-large --repetitions 3 --burn-in 3 > logs/fno.log 2>&1
echo "fno_exit=$?"
set -e
tail -20 logs/fno.log || true

find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
