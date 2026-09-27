#!/bin/bash
#SBATCH --job-name=opt_opt201
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/opstime_20260922/opt201/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/opstime_20260922/opt201/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/opstime_20260922/opt201
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
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
# Codex design audit #3: the training index is an external absolute path. panel.py adds the
# cohort/training disjointness gate only `if ... exists()`, so a missing file would make the
# gate silently vanish instead of failing. Assert it here.
test -s "/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train/index.json" || { echo "MISSING TRAINING INDEX /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train/index.json"; exit 43; }
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/ops-timing-panel/lib:$TASK_ROOT/experiments/ops-timing-panel"
"$PY" experiments/ops-timing-panel/panel.py \
  --config experiments/ops-timing-panel/config-256-ops.json \
  --checkpoint experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \
  --inputs experiments/ops-timing-panel/inputs \
  --fno-train-index /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train/index.json \
  --out output

echo "OPERATOR PHASE"
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/fno-large.pt" \
  --index output/fno-cohort/index.json \
  --out output --name fno-large --repetitions 5 --burn-in 20 > logs/op-fno-large.log 2>&1
echo "op_exit fno-large=$?"
set -e
tail -3 logs/op-fno-large.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/unet-small.pt" \
  --index output/fno-cohort/index.json \
  --out output --name unet-small --repetitions 5 --burn-in 20 > logs/op-unet-small.log 2>&1
echo "op_exit unet-small=$?"
set -e
tail -3 logs/op-unet-small.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/unet-medium.pt" \
  --index output/fno-cohort/index.json \
  --out output --name unet-medium --repetitions 5 --burn-in 20 > logs/op-unet-medium.log 2>&1
echo "op_exit unet-medium=$?"
set -e
tail -3 logs/op-unet-medium.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/unet-large.pt" \
  --index output/fno-cohort/index.json \
  --out output --name unet-large --repetitions 5 --burn-in 20 > logs/op-unet-large.log 2>&1
echo "op_exit unet-large=$?"
set -e
tail -3 logs/op-unet-large.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/unet-refine.pt" \
  --index output/fno-cohort/index.json \
  --out output --name unet-refine --repetitions 5 --burn-in 20 > logs/op-unet-refine.log 2>&1
echo "op_exit unet-refine=$?"
set -e
tail -3 logs/op-unet-refine.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/tsol-small.pt" \
  --index output/fno-cohort/index.json \
  --out output --name tsol-small --repetitions 5 --burn-in 20 > logs/op-tsol-small.log 2>&1
echo "op_exit tsol-small=$?"
set -e
tail -3 logs/op-tsol-small.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/tsol-medium.pt" \
  --index output/fno-cohort/index.json \
  --out output --name tsol-medium --repetitions 5 --burn-in 20 > logs/op-tsol-medium.log 2>&1
echo "op_exit tsol-medium=$?"
set -e
tail -3 logs/op-tsol-medium.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/tsol-large.pt" \
  --index output/fno-cohort/index.json \
  --out output --name tsol-large --repetitions 5 --burn-in 20 > logs/op-tsol-large.log 2>&1
echo "op_exit tsol-large=$?"
set -e
tail -3 logs/op-tsol-large.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/tsol-refine.pt" \
  --index output/fno-cohort/index.json \
  --out output --name tsol-refine --repetitions 5 --burn-in 20 > logs/op-tsol-refine.log 2>&1
echo "op_exit tsol-refine=$?"
set -e
tail -3 logs/op-tsol-refine.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/don-small.pt" \
  --index output/fno-cohort/index.json \
  --out output --name don-small --repetitions 5 --burn-in 20 > logs/op-don-small.log 2>&1
echo "op_exit don-small=$?"
set -e
tail -3 logs/op-don-small.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/don-medium.pt" \
  --index output/fno-cohort/index.json \
  --out output --name don-medium --repetitions 5 --burn-in 20 > logs/op-don-medium.log 2>&1
echo "op_exit don-medium=$?"
set -e
tail -3 logs/op-don-medium.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/don-refine.pt" \
  --index output/fno-cohort/index.json \
  --out output --name don-refine --repetitions 5 --burn-in 20 > logs/op-don-refine.log 2>&1
echo "op_exit don-refine=$?"
set -e
tail -3 logs/op-don-refine.log || true
set +e
"$PY" experiments/ops-timing-panel/lib/fno_panel.py \
  --checkpoint "$TASK_ROOT/opsckpt/don-large.pt" \
  --index output/fno-cohort/index.json \
  --out output --name don-large --repetitions 5 --burn-in 20 > logs/op-don-large.log 2>&1
echo "op_exit don-large=$?"
set -e
tail -3 logs/op-don-large.log || true

find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
