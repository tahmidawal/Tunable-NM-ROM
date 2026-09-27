#!/bin/bash
# One Burgers 3D cell in ONE allocation: stage 1 trains the operator arms (train_b3d.py), stage 2 runs the
# same-job panel (panel_b3d.py). Usage: sbatch cell.sbatch <mesh> <arms> <ntrain> <nval> <jax_fraction_panel>
# [extra ops dirs, comma-separated, or -] [cell|train|panel]  (train = stage 1 only, panel = stage 2 only)
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
set -uo pipefail
MESH="$1"; ARMS="$2"; NTRAIN="$3"; NVAL="$4"; JFRAC="$5"; EXTRA_OPS="${6:-}"; MODE="${7:-cell}"
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT/code"
sha256sum -c --quiet ../MANIFEST.sha256 || { echo "manifest mismatch"; exit 2; }
echo "host=$(hostname) job=$SLURM_JOB_ID mesh=$MESH arms=$ARMS ntrain=$NTRAIN nval=$NVAL jfrac=$JFRAC"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
export SOURCE_COMMIT="$(cat ../COMMIT.txt)"
if [[ "$MODE" != "panel" && ! -f ../output/train/TRAIN_DONE ]]; then
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" train_b3d.py --mesh "$MESH" --arms "$ARMS" --ntrain "$NTRAIN" --nval "$NVAL" --out ../output/train
  TRC=$?
  echo "TRAIN_EXIT=$TRC"
  echo "$TRC" > ../output/train/TRAIN_DONE
fi
if [[ "$MODE" == "train" ]]; then cd "$TASK_ROOT"; find output -type f ! -name '*.pt' -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; echo ALL-DONE; exit 0; fi
OPSDIRS=../output/train; [[ -n "$EXTRA_OPS" && "$EXTRA_OPS" != "-" ]] && OPSDIRS="../output/train,$EXTRA_OPS"
echo "panel sources:"; sha256sum panel_b3d.py opsb3d.py ops3d.py lanes/burgers3d-retry/tables.py lanes/burgers3d-span/common.py
XLA_PYTHON_CLIENT_MEM_FRACTION="$JFRAC" "$PY" panel_b3d.py --config "configs/panel_n$MESH.json" \
  --model lanes/burgers3d-retry/inputs/model_M2 --ops "$OPSDIRS" --out ../output/panel
PRC=$?
echo "PANEL_EXIT=$PRC"
cd "$TASK_ROOT"
find output -type f ! -name '*.pt' ! -path '*/fields/*' -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo ALL-DONE
exit "$PRC"
