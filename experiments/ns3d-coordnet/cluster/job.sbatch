#!/bin/bash
# Usage (from submit.sh): sbatch ... job.sbatch <mode: bank|rom|test> <config> [frozen-dir]
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
set -euo pipefail
MODE="$1"; CONFIG="$2"; FROZEN="${3:-}"
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt) job=$SLURM_JOB_ID mode=$MODE config=$CONFIG"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export SOURCE_COMMIT="$(cat COMMIT.txt)"
D=experiments/ns3d-coordnet
set +e
if [[ "$MODE" == "bank" ]]; then
  "$PY" $D/train_bank.py --config "$CONFIG" --out output; RC=$?; VRC=0
elif [[ "$MODE" == "rom" ]]; then
  "$PY" $D/rom_mesh.py --config "$CONFIG" --out output; RC=$?
  VRC=-1
  if [[ "$RC" -eq 0 ]]; then "$PY" $D/verify_coordnet.py --out output; VRC=$?; fi
else
  # test: CONFIG is a comma-separated list, one sub-directory per mesh; each mesh's
  # fields are audited and deleted before the next mesh starts
  RC=0; VRC=0
  for C in ${CONFIG//,/ }; do
    NAME=$(basename "$C" .json)
    "$PY" $D/rom_mesh.py --config "$C" --out "output/$NAME" --frozen "$FROZEN"; R1=$?
    V1=-1
    if [[ "$R1" -eq 0 ]]; then "$PY" $D/verify_coordnet.py --out "output/$NAME"; V1=$?; fi
    find "output/$NAME" -type f -print0 | sort -z | xargs -0 -r sha256sum > "output/$NAME.OUTPUTS.sha256"
    rm -f output/$NAME/fields_*.npy output/$NAME/dev_truth.npy
    echo "MESH=$NAME RUN_EXIT=$R1 VERIFY_EXIT=$V1"
    if [[ "$R1" -ne 0 ]]; then RC=$R1; fi
    if [[ "$V1" -ne 0 ]]; then VRC=$V1; fi
  done
fi
set -e
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
# field files served the in-job independent audit; they are not transferred
rm -f output/fields_*.npy output/dev_truth.npy output/*/fields_*.npy output/*/dev_truth.npy
echo "RUN_EXIT=$RC VERIFY_EXIT=$VRC"
exit "$RC"
