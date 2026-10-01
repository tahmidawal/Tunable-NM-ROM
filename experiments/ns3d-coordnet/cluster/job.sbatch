#!/bin/bash
# Usage (from submit.sh): sbatch ... job.sbatch <mode: bank|rom|test> <config[,config...]> [frozen-dir]
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

# run one mesh: driver, then (if it wrote a summary) the independent audit; field files
# are deleted only when the audit passed, otherwise kept as evidence
one_mesh() {  # <config> <outdir> [frozen]
  local C="$1" O="$2" FZ="${3:-}" R1 V1
  local EX=(); [[ -n "$FZ" ]] && EX=(--frozen "$FZ")
  set +e
  "$PY" $D/rom_mesh.py --config "$C" --out "$O" "${EX[@]}"; R1=$?
  V1=-1
  if [[ -f "$O/summary.json" && ( "$R1" -eq 0 || "$R1" -eq 3 ) ]]; then
    "$PY" $D/verify_coordnet.py --out "$O"; V1=$?
  fi
  set -e
  find "$O" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$O.OUTPUTS.sha256"
  if [[ "$V1" -eq 0 ]]; then rm -f "$O"/fields_*.npy "$O"/dev_truth.npy; fi
  echo "MESH=$O RUN_EXIT=$R1 VERIFY_EXIT=$V1"
  [[ "$R1" -eq 0 && "$V1" -eq 0 ]]
}

RC=0
if [[ "$MODE" == "bank" ]]; then
  set +e; "$PY" $D/train_bank.py --config "$CONFIG" --out output; RC=$?; set -e
  echo "RUN_EXIT=$RC VERIFY_EXIT=0"
elif [[ "$MODE" == "rom" ]]; then
  one_mesh "$CONFIG" output/mesh || RC=1
  echo "ALL_EXIT=$RC"
else
  for C in ${CONFIG//,/ }; do
    one_mesh "$C" "output/$(basename "$C" .json)" "$FROZEN" || RC=1
  done
  echo "ALL_EXIT=$RC"
fi
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
exit "$RC"
