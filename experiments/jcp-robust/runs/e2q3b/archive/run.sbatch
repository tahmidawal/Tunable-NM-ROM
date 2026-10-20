#!/bin/bash
#SBATCH --job-name=jr_e2q3b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=07:15:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcp-robust/e2q3b/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcp-robust/e2q3b/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcp-robust/e2q3b
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export PY
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
mkdir -p "$(dirname "$TASK_ROOT/experiments/jcp-robust/inputs/banks3d/s3/bank.pkl")" && cp "/cluster/tufts/paralab/tawal01/burgers-archive/jcp-robust/e2t3d3/output/train3d_s3/bank.pkl" "$TASK_ROOT/experiments/jcp-robust/inputs/banks3d/s3/bank.pkl" && echo "8912c8e382f0cf3240fe39d6751c90276f1f7466536e16189655bbf1f18e38e1  $TASK_ROOT/experiments/jcp-robust/inputs/banks3d/s3/bank.pkl" | sha256sum -c --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
ALLOC="${CUDA_VISIBLE_DEVICES:?scheduler did not set CUDA_VISIBLE_DEVICES}"
echo "allocated CUDA_VISIBLE_DEVICES=$ALLOC"
for i in $(seq 0 $((1 - 1))); do
  CUDA_VISIBLE_DEVICES=$(echo "$ALLOC" | cut -d, -f$((i + 1))) "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b} gpu_index=$i',flush=True); sys.exit(0 if b=='gpu' else 42)"
done
cd experiments/jcp-robust
PIDS=()
( export CUDA_VISIBLE_DEVICES=$(echo "$ALLOC" | cut -d, -f1)   # the scheduler's device for group 0
set -euo pipefail
echo "STEP e2run3d configs/e2q3b.json $(date -u +%FT%TZ)"
"$PY" e2run3d.py --config configs/e2q3b.json --out "$TASK_ROOT/output/e2run3d_e2q3b"
) > "$TASK_ROOT/logs/group0.log" 2>&1 &
PIDS+=($!)
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
cat "$TASK_ROOT"/logs/group*.log
[ "$RC" = 0 ] || { echo "A GROUP FAILED"; exit 1; }
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
