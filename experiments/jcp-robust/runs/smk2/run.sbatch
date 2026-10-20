#!/bin/bash
#SBATCH --job-name=jr_smk2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcp-robust/smk2/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcp-robust/smk2/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcp-robust/smk2
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export PY
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet

export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
for i in $(seq 0 $((1 - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b} gpu_index=$i',flush=True); sys.exit(0 if b=='gpu' else 42)"
done
cd experiments/jcp-robust
PIDS=()
( export CUDA_VISIBLE_DEVICES=0
set -euo pipefail
echo "STEP r2acc configs/smk2r.json $(date -u +%FT%TZ)"
"$PY" r2acc.py --config configs/smk2r.json --out "$TASK_ROOT/output/r2acc_smk2r"
echo "STEP sh train2d/run_seed.sh TRAIN2D_SMOKE=1 1 $TASK_ROOT/output/train2d_s1 $(date -u +%FT%TZ)"
env TRAIN2D_SMOKE=1 bash train2d/run_seed.sh 1 $TASK_ROOT/output/train2d_s1
echo "STEP r2acc configs/smk2t.json $(date -u +%FT%TZ)"
"$PY" r2acc.py --config configs/smk2t.json --out "$TASK_ROOT/output/r2acc_smk2t"
echo "STEP r2acc configs/smk2u.json $(date -u +%FT%TZ)"
"$PY" r2acc.py --config configs/smk2u.json --out "$TASK_ROOT/output/r2acc_smk2u"
) > "$TASK_ROOT/logs/group0.log" 2>&1 &
PIDS+=($!)
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
cat "$TASK_ROOT"/logs/group*.log
[ "$RC" = 0 ] || { echo "A GROUP FAILED"; exit 1; }
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
