#!/bin/bash
#SBATCH --job-name=jw_j3b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=420G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcpwide/j3b/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcpwide/j3b/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcpwide/j3b
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
TIMECMD=""; [ -x /usr/bin/time ] && TIMECMD="/usr/bin/time -v"
$TIMECMD "$PY" -u experiments/jcp-wide-bank/train3d/train2w.py --config experiments/jcp-wide-bank/train3d/configs/train_r1024.json --out "$TASK_ROOT/output" > "$TASK_ROOT/output/train.out" 2> "$TASK_ROOT/output/train.err" &
TPID=$!
( set +e; while kill -0 $TPID 2>/dev/null; do C=$(pgrep -P $TPID | head -1); if [ -n "$C" ]; then echo "RSS_SAMPLE $(date -Is) $(ps -o rss= -p $C) kB | $(tail -n 1 "$TASK_ROOT/output/train.out" | cut -c1-200)" >> "$TASK_ROOT/output/rss.log"; fi; sleep 30; done ) &
SPID=$!
set +e; wait $TPID; RC=$?; set -e
kill $SPID 2>/dev/null || true
cat "$TASK_ROOT/output/train.out"; cat "$TASK_ROOT/output/train.err" >&2
[ $RC -eq 0 ] || exit $RC
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
