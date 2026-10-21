#!/bin/bash
#SBATCH --job-name=imm_j5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=02:45:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-multimesh/j5/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-multimesh/j5/logs/%j.err
set -uo pipefail
export TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-multimesh/j5
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 JAX_ENABLE_COMPILATION_CACHE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
unset JCPREF_META
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1 SOURCE_COMMIT=$(cat $TASK_ROOT/COMMIT.txt) OUT="$TASK_ROOT/output"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$OUT"
cd "$TASK_ROOT"
finalize() { cd "$TASK_ROOT"; mkdir -p output; printf '{"early_exit": %s, "reason": "%s", "all_passed": false}
' "$1" "$2" > output/EARLY_EXIT.json
  :
  find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256; echo "early_exit=$1 ($2)"; exit $1; }
sha256sum -c MANIFEST.sha256 --quiet || finalize 3 manifest; echo "stage MANIFEST OK"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || finalize 42 gpu-preflight
cd experiments/imp-multimesh
FAIL=0
echo "=== $(date -u +%FT%TZ) p1acc256"
if [ $FAIL -ne 0 ]; then echo "skip p1acc256 (earlier failure)"; else
timeout --signal=TERM --kill-after=30 1500 bash "$TASK_ROOT/cmds/00_p1acc256.sh"; rc=$?
echo "step_exit[p1acc256]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) oursq256"
if [ $FAIL -ne 0 ]; then echo "skip oursq256 (earlier failure)"; else
timeout --signal=TERM --kill-after=30 600 bash "$TASK_ROOT/cmds/01_oursq256.sh"; rc=$?
echo "step_exit[oursq256]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) p1acc1024"
if [ $FAIL -ne 0 ]; then echo "skip p1acc1024 (earlier failure)"; else
timeout --signal=TERM --kill-after=30 2400 bash "$TASK_ROOT/cmds/02_p1acc1024.sh"; rc=$?
echo "step_exit[p1acc1024]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) oursq1024"
if [ $FAIL -ne 0 ]; then echo "skip oursq1024 (earlier failure)"; else
timeout --signal=TERM --kill-after=30 600 bash "$TASK_ROOT/cmds/03_oursq1024.sh"; rc=$?
echo "step_exit[oursq1024]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) p1acc4096"
if [ $FAIL -ne 0 ]; then echo "skip p1acc4096 (earlier failure)"; else
timeout --signal=TERM --kill-after=30 3600 bash "$TASK_ROOT/cmds/04_p1acc4096.sh"; rc=$?
echo "step_exit[p1acc4096]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) oursq4096"
if [ $FAIL -ne 0 ]; then echo "skip oursq4096 (earlier failure)"; else
timeout --signal=TERM --kill-after=30 600 bash "$TASK_ROOT/cmds/05_oursq4096.sh"; rc=$?
echo "step_exit[oursq4096]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 || FAIL=1
echo "job_fail=$FAIL"
echo ALL-DONE
exit $FAIL
