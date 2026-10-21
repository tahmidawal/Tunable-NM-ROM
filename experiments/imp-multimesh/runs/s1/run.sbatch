#!/bin/bash
#SBATCH --job-name=imm_s1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=01:35:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-multimesh/s1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-multimesh/s1/logs/%j.err
set -uo pipefail
export TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-multimesh/s1
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
finalize() { cd "$TASK_ROOT"; mkdir -p output; echo "{"early_exit": $1, "reason": "$2", "all_passed": false}" > output/EARLY_EXIT.json
  find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256; echo "early_exit=$1 ($2)"; exit $1; }
sha256sum -c MANIFEST.sha256 --quiet || finalize 3 manifest; echo "stage MANIFEST OK"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || finalize 42 gpu-preflight
cd experiments/imp-multimesh
FAIL=0
echo "=== $(date -u +%FT%TZ) preflight_cpu"
if [ $FAIL -ne 0 ]; then echo "skip preflight_cpu (earlier failure)"; else
timeout --signal=TERM --kill-after=30 120 bash "$TASK_ROOT/cmds/00_preflight_cpu.sh"; rc=$?
echo "step_exit[preflight_cpu]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) smoke"
if [ $FAIL -ne 0 ]; then echo "skip smoke (earlier failure)"; else
timeout --signal=TERM --kill-after=30 3600 bash "$TASK_ROOT/cmds/01_smoke.sh"; rc=$?
echo "step_exit[smoke]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) repl"
timeout --signal=TERM --kill-after=30 900 bash "$TASK_ROOT/cmds/02_repl.sh"; rc=$?
echo "step_exit[repl]=$rc"
[ $rc -eq 0 ] || FAIL=1

echo "=== $(date -u +%FT%TZ) repl_score"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/03_repl_score.sh"; rc=$?
echo "step_exit[repl_score]=$rc"
[ $rc -eq 0 ] || FAIL=1

echo "=== $(date -u +%FT%TZ) accept"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/04_accept.sh"; rc=$?
echo "step_exit[accept]=$rc"
[ $rc -eq 0 ] || FAIL=1

cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 || FAIL=1
echo "job_fail=$FAIL"
echo ALL-DONE
exit $FAIL
