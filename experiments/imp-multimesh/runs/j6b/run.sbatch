#!/bin/bash
#SBATCH --job-name=imm_j6b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=02:35:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-multimesh/j6b/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-multimesh/j6b/logs/%j.err
set -uo pipefail
export TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-multimesh/j6b
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
echo "=== $(date -u +%FT%TZ) time"
if [ $FAIL -ne 0 ]; then echo "skip time (earlier failure)"; else
timeout --signal=TERM --kill-after=30 5400 bash "$TASK_ROOT/cmds/00_time.sh"; rc=$?
echo "step_exit[time]=$rc"
[ $rc -eq 0 ] || FAIL=1
fi

echo "=== $(date -u +%FT%TZ) compile_ours_R128_256"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/01_compile_ours_R128_256.sh"; rc=$?
echo "step_exit[compile_ours_R128_256]=$rc"

echo "=== $(date -u +%FT%TZ) compile_ours_R384_256"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/02_compile_ours_R384_256.sh"; rc=$?
echo "step_exit[compile_ours_R384_256]=$rc"

echo "=== $(date -u +%FT%TZ) compile_PM-256_r128k4_256"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/03_compile_PM-256_r128k4_256.sh"; rc=$?
echo "step_exit[compile_PM-256_r128k4_256]=$rc"

echo "=== $(date -u +%FT%TZ) compile_PM-256_r384k8_256"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/04_compile_PM-256_r384k8_256.sh"; rc=$?
echo "step_exit[compile_PM-256_r384k8_256]=$rc"

echo "=== $(date -u +%FT%TZ) compile_ours_R128_1024"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/05_compile_ours_R128_1024.sh"; rc=$?
echo "step_exit[compile_ours_R128_1024]=$rc"

echo "=== $(date -u +%FT%TZ) compile_ours_R384_1024"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/06_compile_ours_R384_1024.sh"; rc=$?
echo "step_exit[compile_ours_R384_1024]=$rc"

echo "=== $(date -u +%FT%TZ) compile_PM-1024_r128k4_1024"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/07_compile_PM-1024_r128k4_1024.sh"; rc=$?
echo "step_exit[compile_PM-1024_r128k4_1024]=$rc"

echo "=== $(date -u +%FT%TZ) compile_PM-1024_r384k8_1024"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/08_compile_PM-1024_r384k8_1024.sh"; rc=$?
echo "step_exit[compile_PM-1024_r384k8_1024]=$rc"

echo "=== $(date -u +%FT%TZ) compile_ours_R128_4096"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/09_compile_ours_R128_4096.sh"; rc=$?
echo "step_exit[compile_ours_R128_4096]=$rc"

echo "=== $(date -u +%FT%TZ) compile_ours_R384_4096"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/10_compile_ours_R384_4096.sh"; rc=$?
echo "step_exit[compile_ours_R384_4096]=$rc"

echo "=== $(date -u +%FT%TZ) compile_CPR-256_r128k4_4096"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/11_compile_CPR-256_r128k4_4096.sh"; rc=$?
echo "step_exit[compile_CPR-256_r128k4_4096]=$rc"

echo "=== $(date -u +%FT%TZ) compile_CPR-256_r384k8_4096"
timeout --signal=TERM --kill-after=30 300 bash "$TASK_ROOT/cmds/12_compile_CPR-256_r384k8_4096.sh"; rc=$?
echo "step_exit[compile_CPR-256_r384k8_4096]=$rc"

cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 || FAIL=1
echo "job_fail=$FAIL"
echo ALL-DONE
exit $FAIL
