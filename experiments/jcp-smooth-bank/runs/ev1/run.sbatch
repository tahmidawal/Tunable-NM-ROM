#!/bin/bash
#SBATCH --job-name=jcps_ev1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:6

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcpsmooth/ev1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcpsmooth/ev1/logs/%j.err
set -uo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcpsmooth/ev1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet || exit 2
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES}"
[ "${#DEVS[@]}" -ge 6 ] || { echo "allocated devices ${CUDA_VISIBLE_DEVICES} < 6"; exit 3; }
for i in $(seq 0 5); do
  CUDA_VISIBLE_DEVICES=${DEVS[$i]} "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'dev ${DEVS[$i]} jax_backend={b} {jax.devices()[0].device_kind}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
done
cd experiments/jcp-smooth-bank
declare -a PIDS NAMES
CUDA_VISIBLE_DEVICES=${DEVS[0]} "$PY" bankeval.py --config jobs/ev1.json --task 0 --out "$TASK_ROOT/output/"eval > "$TASK_ROOT/logs/task0.log" 2>&1 &
PIDS[0]=$!; NAMES[0]=task0
CUDA_VISIBLE_DEVICES=${DEVS[1]} "$PY" bankeval.py --config jobs/ev1.json --task 1 --out "$TASK_ROOT/output/"eval > "$TASK_ROOT/logs/task1.log" 2>&1 &
PIDS[1]=$!; NAMES[1]=task1
CUDA_VISIBLE_DEVICES=${DEVS[2]} "$PY" bankeval.py --config jobs/ev1.json --task 2 --out "$TASK_ROOT/output/"eval > "$TASK_ROOT/logs/task2.log" 2>&1 &
PIDS[2]=$!; NAMES[2]=task2
CUDA_VISIBLE_DEVICES=${DEVS[3]} "$PY" bankeval.py --config jobs/ev1.json --task 3 --out "$TASK_ROOT/output/"eval > "$TASK_ROOT/logs/task3.log" 2>&1 &
PIDS[3]=$!; NAMES[3]=task3
CUDA_VISIBLE_DEVICES=${DEVS[4]} "$PY" bankeval.py --config jobs/ev1.json --task 4 --out "$TASK_ROOT/output/"eval > "$TASK_ROOT/logs/task4.log" 2>&1 &
PIDS[4]=$!; NAMES[4]=task4
CUDA_VISIBLE_DEVICES=${DEVS[5]} "$PY" bankeval.py --config jobs/ev1.json --task 5 --out "$TASK_ROOT/output/"eval > "$TASK_ROOT/logs/task5.log" 2>&1 &
PIDS[5]=$!; NAMES[5]=task5
FAIL=0
for i in $(seq 0 5); do
  wait ${PIDS[$i]}; rc=$?
  echo "task ${NAMES[$i]} exit=$rc"; echo "${NAMES[$i]} $rc" >> "$TASK_ROOT/output/TASK_STATUS.txt"
  [ $rc -eq 0 ] || FAIL=1
done
if [ $FAIL -eq 0 ]; then CUDA_VISIBLE_DEVICES=${DEVS[0]} "$PY" banktime.py --config jobs/ev1.json --evals "$TASK_ROOT/output/"eval --out "$TASK_ROOT/output/"timing > "$TASK_ROOT/logs/post0.log" 2>&1; rc=$?; echo "post0 $rc" >> "$TASK_ROOT/output/TASK_STATUS.txt"; [ $rc -eq 0 ] || FAIL=1; fi

cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 || { echo "CHECKSUM FAILED"; FAIL=1; }
[ $FAIL -eq 0 ] || { echo "SOME TASKS FAILED"; exit 1; }
echo ALL-DONE
