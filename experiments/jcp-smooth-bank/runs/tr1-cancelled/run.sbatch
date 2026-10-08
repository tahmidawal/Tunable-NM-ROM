#!/bin/bash
#SBATCH --job-name=jcps_tr1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100-80G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=07:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcpsmooth/tr1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcpsmooth/tr1/logs/%j.err
set -uo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcpsmooth/tr1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet || exit 2
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES}"
[ "${#DEVS[@]}" -ge 4 ] || { echo "allocated devices ${CUDA_VISIBLE_DEVICES} < 4"; exit 3; }
for i in $(seq 0 3); do
  CUDA_VISIBLE_DEVICES=${DEVS[$i]} "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'dev ${DEVS[$i]} jax_backend={b} {jax.devices()[0].device_kind}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
done
cd experiments/jcp-smooth-bank
declare -a PIDS NAMES
CUDA_VISIBLE_DEVICES=${DEVS[0]} "$PY" smoothtrain.py --arm base  --sigma 4 --lam 0   --nodes 256 --out "$TASK_ROOT/output/"base --frozen ../separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl > "$TASK_ROOT/logs/base.log" 2>&1 &
PIDS[0]=$!; NAMES[0]=base
CUDA_VISIBLE_DEVICES=${DEVS[1]} "$PY" smoothtrain.py --arm sob01 --sigma 4 --lam 0.1 --nodes 256 --out "$TASK_ROOT/output/"sob01 > "$TASK_ROOT/logs/sob01.log" 2>&1 &
PIDS[1]=$!; NAMES[1]=sob01
CUDA_VISIBLE_DEVICES=${DEVS[2]} "$PY" smoothtrain.py --arm sig2  --sigma 2 --lam 0   --nodes 256 --out "$TASK_ROOT/output/"sig2 > "$TASK_ROOT/logs/sig2.log" 2>&1 &
PIDS[2]=$!; NAMES[2]=sig2
CUDA_VISIBLE_DEVICES=${DEVS[3]} "$PY" smoothtrain.py --arm sig1  --sigma 1 --lam 0   --nodes 256 --out "$TASK_ROOT/output/"sig1 > "$TASK_ROOT/logs/sig1.log" 2>&1 &
PIDS[3]=$!; NAMES[3]=sig1
FAIL=0
for i in $(seq 0 3); do
  wait ${PIDS[$i]}; rc=$?
  echo "task ${NAMES[$i]} exit=$rc"; echo "${NAMES[$i]} $rc" >> "$TASK_ROOT/output/TASK_STATUS.txt"
  [ $rc -eq 0 ] || FAIL=1
done

cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 || { echo "CHECKSUM FAILED"; FAIL=1; }
[ $FAIL -eq 0 ] || { echo "SOME TASKS FAILED"; exit 1; }
echo ALL-DONE
