#!/bin/bash
#SBATCH --job-name=g2d_pb512
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/job.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/job.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_FRAC:-0.90}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ROOT/out"
cd "$ROOT"; sha256sum -c MANIFEST.sha256 --quiet
echo "host=$(hostname) job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
export PYTHONPATH="/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/mr-burgers2d:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/separable-decoder:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/head-ablation:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/cheap-corrections:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/b-ladder-top:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/b-panel/speed:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/hires-burgers:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/burgers-repanel:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/burgers-bank-knob:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/burgers2d-speed:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/burgers-compare-hires:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/src/experiments/burgers-compare-hires/ops:/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/pb512/scripts"
export OUT=$ROOT/out
cd "$ROOT/src"
set +e
PY=$PY OUT=$OUT ROOT=$ROOT bash "$ROOT/body.sh"; STATUS=$?
set -e
cd "$ROOT"; find out -type f \( -name '*.json' -o -name '*.pt' -o -name '*.log' \) -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo "run_exit=$STATUS"; exit $STATUS
