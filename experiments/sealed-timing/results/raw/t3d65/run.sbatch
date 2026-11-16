#!/usr/bin/env bash
#SBATCH --job-name=stm_t3d65
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=45
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/sealed-timing-20261115/t3d65
#SBATCH --output=/cluster/tufts/paralab/tawal01/sealed-timing-20261115/t3d65/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sealed-timing-20261115/t3d65/%j.err
set -euo pipefail
umask 077
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/sealed-timing-20261115/t3d65/cache TMPDIR=/cluster/tufts/paralab/tawal01/sealed-timing-20261115/t3d65/tmp PYTHONDONTWRITEBYTECODE=1
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store SOURCE_COMMIT=138ec364b21661f6b687879b0db7bd6626b7884a
unset JCP_OPEN_REGISTRY JCP_APPROVED_REVIEW_SHA256
mkdir -p output
nvidia-smi --query-gpu=name,uuid,memory.total,clocks.max.sm,clocks.max.mem --format=csv,noheader | tee output/gpu-start.txt
nvidia-smi >> output/gpu-start.txt
squeue -w "$(hostname -s)" -o "%i|%u|%j|%b|%C" > output/node-occupancy-start.txt || true
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' and jax.config.jax_enable_x64 and str(jax.config.jax_default_matmul_precision)=='highest' else 42)"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_throttle_reasons.active --format=csv,noheader -l 1 > output/gpu-trace.csv &
SMI=$!
"$PY" experiments/sealed-timing/timing.py --base /cluster/tufts/paralab/tawal01/sealed-timing-20261115/base/experiments/burgers-quadrature-plateau --dim 3 --mesh 65 --out output
kill $SMI || true
squeue -w "$(hostname -s)" -o "%i|%u|%j|%b|%C" > output/node-occupancy-end.txt || true
rm -rf output/fields
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
