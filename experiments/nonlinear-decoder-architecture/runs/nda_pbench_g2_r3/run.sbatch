#!/bin/bash
#SBATCH -J nda_pbench_g2_r3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pbench_g2_r3/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pbench_g2_r3/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pbench_g2_r3"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "cell=nda_pbench_g2_r3 commit=a5e54a80f44fbede236212f4d58b8bc5adb5c0f7 host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export PDE=poisson HIDDEN=128 N_LAYERS=4 TIME_REPS=9 TIME_WARM=3 BURN_SECONDS=1.0
$PY -u nda_benchmark.py ../ckpt/control.pkl ../ckpt/variant.pkl ../out/benchmark.json
$PY -c "import json; d=json.load(open('../out/benchmark.json')); assert d['backend']=='gpu' and d['checks']['raw_vs_cached_max_relative'] <= 1e-13"
echo ALL-DONE
