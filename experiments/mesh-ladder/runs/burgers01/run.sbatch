#!/bin/bash
#SBATCH --job-name=ctol_mrladder_burgers
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mrladder_20260914/burgers01/job.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mrladder_20260914/burgers01/job.err
set -euo pipefail
REMOTE=/cluster/tufts/paralab/tawal01/mrladder_20260914/burgers01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd "$REMOTE/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export COMMIT=521cdced6f4d0292872f08c0479ebd257c7b4350
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES commit=$COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# A node with a failed cuInit falls back to CPU silently and returns plausible,
# invalid timings. Refuse to do any work in that case.
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
df -h "$REMOTE" | tail -1
"$PY" experiments/mesh-ladder/ladder_burgers.py \
  --config experiments/mesh-ladder/config-burgers.json \
  --out "$REMOTE/out"
echo "DRIVER-EXIT=$?"
cd "$REMOTE/out"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$REMOTE/OUTPUTS.sha256"
echo "SBATCH-COMPLETE"
