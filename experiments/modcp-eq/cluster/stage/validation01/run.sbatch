#!/bin/bash
#SBATCH --job-name=ctol_modcp_b2d_validation01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/validation01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/validation01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/modcp_burgers2d_20260910/validation01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export COMMIT=$(cat COMMIT.txt)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" code/tests/check_burgers.py
"$PY" code/burgers/campaign.py --out out --phase validate
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
printf 'ALL-DONE\n'
