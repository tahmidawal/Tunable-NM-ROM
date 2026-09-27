#!/bin/bash
#SBATCH --job-name=ctol_fw_preflight01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-40G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/preflight01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/preflight01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/preflight01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT=179ce57e3ac0dc5a4c28a4916e68f83fd02ef1f7
export OUT="$TASK_ROOT/out"
export INPUT="$TASK_ROOT/in"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
finish() {
    task_exit=$?
    cd "$TASK_ROOT"
    printf '%s\n' "$task_exit" > EXIT_CODE
    find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
    exit "$task_exit"
}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
"$PY" -u fresh_preflight.py --config campaign-config.json --out /cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/preflight01/out/preflight
