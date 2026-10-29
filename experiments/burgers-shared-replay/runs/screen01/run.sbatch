#!/bin/bash
#SBATCH --job-name=bshared_replay_screen01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/bshared-replay-20261028/screen01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/bshared-replay-20261028/screen01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/bshared-replay-20261028/screen01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
trap 'rc=$?; cd "$TASK_ROOT"; find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; exit "$rc"' EXIT
timeout --signal=TERM --kill-after=15s 840s "$PY" experiments/burgers-shared-replay/pilot.py --model-dir model --out output/dev6
echo ALL-DONE
