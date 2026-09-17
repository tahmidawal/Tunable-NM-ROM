#!/bin/bash
#SBATCH --job-name=wl_wl1024a
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/w_ladder_20260917/wl1024a/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/w_ladder_20260917/wl1024a/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/w_ladder_20260917/wl1024a
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
FREE_GB=$(df -BG --output=avail /cluster/tufts/paralab | tail -1 | tr -dc 0-9)
test "$FREE_GB" -ge 30 || { echo "paralab free ${FREE_GB}G < 30G: abort before writing"; exit 43; }
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" experiments/w-ladder/ladder.py \
  --config experiments/w-ladder/config-1024.json \
  --inputs experiments/multiresolution-wave/runs/accel12/cluster/in/dirichlet \
  --gates experiments/w-ladder/retained-gates.json \
  --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
