#!/bin/bash
#SBATCH --job-name=hp_hpl32fix
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/hires_p_20260920/hpl32/logs/%j.repair.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/hires_p_20260920/hpl32/logs/%j.repair.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/hires_p_20260920/hpl32
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
export SOURCE_COMMIT=34e0c645622e75e9e43729c20a06c54ead86b92a
cd "$TASK_ROOT/code"
echo "967ca1eacddce5d7c8725ac3e3e89fac430c84500534e93af2c744021c7fa228  hpl_repair_fine.py" | sha256sum -c --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" hpl_repair_fine.py ../output2
"$PY" hpl_audit_np.py ../output2 --subsample 128 --delete-fields
cd "$TASK_ROOT"
rm -rf cache tmp
find output output2 -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
