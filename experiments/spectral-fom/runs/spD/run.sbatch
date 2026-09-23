#!/bin/bash
#SBATCH --job-name=specfom_spD
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/specfom_20260923/spD/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/specfom_20260923/spD/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/specfom_20260923/spD
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
export PYTHONPATH="$TASK_ROOT/code/experiments/heat-bank-knob"
"$PY" sp_heat.py --config config-h2d-1024.json --out ../output0
"$PY" sp_audit_np.py ../output0 --delete-fields || echo "AUDIT-FAILED ../output0"
"$PY" sp_heat.py --config config-h2d-2048.json --out ../output1
"$PY" sp_audit_np.py ../output1 --delete-fields || echo "AUDIT-FAILED ../output1"
"$PY" sp_heat.py --config config-h2d-4096.json --out ../output2
"$PY" sp_audit_np.py ../output2 --delete-fields || echo "AUDIT-FAILED ../output2"
cd "$TASK_ROOT"
rm -rf cache tmp
find output* -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
