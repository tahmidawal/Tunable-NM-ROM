#!/usr/bin/env bash
# SLURM job: run pytest smoke tests on a compute node for both packages.
# Gates the full sweep — only submit training after this passes.
#SBATCH --job-name=mirvit_smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=runs/smoke_%j.out
#SBATCH --error=runs/smoke_%j.err

set -uo pipefail   # don't -e: we want both packages' pytest to run

module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
export PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

ROOT="/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Tunable-ROM/mirrored-vit-ablation"
mkdir -p "$ROOT/runs"

echo "============================================"
echo "mirrored-ViT smoke test"
echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start: $(date)"
echo "============================================"

H_RC=0
P_RC=0

cd "$ROOT/heat"
export PYTHONPATH="$ROOT/heat/src:$PYTHONPATH"
echo "[heat] running pytest tests/test_smoke.py"
python -m pytest -x -v tests/test_smoke.py
H_RC=$?
echo "[heat] pytest rc=$H_RC"

# Reset PYTHONPATH before poisson so heat's src isn't shadowing.
export PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:${PYTHONPATH_ORIG:-}
cd "$ROOT/poisson"
export PYTHONPATH="$ROOT/poisson/src:$PYTHONPATH"
echo "[poisson] running pytest tests/test_smoke.py"
python -m pytest -x -v tests/test_smoke.py
P_RC=$?
echo "[poisson] pytest rc=$P_RC"

echo "============================================"
echo "Heat rc=$H_RC  Poisson rc=$P_RC"
echo "End: $(date)"
echo "============================================"

if [[ $H_RC -ne 0 || $P_RC -ne 0 ]]; then
  exit 1
fi
exit 0
