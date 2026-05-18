#!/usr/bin/env bash
# Oracle-vs-ROM diagnostic for affine v1/v2.
#SBATCH --job-name=siren_speed_diag_affine
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=0:20:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

set -uo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load modtree/deprecated
module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
export PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

ROOT="/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Tunable-ROM/Experiments/2026-05-18-INR-SIREN-SPEED-ACCURACY-TRADEOFF"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

for V in v1 v2; do
  echo "=== diagnose affine $V ==="
  python3 -m scripts.diagnose_affine_manifold \
    --ckpt checkpoints/poisson2d_affine_${V}.pkl \
    --eq runs/rom/eq/eq_siren_wide_500.npz \
    --data data/poisson2d_cg_n128_s42.npz \
    --n-samples 12
done
