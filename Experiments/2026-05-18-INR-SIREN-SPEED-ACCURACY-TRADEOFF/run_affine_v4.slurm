#!/usr/bin/env bash
# Lever D iter 4: no anchor + very weak L2(z).
#SBATCH --job-name=siren_speed_affine_v4
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1:00:00
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

DATA="data/poisson2d_cg_n128_s42.npz"
CKPT="checkpoints/poisson2d_affine_v4.pkl"

echo "=== Affine v4 train: $(date) ==="
python3 -m scripts.train_rom_anchor \
    --config configs/poisson2d_affine_v4.yaml \
    --data "$DATA" --out "$CKPT"
RC=$?; [[ $RC -ne 0 ]] && exit $RC

echo "=== Affine v4 cold-start eval: $(date) ==="
python3 -m scripts.run_rom_affine_coarse \
    --ckpt "$CKPT" --eq runs/rom/eq/eq_siren_wide_500.npz \
    --data "$DATA" --iters 30 \
    --out runs/rom/affine_v4/coarse_500.json
python3 -m scripts.run_rom_affine_coarse \
    --ckpt "$CKPT" --eq runs/rom/eq/eq_siren_wide_2000.npz \
    --data "$DATA" --iters 30 \
    --out runs/rom/affine_v4/coarse_2000.json

echo "=== Affine v4 manifold diagnostic ==="
python3 -m scripts.diagnose_affine_manifold \
    --ckpt "$CKPT" \
    --eq runs/rom/eq/eq_siren_wide_500.npz \
    --data "$DATA" --n-samples 12
