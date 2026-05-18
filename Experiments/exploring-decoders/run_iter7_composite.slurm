#!/usr/bin/env bash
# Iter 7: composite cold-start using iter2 anchor (coarse) + iter5 lap (fine).
# The user effectively gets cold-start without giving up iter5's manifold quality.
#SBATCH --job-name=expdec_iter7_comp
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=0:45:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

set -uo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load modtree/deprecated
module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
export PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

PY=python3
ROOT="/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Tunable-ROM/Experiments/exploring-decoders"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"
mkdir -p runs/rom/iter7

DATA="data/poisson2d_cg_n128_s42.npz"
COARSE="checkpoints/poisson2d_siren_anchor_v1.pkl"     # iter2
FINE="checkpoints/poisson2d_siren_lap_v1.pkl"          # iter5

echo "=== Iter 7 composite: coarse iter2 + fine iter5, eq 500 -> 8000 ==="
$PY -m scripts.run_rom_composite \
    --coarse-ckpt "$COARSE" --fine-ckpt "$FINE" \
    --coarse-eq runs/rom/eq/eq_siren_wide_500.npz \
    --fine-eq runs/rom/eq/eq_siren_wide_8000.npz \
    --coarse-iters 30 --fine-iters 30 \
    --data "$DATA" \
    --out runs/rom/iter7/composite_iter2_iter5_500_to_8000.json
RC_A=$?

echo "=== Iter 7 composite: coarse iter2 + fine iter5, eq 500 -> 2000 ==="
$PY -m scripts.run_rom_composite \
    --coarse-ckpt "$COARSE" --fine-ckpt "$FINE" \
    --coarse-eq runs/rom/eq/eq_siren_wide_500.npz \
    --fine-eq runs/rom/eq/eq_siren_wide_2000.npz \
    --coarse-iters 30 --fine-iters 30 \
    --data "$DATA" \
    --out runs/rom/iter7/composite_iter2_iter5_500_to_2000.json
RC_B=$?

echo "=== Iter 7 composite: coarse iter2 + fine iter5, eq 500 -> 500 ==="
$PY -m scripts.run_rom_composite \
    --coarse-ckpt "$COARSE" --fine-ckpt "$FINE" \
    --coarse-eq runs/rom/eq/eq_siren_wide_500.npz \
    --fine-eq runs/rom/eq/eq_siren_wide_500.npz \
    --coarse-iters 30 --fine-iters 30 \
    --data "$DATA" \
    --out runs/rom/iter7/composite_iter2_iter5_500_to_500.json
RC_C=$?

echo
echo "============================================"
echo "End: $(date)  rcs=($RC_A $RC_B $RC_C)"
echo "============================================"
exit $((RC_A + RC_B + RC_C))
