#!/usr/bin/env bash
# Iter 3: solver-side experiments on the iter2 anchor checkpoint.
#   3a: cold sweep with gn_max_iters=30 instead of 12.
#   3b: EQ continuation (cold @ n_eq=500 -> warm @ n_eq=8000).
#SBATCH --job-name=expdec_iter3_solver
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
mkdir -p runs/rom/iter3

DATA="data/poisson2d_cg_n128_s42.npz"
CKPT="checkpoints/poisson2d_siren_anchor_v1.pkl"

echo "=== Iter 3a: cold sweep with gn_max_iters=30 ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_ANCHOR=$CKPT" \
    --eq-targets 500 1000 2000 4000 8000 \
    --eq-tag siren_wide \
    --gn-max-iters 30 \
    --out runs/rom/iter3/sweep_anchor_cold_30it.json
RC_A=$?

echo "=== Iter 3b: EQ continuation 500 -> 8000 ==="
$PY -m scripts.run_rom_continuation \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --eq-coarse runs/rom/eq/eq_siren_wide_500.npz \
    --eq-fine runs/rom/eq/eq_siren_wide_8000.npz \
    --gn-max-iters-coarse 30 \
    --gn-max-iters-fine 30 \
    --out runs/rom/iter3/continuation_500_to_8000.json
RC_B=$?

echo "=== Iter 3c: EQ continuation 500 -> 2000 ==="
$PY -m scripts.run_rom_continuation \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --eq-coarse runs/rom/eq/eq_siren_wide_500.npz \
    --eq-fine runs/rom/eq/eq_siren_wide_2000.npz \
    --gn-max-iters-coarse 30 \
    --gn-max-iters-fine 30 \
    --out runs/rom/iter3/continuation_500_to_2000.json
RC_C=$?

echo
echo "============================================"
echo "End: $(date)  rcs=($RC_A $RC_B $RC_C)"
echo "============================================"
exit $((RC_A + RC_B + RC_C))
