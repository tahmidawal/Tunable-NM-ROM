#!/usr/bin/env bash
# Iteration 2: baseline + strong L2(z) + anchor-at-zero loss.
#SBATCH --job-name=expdec_iter2_anchor
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

PY=python3
ROOT="/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Tunable-ROM/Experiments/exploring-decoders"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"
mkdir -p runs/rom/iter2

DATA="data/poisson2d_cg_n128_s42.npz"
CKPT="checkpoints/poisson2d_siren_anchor_v1.pkl"

echo "=== Iter 2 anchor train: $(date) ==="
$PY -m scripts.train_rom_anchor \
    --config configs/poisson2d_siren_anchor_v1.yaml \
    --data "$DATA" \
    --out "$CKPT"
RC_TRAIN=$?
if [[ $RC_TRAIN -ne 0 ]]; then echo "[iter2] train FAILED rc=$RC_TRAIN"; exit $RC_TRAIN; fi

echo "=== Iter 2 ROM sweep COLD: $(date) ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_ANCHOR=$CKPT" \
    --eq-targets 500 1000 2000 4000 8000 \
    --eq-tag siren_wide \
    --out runs/rom/iter2/sweep_anchor_cold.json
RC_COLD=$?

echo "=== Iter 2 ROM sweep WARM: $(date) ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_ANCHOR=$CKPT" \
    --eq-targets 500 1000 2000 4000 8000 \
    --eq-tag siren_wide \
    --out runs/rom/iter2/sweep_anchor_warm.json \
    --warm-start
RC_WARM=$?

echo
echo "============================================"
echo "End: $(date)  rcs=($RC_TRAIN $RC_COLD $RC_WARM)"
echo "============================================"
exit $((RC_TRAIN + RC_COLD + RC_WARM))
