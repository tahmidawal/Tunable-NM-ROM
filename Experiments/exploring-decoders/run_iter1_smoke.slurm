#!/usr/bin/env bash
# Smoke: train iter1 VDF for 800 epochs and run a single ROM solve, to
# verify the model wires through cleanly.
#SBATCH --job-name=expdec_iter1_smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:25:00
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
mkdir -p runs/rom/iter1

DATA="data/poisson2d_cg_n128_s42.npz"
CKPT="checkpoints/poisson2d_siren_vdf_smoke.pkl"

echo "=== Iter 1 smoke train: $(date) ==="
$PY -m scripts.train_rom_vdf \
    --config configs/poisson2d_siren_vdf_smoke.yaml \
    --data "$DATA" \
    --out "$CKPT"
RC_TRAIN=$?
echo "[smoke] train rc=$RC_TRAIN"

echo "=== Iter 1 smoke ROM (cold, n_eq=1000 only): $(date) ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_VDF=$CKPT" \
    --eq-targets 1000 \
    --eq-tag siren_wide \
    --out runs/rom/iter1/smoke_cold.json
RC_COLD=$?
echo "[smoke] cold rc=$RC_COLD"

echo
echo "============================================"
echo "End: $(date)  rcs=($RC_TRAIN $RC_COLD)"
echo "============================================"
exit $((RC_TRAIN + RC_COLD))
