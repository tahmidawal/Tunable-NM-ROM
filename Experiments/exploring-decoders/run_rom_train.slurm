#!/usr/bin/env bash
# Full ROM-faithful retraining for one of the 3 Pareto-frontier decoders.
# Usage:  sbatch run_rom_train.slurm <config_name>
#         e.g.  sbatch run_rom_train.slurm poisson2d_xattn_narrow_cg
#SBATCH --job-name=expdec_rom
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

set -uo pipefail

CONFIG_NAME="${1:?config name required, e.g. poisson2d_xattn_narrow_cg}"

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
mkdir -p runs data checkpoints

CONFIG="configs/${CONFIG_NAME}.yaml"
DATA="data/poisson2d_cg_n128_s42.npz"
CKPT="checkpoints/${CONFIG_NAME}.pkl"

echo "============================================"
echo "ROM retrain: ${CONFIG_NAME}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start:  $(date)"
echo "============================================"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG"
    exit 2
fi

if [[ -f "$DATA" ]]; then
    echo "[data] already exists, skipping: $DATA"
else
    echo "[data] generating CG-discrete dataset"
    $PY -m scripts.generate_cg_data --out "$DATA"
    RC=$?
    if [[ $RC -ne 0 ]]; then echo "[data] FAILED rc=$RC"; exit $RC; fi
fi

echo
echo "[train_rom] starting"
$PY -m scripts.train_rom --config "$CONFIG" --data "$DATA" --out "$CKPT"
RC=$?
echo
echo "============================================"
echo "End: $(date)   train_rom rc=$RC"
echo "============================================"
exit $RC
