#!/usr/bin/env bash
# Full training SLURM job for exploring-decoders.
# Usage:  sbatch run_experiment.slurm <config_name>
#         e.g.  sbatch run_experiment.slurm poisson2d_siren
#               sbatch run_experiment.slurm poisson2d_xattn
#SBATCH --job-name=expdec
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

CONFIG_NAME="${1:-poisson2d_siren}"

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
# Data is shared across decoder_kind, so we use a single shared data file
# keyed on (N, n_train, n_val, seed).
DATA="data/poisson2d_n128_s42.npz"
CKPT="checkpoints/${CONFIG_NAME}.pkl"

echo "============================================"
echo "exploring-decoders run: ${CONFIG_NAME}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start:  $(date)"
echo "Config: $CONFIG"
echo "Data:   $DATA"
echo "Ckpt:   $CKPT"
echo "============================================"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG"
    exit 2
fi

# Stage 1: shared data generation (skip if already present).
if [[ -f "$DATA" ]]; then
    echo "[data] already exists, skipping: $DATA"
else
    echo "[data] generating"
    $PY -m scripts.generate_data --config "$CONFIG" --out "$DATA"
    RC=$?
    if [[ $RC -ne 0 ]]; then
        echo "[data] FAILED (rc=$RC)"
        exit $RC
    fi
fi

# Stage 2: train.
echo
echo "[train] starting"
$PY -m scripts.train --config "$CONFIG" --data "$DATA" --out "$CKPT"
RC=$?
echo
echo "============================================"
echo "End: $(date)   train rc=$RC"
echo "============================================"
exit $RC
