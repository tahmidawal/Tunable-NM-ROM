#!/usr/bin/env bash
# SLURM launcher for one Heat-3D mirrored-ViT config.
# Usage:
#   sbatch run_experiment.slurm <config_name> [stage]
# Example:
#   sbatch run_experiment.slurm heat3d_n64_A_mirror all
# Stages: data | train | rom | all   (default: all)
#SBATCH --job-name=h3d_vit
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

set -euo pipefail

CONFIG_NAME="${1:?config name required, e.g. heat3d_n64_A_mirror}"
STAGE="${2:-all}"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load modtree/deprecated
module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x
export PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

PKG_DIR="/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Tunable-ROM/mirrored-vit-ablation/heat"
cd "$PKG_DIR"
export PYTHONPATH="$PKG_DIR/src:$PYTHONPATH"

mkdir -p data checkpoints runs/$CONFIG_NAME
CONFIG="configs/${CONFIG_NAME}.yaml"
DATA="data/${CONFIG_NAME}.npz"
CKPT="checkpoints/${CONFIG_NAME}.pkl"
LOG="runs/$CONFIG_NAME"

echo "============================================"
echo "Heat-3D mirrored-ViT: $CONFIG_NAME  stage=$STAGE"
echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start: $(date)"
echo "============================================"

if [[ "$STAGE" == "data" || "$STAGE" == "all" ]]; then
  if [[ ! -f "$DATA" ]]; then
    echo "[data] generating $DATA"
    python3 -m scripts.generate_data --config "$CONFIG" --out "$DATA" 2>&1 | tee "$LOG/data.log"
  else
    echo "[data] $DATA already exists; skipping"
  fi
fi

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  echo "[train] training to $CKPT"
  python3 -m scripts.train --config "$CONFIG" --data "$DATA" --out "$CKPT" 2>&1 | tee "$LOG/train.log"
fi

if [[ "$STAGE" == "rom" || "$STAGE" == "all" ]]; then
  echo "[rom] running ROM benchmark"
  python3 -m scripts.run_rom --config "$CONFIG" --data "$DATA" --ckpt "$CKPT" 2>&1 | tee "$LOG/rom.log"
fi

echo "End: $(date)"
