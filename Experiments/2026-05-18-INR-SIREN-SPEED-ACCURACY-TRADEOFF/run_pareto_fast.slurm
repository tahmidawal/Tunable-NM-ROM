#!/usr/bin/env bash
# Pareto sweep using FastINRNMROMSolver (lever A').
# Cold-start composite: iter-2 anchor coarse + iter-5 lap fine.
#SBATCH --job-name=siren_speed_pareto_fast
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3:00:00
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
mkdir -p runs/pareto/fast

python3 -m scripts.pareto_sweep_fast \
  --coarse-ckpt checkpoints/poisson2d_siren_anchor_v1.pkl \
  --fine-ckpt   checkpoints/poisson2d_siren_lap_v1.pkl \
  --data        data/poisson2d_cg_n128_s42.npz \
  --eq-dir      runs/rom/eq \
  --eq-extra-dir runs/rom/eq_extra \
  --coarse-neq 200 250 500 1000 \
  --fine-neq   500 1000 2000 4000 8000 \
  --coarse-iters 10 20 30 \
  --fine-iters   3 5 10 20 \
  --out-dir     runs/pareto/fast \
  --out-csv     runs/pareto/fast/results_pareto_fast.csv
