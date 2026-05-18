#!/usr/bin/env bash
# Full M-sweep eval across all 5 checkpoints + plot.
#SBATCH --job-name=expdec_eval_full
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=runs/eval_full_%j.out
#SBATCH --error=runs/eval_full_%j.err

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
mkdir -p runs

DATA="data/poisson2d_n128_s42.npz"
OUT_JSON="runs/sweep_full.json"
PLOT_DIR="runs/plots_full"

$PY -m scripts.eval_sweep_M \
    --ckpt checkpoints/poisson2d_siren.pkl \
    --ckpt checkpoints/poisson2d_siren_L3.pkl \
    --ckpt checkpoints/poisson2d_siren_wide.pkl \
    --ckpt checkpoints/poisson2d_xattn.pkl \
    --ckpt checkpoints/poisson2d_xattn_narrow.pkl \
    --data "$DATA" --out "$OUT_JSON"
RC1=$?
if [[ $RC1 -ne 0 ]]; then echo "[eval] FAILED rc=$RC1"; exit $RC1; fi

$PY -m scripts.plot_sweep --json "$OUT_JSON" --out "$PLOT_DIR"
exit $?
