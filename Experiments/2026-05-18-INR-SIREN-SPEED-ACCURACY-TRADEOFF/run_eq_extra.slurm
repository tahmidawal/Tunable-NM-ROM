#!/usr/bin/env bash
# Build EQ index sets for smaller coarse-stage counts (50/100/200/250).
#SBATCH --job-name=siren_speed_eq
#SBATCH --partition=preempt
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

set -uo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load modtree/deprecated
module load python/3.10.4
export PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:${PYTHONPATH:-}

ROOT="/cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Tunable-ROM/Experiments/2026-05-18-INR-SIREN-SPEED-ACCURACY-TRADEOFF"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

# We need a real directory (not the symlink) for new EQ outputs.
EQ_OUT="$ROOT/runs/rom/eq_extra"
mkdir -p "$EQ_OUT"

python3 -m scripts.build_eq_sweep \
  --data data/poisson2d_cg_n128_s42.npz \
  --N 128 --spatial-dim 2 \
  --n-eq-targets 50 100 200 250 \
  --tag siren_wide \
  --out-dir "$EQ_OUT"

echo "=== EQ files created ==="
ls -la "$EQ_OUT"
