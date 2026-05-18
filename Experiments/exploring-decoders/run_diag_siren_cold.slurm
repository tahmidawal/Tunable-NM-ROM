#!/usr/bin/env bash
# Diagnose cold-start NM-ROM failure for SIREN_WIDE.
#SBATCH --job-name=expdec_diag_siren
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:20:00
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
mkdir -p runs/rom

echo "=== diagnose_siren_cold start: $(date) ==="
$PY -m scripts.diagnose_siren_cold \
    --ckpt checkpoints/poisson2d_siren_wide_cg.pkl \
    --eq runs/rom/eq/eq_siren_wide_1000.npz \
    --data data/poisson2d_cg_n128_s42.npz \
    --out runs/rom/diagnostic_siren_cold.json
RC=$?
echo
echo "============================================"
echo "End: $(date)  diag rc=$RC"
echo "============================================"
exit $RC
