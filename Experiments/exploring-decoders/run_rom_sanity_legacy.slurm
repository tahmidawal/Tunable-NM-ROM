#!/usr/bin/env bash
# Sanity B: run the ORIGINAL training pipeline (scripts/train.py) but
# pointed at CG-discrete inputs. GT is still analytical-formula. This
# isolates whether the issue is the training pipeline vs the GT source.
#SBATCH --job-name=expdec_rom_sanity_legacy
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
mkdir -p runs data checkpoints

DATA="data/poisson2d_cg_n128_s42.npz"
# Reuse the analytical-data training pipeline; it expects U_train + freqs_train,
# both of which the CG data file has.
$PY -m scripts.train \
    --config configs/poisson2d_xattn_narrow_cg_2k.yaml \
    --data "$DATA" \
    --out checkpoints/poisson2d_xattn_narrow_cglegacy_2k.pkl
RC=$?
echo
echo "============================================"
echo "End: $(date)  train rc=$RC"
echo "============================================"
exit $RC
