#!/usr/bin/env bash
# 2000-epoch sanity run on XATTN_NARROW — does full-grid CG training learn?
#SBATCH --job-name=expdec_rom_sanity
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:30:00
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
if [[ ! -f "$DATA" ]]; then
    echo "[sanity] regenerating CG data"
    $PY -m scripts.generate_cg_data --out "$DATA"
    RC_GEN=$?
    if [[ $RC_GEN -ne 0 ]]; then echo "[gen] FAILED rc=$RC_GEN"; exit $RC_GEN; fi
fi
$PY -m scripts.train_rom \
    --config configs/poisson2d_xattn_narrow_cg_2k.yaml \
    --data "$DATA" \
    --out checkpoints/poisson2d_xattn_narrow_cg_2k.pkl
RC=$?
echo
echo "============================================"
echo "End: $(date)  train_rom rc=$RC"
echo "============================================"
exit $RC
