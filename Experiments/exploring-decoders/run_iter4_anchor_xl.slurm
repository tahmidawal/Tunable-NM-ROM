#!/usr/bin/env bash
# Iter 4: scaled-up anchor SIREN (hidden 512, mod 384, 80k epochs).
#SBATCH --job-name=expdec_iter4_xl
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:30:00
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
mkdir -p runs/rom/iter4

DATA="data/poisson2d_cg_n128_s42.npz"
CKPT="checkpoints/poisson2d_siren_anchor_xl.pkl"

echo "=== Iter 4 anchor XL train: $(date) ==="
$PY -m scripts.train_rom_anchor \
    --config configs/poisson2d_siren_anchor_xl.yaml \
    --data "$DATA" \
    --out "$CKPT"
RC_TRAIN=$?
if [[ $RC_TRAIN -ne 0 ]]; then echo "[iter4] train FAILED rc=$RC_TRAIN"; exit $RC_TRAIN; fi

# Need to build per-XL EQ set since EQ-NNLS uses |K @ u| (independent of
# decoder), but the user previously said to keep per-decoder NNLS files.
# These XL EQ sets reuse the same U_train so are numerically identical
# to siren_wide; we save under tag "siren_xl" for hygiene.
echo "=== Iter 4 build EQ: $(date) ==="
$PY -m scripts.build_eq_sweep \
    --data "$DATA" \
    --n-eq-targets 500 1000 2000 4000 8000 \
    --tag siren_xl \
    --out-dir runs/rom/eq
RC_EQ=$?
if [[ $RC_EQ -ne 0 ]]; then echo "[iter4] EQ build FAILED rc=$RC_EQ"; exit $RC_EQ; fi

echo "=== Iter 4 ROM sweep COLD (30 it): $(date) ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_ANCHOR_XL=$CKPT" \
    --eq-targets 500 1000 2000 4000 8000 \
    --eq-tag siren_xl \
    --gn-max-iters 30 \
    --out runs/rom/iter4/sweep_xl_cold_30it.json
RC_COLD=$?

echo "=== Iter 4 EQ continuation 500 -> 8000: $(date) ==="
$PY -m scripts.run_rom_continuation \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --eq-coarse runs/rom/eq/eq_siren_xl_500.npz \
    --eq-fine runs/rom/eq/eq_siren_xl_8000.npz \
    --gn-max-iters-coarse 30 \
    --gn-max-iters-fine 30 \
    --out runs/rom/iter4/continuation_xl_500_to_8000.json
RC_CONT=$?

echo "=== Iter 4 ROM sweep WARM (30 it): $(date) ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_ANCHOR_XL=$CKPT" \
    --eq-targets 500 1000 2000 4000 8000 \
    --eq-tag siren_xl \
    --gn-max-iters 30 \
    --out runs/rom/iter4/sweep_xl_warm_30it.json \
    --warm-start
RC_WARM=$?

echo
echo "============================================"
echo "End: $(date)  rcs=($RC_TRAIN $RC_EQ $RC_COLD $RC_CONT $RC_WARM)"
echo "============================================"
exit $((RC_TRAIN + RC_EQ + RC_COLD + RC_CONT + RC_WARM))
