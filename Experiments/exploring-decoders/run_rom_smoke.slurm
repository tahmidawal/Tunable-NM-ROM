#!/usr/bin/env bash
# Smoke test: generate CG data, train two small variants for 200 epochs,
# build a single EQ set, and run one ROM solve per decoder. Catches all
# the integration bugs before the 10h full jobs.
#SBATCH --job-name=expdec_rom_smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:45:00
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
mkdir -p runs/rom runs/rom/eq data checkpoints

DATA="data/poisson2d_cg_n128_s42.npz"

echo "=== Stage 1: CG data ==="
if [[ -f "$DATA" ]]; then
    echo "data already exists: $DATA"
else
    $PY -m scripts.generate_cg_data --out "$DATA"
    RC=$?
    if [[ $RC -ne 0 ]]; then echo "[data] FAILED rc=$RC"; exit $RC; fi
fi

echo
echo "=== Stage 2: smoke train XATTN_NARROW ==="
$PY -m scripts.train_rom \
    --config configs/poisson2d_xattn_narrow_cg_smoke.yaml \
    --data "$DATA" \
    --out checkpoints/poisson2d_xattn_narrow_cg_smoke.pkl
RC_A=$?

echo
echo "=== Stage 3: smoke train SIREN_WIDE ==="
$PY -m scripts.train_rom \
    --config configs/poisson2d_siren_wide_cg_smoke.yaml \
    --data "$DATA" \
    --out checkpoints/poisson2d_siren_wide_cg_smoke.pkl
RC_B=$?

echo
echo "=== Stage 4: build a single EQ at n_eq=1000 ==="
$PY -m scripts.build_eq_sweep \
    --data "$DATA" \
    --n-eq-targets 1000 \
    --tag smoke \
    --out-dir runs/rom/eq
RC_C=$?

echo
echo "=== Stage 5: ROM solve XATTN_NARROW (smoke) ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "XATTN_NARROW=checkpoints/poisson2d_xattn_narrow_cg_smoke.pkl" \
    --eq-targets 1000 \
    --eq-tag smoke \
    --out runs/rom/sweep_smoke_xattn.json
RC_D=$?

echo
echo "=== Stage 6: ROM solve SIREN_WIDE (smoke) ==="
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_WIDE=checkpoints/poisson2d_siren_wide_cg_smoke.pkl" \
    --eq-targets 1000 \
    --eq-tag smoke \
    --out runs/rom/sweep_smoke_siren.json
RC_E=$?

echo
echo "============================================"
echo "End: $(date)  rcs=($RC_A $RC_B $RC_C $RC_D $RC_E)"
echo "============================================"
exit $((RC_A + RC_B + RC_C + RC_D + RC_E))
