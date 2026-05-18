#!/usr/bin/env bash
# Smoke SLURM job for exploring-decoders.
# Runs the pytest smoke + a tiny end-to-end train on a compute node.
#SBATCH --job-name=expdec_smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=0:30:00
#SBATCH --output=runs/smoke_%j.out
#SBATCH --error=runs/smoke_%j.err

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

echo "============================================"
echo "exploring-decoders smoke"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start:  $(date)"
echo "Python: $($PY --version)"
echo "============================================"

# Stage 1: pytest unit smoke.
echo
echo "[smoke] pytest"
$PY -m pytest -x -v tests/test_decoders.py
RC_PY=$?
if [[ $RC_PY -ne 0 ]]; then
    echo "[smoke] pytest FAILED (rc=$RC_PY)"
    exit $RC_PY
fi

# Stage 2: end-to-end smoke train for both decoders.
DATA="data/smoke.npz"
echo
echo "[smoke] generate"
$PY -m scripts.generate_data --config configs/smoke.yaml --out "$DATA"

for KIND in siren xattn; do
    CFG="configs/smoke_${KIND}.yaml"
    # Build a per-decoder smoke config from the base smoke.yaml by overriding decoder_kind.
    $PY - <<PY
import yaml, pathlib
base = yaml.safe_load(open("configs/smoke.yaml"))
base["decoder_kind"] = "${KIND}"
pathlib.Path("${CFG}").write_text(yaml.safe_dump(base))
PY
    CKPT="checkpoints/smoke_${KIND}.pkl"
    echo
    echo "[smoke] train ${KIND}"
    $PY -m scripts.train --config "$CFG" --data "$DATA" --out "$CKPT"
    RC=$?
    if [[ $RC -ne 0 ]]; then
        echo "[smoke] train ${KIND} FAILED (rc=$RC)"
        exit $RC
    fi
done

echo
echo "============================================"
echo "End: $(date)   smoke OK"
echo "============================================"
exit 0
