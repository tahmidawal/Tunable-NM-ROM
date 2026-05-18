#!/usr/bin/env bash
# Build EQ sets (per decoder, per n_eq) and run the ROM Pareto sweep.
#SBATCH --job-name=expdec_rom_sweep
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
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
mkdir -p runs/rom runs/rom/eq

DATA="data/poisson2d_cg_n128_s42.npz"
EQ_TARGETS=(500 1000 2000 4000 8000)

# Per-decoder NNLS solves (user requested per-decoder, even though NNLS
# is decoder-agnostic — we honor the request for safety).
for TAG in xattn_narrow siren_wide xattn_wide; do
    echo "=== building EQ for ${TAG} ==="
    $PY -m scripts.build_eq_sweep \
        --data "$DATA" \
        --n-eq-targets ${EQ_TARGETS[@]} \
        --tag "$TAG" \
        --out-dir runs/rom/eq
    RC=$?
    if [[ $RC -ne 0 ]]; then echo "[eq] FAILED tag=$TAG rc=$RC"; exit $RC; fi
done

echo
echo "=== running ROM sweep ==="
# Three separate sweep runs, one per decoder, each using its own EQ tag.
$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "XATTN_NARROW=checkpoints/poisson2d_xattn_narrow_cg.pkl" \
    --eq-targets ${EQ_TARGETS[@]} \
    --eq-tag xattn_narrow \
    --out runs/rom/sweep_xattn_narrow.json
RC1=$?

$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "SIREN_WIDE=checkpoints/poisson2d_siren_wide_cg.pkl" \
    --eq-targets ${EQ_TARGETS[@]} \
    --eq-tag siren_wide \
    --out runs/rom/sweep_siren_wide.json
RC2=$?

$PY -m scripts.run_rom_sweep \
    --data "$DATA" \
    --ckpts "XATTN_WIDE=checkpoints/poisson2d_xattn_wide_cg.pkl" \
    --eq-targets ${EQ_TARGETS[@]} \
    --eq-tag xattn_wide \
    --out runs/rom/sweep_xattn_wide.json
RC3=$?

echo "sweep return codes: $RC1 $RC2 $RC3"

# Merge JSON outputs into a single sweep.json and plot.
$PY - <<'PY'
import json, pathlib
out = {"results": []}
fom_t = None
for f in [
    "runs/rom/sweep_xattn_narrow.json",
    "runs/rom/sweep_siren_wide.json",
    "runs/rom/sweep_xattn_wide.json",
]:
    p = pathlib.Path(f)
    if not p.exists(): continue
    obj = json.loads(p.read_text())
    if fom_t is None and "fom_time_median" in obj:
        fom_t = obj["fom_time_median"]
    out["results"].extend(obj.get("results", []))
out["fom_time_median"] = fom_t
pathlib.Path("runs/rom/sweep.json").write_text(json.dumps(out, indent=2))
print(f"Merged {len(out['results'])} sweep entries into runs/rom/sweep.json")
PY

$PY -m scripts.plot_rom_pareto --json runs/rom/sweep.json --out runs/rom/plots
RC_PLOT=$?

echo
echo "============================================"
echo "End: $(date)  rcs=($RC1 $RC2 $RC3 $RC_PLOT)"
echo "============================================"
exit $((RC1 + RC2 + RC3 + RC_PLOT))
