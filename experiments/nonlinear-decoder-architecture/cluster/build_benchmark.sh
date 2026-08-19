#!/usr/bin/env bash
# build_benchmark.sh <poisson|burgers> <cell> <variant-pkl>
set -euo pipefail

pde="$1"; cell="$2"; variant="$3"
[[ "$pde" == poisson || "$pde" == burgers ]] || { echo "invalid PDE" >&2; exit 2; }
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell" >&2; exit 2; }
[[ -f "$variant" ]] || { echo "missing variant checkpoint $variant" >&2; exit 3; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
WT="$(cd "$EXP/../.." && pwd)"
MSP="$WT/../2026-08-14-multistage-precision/experiments/multistage-precision"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
STAGE="$HERE/stage/$cell"
[[ ! -e "$STAGE" ]] || { echo "stage exists: $STAGE" >&2; exit 4; }

if [[ "$pde" == poisson ]]; then
  control="$WT/experiments/cost-to-tolerance/ckpt_poisson/autodec_K16_N64_hbc_stages.pkl"
  hidden=128; layers=4
else
  control="$WT/experiments/cost-to-tolerance/ckpt_burgers/blat_ad_N64_K16.pkl"
  hidden=256; layers=5
fi
[[ -f "$control" ]] || { echo "missing control checkpoint $control" >&2; exit 3; }

mkdir -p "$STAGE/code/deps/multistage-precision" "$STAGE/ckpt" "$STAGE/out" "$STAGE/logs"
cp "$EXP/nda_arch.py" "$EXP/nda_benchmark.py" "$STAGE/code/"
cp "$MSP/ms_parametric.py" "$STAGE/code/deps/multistage-precision/"
cp "$control" "$STAGE/ckpt/control.pkl"
cp "$variant" "$STAGE/ckpt/variant.pkl"
commit="$(git -C "$WT" rev-parse HEAD)"
cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o $REMOTE_ROOT/$cell/logs/%j.out
#SBATCH -e $REMOTE_ROOT/$cell/logs/%j.err
set -euo pipefail
cd "$REMOTE_ROOT/$cell"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=$PY
echo "cell=$cell commit=$commit host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
\$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export PDE=$pde HIDDEN=$hidden N_LAYERS=$layers TIME_REPS=9 TIME_WARM=3 BURN_SECONDS=1.0
\$PY -u nda_benchmark.py ../ckpt/control.pkl ../ckpt/variant.pkl ../out/benchmark.json
\$PY -c "import json; d=json.load(open('../out/benchmark.json')); assert d['backend']=='gpu' and d['checks']['raw_vs_cached_max_relative'] <= 1e-13"
echo ALL-DONE
EOF
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"

