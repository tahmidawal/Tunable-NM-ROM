#!/usr/bin/env bash
# make_cell.sh <cell-name> <mem> <hours> "<env assignments>" "<command line...>"
# Emits the local staging dir cluster/stage/<cell>/ with code + deps + run.sbatch
# for /cluster/tufts/paralab/tawal01/hlat/<cell>/ (one job per dir).
set -euo pipefail
cell="$1"; mem="$2"; hours="$3"; envs="$4"; cmd="$5"
[[ "$cell" =~ ^[a-z0-9_]+$ ]] || { echo "bad cell name '$cell' (need ^[a-z0-9_]+\$)" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="$(cd "$EXP/../../.." && pwd)"
REMOTE="/cluster/tufts/paralab/tawal01/hlat/$cell"
STAGE="$HERE/stage/$cell"
rm -rf "$STAGE"; mkdir -p "$STAGE/code/deps" "$STAGE/logs" "$STAGE/out"
cp "$EXP"/hlat_*.py "$STAGE/code/"
D="$STAGE/code/deps"
mkdir -p "$D/heat2d-coord-decoder" "$D/burgers2d-rom-latent-stepping" "$D/burgers2d-coord-rom" "$D/multistage-precision"
cp "$WT/2026-08-13-heat2d-coord-decoder/experiments/heat2d-coord-decoder/heat2d_film.py" "$D/heat2d-coord-decoder/"
cp "$WT/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/blat_common.py" "$D/burgers2d-rom-latent-stepping/"
cp "$WT/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$D/burgers2d-coord-rom/"
cp "$WT/2026-08-14-multistage-precision/experiments/multistage-precision/ms_parametric.py" \
   "$WT/2026-08-14-multistage-precision/experiments/multistage-precision/ms_autodecoder.py" "$D/multistage-precision/"
# optional extra inputs (checkpoints) are copied by the caller into $STAGE/code/in/
mkdir -p "$STAGE/code/in"
cat > "$STAGE/run.sbatch" <<EOS
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=$mem
#SBATCH -t $hours:00:00
#SBATCH -o $REMOTE/logs/%j.out
#SBATCH -e $REMOTE/logs/%j.err
set -euo pipefail
cd "$REMOTE/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=$(git -C "$EXP" rev-parse HEAD 2>/dev/null || echo unknown) dirty=$(git -C "$EXP" status --porcelain -- . | wc -l)"
\$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export $envs
$cmd
rc=\$?
[[ \$rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=\$rc"; exit \$rc; }
EOS
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "manifest=$(cd "$STAGE" && sha256sum MANIFEST.sha256 | cut -d' ' -f1)" >&2
echo "$STAGE"
