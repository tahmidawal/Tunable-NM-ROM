#!/usr/bin/env bash
# make_cell.sh <cell-name> <mem> <hours> "<env assignments>" "<command line...>"
# Emits the local staging dir cluster/stage/<cell>/ with code + FROZEN deps +
# run.sbatch for /cluster/tufts/paralab/tawal01/wlat/<cell>/ (ONE job per dir).
set -euo pipefail
cell="$1"; mem="$2"; hours="$3"; envs="$4"; cmd="$5"
gpu="${GPU_TYPE:-a100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REMOTE="/cluster/tufts/paralab/tawal01/wlat/$cell"
STAGE="$HERE/stage/$cell"
rm -rf "$STAGE"; mkdir -p "$STAGE/code" "$STAGE/logs" "$STAGE/out" "$STAGE/code/in"
cp "$EXP"/wlat_*.py "$STAGE/code/"
# frozen dependency copies (deps/PROVENANCE.md) ship verbatim
cp -r "$EXP/deps" "$STAGE/code/deps"
find "$STAGE/code" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cat > "$STAGE/run.sbatch" <<SBATCH
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:$gpu:1
#SBATCH -c 8
#SBATCH --mem=$mem
#SBATCH -t $hours:00:00
#SBATCH -o $REMOTE/logs/%j.out
#SBATCH -e $REMOTE/logs/%j.err
set -euo pipefail
cd "$REMOTE/code"
export PYTHONPATH="$REMOTE/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=$(git -C "$EXP" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "cell=$cell"
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
SBATCH
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
