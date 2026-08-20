#!/usr/bin/env bash
# make_cell.sh CELL MEMORY WALLTIME ENV COMMAND [GPU]
set -euo pipefail
cell="$1"
memory="$2"
walltime="$3"
envs="$4"
command="$5"
gpu_type="${6:-h100}"
job_name="ctol_b10_${cell}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"
ROOT="$(cd "$WORKTREE/../.." && pwd)"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
STAGE="$HERE/stage/$cell"
COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
DIRTY="$(git -C "$WORKTREE" status --porcelain -- "$EXP" | sha256sum | cut -c1-12)"

# Remove only this cell's local staging copy, never a pulled run.
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/burgers2d-coord-rom"
cp "$EXP"/b10_*.py "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
  "$STAGE/code/deps/burgers2d-coord-rom/"

INHERITED="$ROOT/worktrees/2026-08-19-nonlinear-decoder-architecture/experiments/nonlinear-decoder-architecture/runs/nda_bg160l4g2f31_r2/out/blat_rom_N64_K16.json"
if [[ -f "$INHERITED" ]]; then
  cp "$INHERITED" "$STAGE/code/deps/inherited_h160.json"
fi

cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $job_name
#SBATCH -p gpu
#SBATCH --gres=gpu:$gpu_type:1
#SBATCH -c 8
#SBATCH --mem=$memory
#SBATCH -t $walltime
#SBATCH -o $REMOTE/logs/%j.out
#SBATCH -e $REMOTE/logs/%j.err
set -euo pipefail
cd "$REMOTE/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export B10_COMMIT=$COMMIT
echo "host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$COMMIT dirty=$DIRTY cell=$cell"
\$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export $envs
$command
echo ALL-DONE
EOF

(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
