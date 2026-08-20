#!/usr/bin/env bash
# Stage the one preregistered N=2048 learned-sensitivity cell.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: make_learned_cell.sh <cell>" >&2; exit 2; }
cell="$1"
if [[ "$cell" != ctol_hybb2048_learned* ]]; then
  echo "learned cell must match ctol_hybb2048_learned*" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
OLD="$EXP/../burgers-hybrid-1024"
WORKTREES="$(cd "$EXP/../../.." && pwd)"
REMOTE="/cluster/tufts/paralab/tawal01/hybb2048/$cell"
STAGE="$HERE/stage/$cell"
COMMIT="$(git -C "$EXP" rev-parse HEAD)"
if [[ -n "$(git -C "$EXP" status --porcelain)" ]]; then
  echo "refusing to stage a dirty worktree" >&2
  exit 3
fi
DIRTY="$(git -C "$EXP" status --porcelain -- . | sha256sum | cut -c1-12)"

# This removes only a local cell-specific staging directory, never results.
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" \
  "$STAGE/code/deps/burgers2d-coord-rom" \
  "$STAGE/code/deps/burgers2d-rom-latent-stepping/followup" \
  "$STAGE/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom" \
  "$STAGE/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision" \
  "$STAGE/code/smoke"

cp "$EXP/bh_2048_learned.py" "$EXP/bh_2048_learned_audit.py" "$STAGE/code/"
cp "$OLD/bh_common.py" "$OLD/bh_dynamic_correction.py" \
  "$OLD/bh_film_control.py" "$STAGE/code/"
cp "$WORKTREES/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
  "$STAGE/code/deps/burgers2d-coord-rom/"

FILM_DEP="$STAGE/code/deps/burgers2d-rom-latent-stepping"
cp "$EXP/../burgers2d-rom-latent-stepping/blat_common.py" "$FILM_DEP/"
cp "$EXP/../burgers2d-rom-latent-stepping/followup/fu_common.py" \
  "$FILM_DEP/followup/"
cp "$WORKTREES/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
  "$FILM_DEP/deps/burgers2d-coord-rom/"
cp "$WORKTREES/2026-08-14-multistage-precision/experiments/multistage-precision/ms_parametric.py" \
  "$WORKTREES/2026-08-14-multistage-precision/experiments/multistage-precision/ms_autodecoder.py" \
  "$FILM_DEP/deps/multistage-precision/"
cp "$EXP/../cost-to-tolerance/ckpt_burgers/blat_ad_N64_K8.pkl" \
  "$FILM_DEP/"
cp "$EXP/runs/smoke1/AUDIT.json" "$STAGE/code/smoke/smoke_audit.json"

cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH -o $REMOTE/logs/%j.out
#SBATCH -e $REMOTE/logs/%j.err
set -euo pipefail
cd "$REMOTE/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=$COMMIT
export FILM_CHECKPOINT=deps/burgers2d-rom-latent-stepping/blat_ad_N64_K8.pkl
export SMOKE_AUDIT=smoke/smoke_audit.json
export N=64 N_TRAIN=512 N_VAL=64 SEED=0 BC_MODE=poly K_LAT=8
export AD_HIDDEN=256 AD_LAYERS=5 GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100
export EQ_SNAPS=64 EQ_GRID_POOL=4096
echo "host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$COMMIT dirty=$DIRTY cell=$cell"
\$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
\$PY bh_2048_learned.py ../out/learned.json
echo ALL-DONE
EOF

(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 \
  -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
