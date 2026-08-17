#!/usr/bin/env bash
# make_cell.sh <cell-name> <mem> <hours> "<env assignments>" "<command line...>"
#
# Emits cluster/stage/<cell>/ with a MIRROR of the experiments/ tree (so every
# reference harness resolves its own deps exactly as it does locally), the
# checkpoints, and run.sbatch for /cluster/tufts/paralab/tawal01/wsfom/<cell>/.
# ONE JOB PER DIRECTORY.
set -euo pipefail
cell="$1"; mem="$2"; hours="$3"; envs="$4"; cmd="$5"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"                       # experiments/rom-warmstart-fom
EXPS="$(dirname "$EXP")"                       # experiments/
WT="$(cd "$EXPS/../.." && pwd)"                # worktrees/
REMOTE="/cluster/tufts/paralab/tawal01/wsfom/$cell"
STAGE="$HERE/stage/$cell"
COMMIT="$(git -C "$EXP" rev-parse HEAD)"
DIRTY="$(git -C "$EXP" status --porcelain -- . | sha256sum | cut -c1-12)"

rm -rf "$STAGE"; mkdir -p "$STAGE/logs" "$STAGE/out"
C="$STAGE/code"
mkdir -p "$C/rom-warmstart-fom/in" \
         "$C/poisson2d-rom-objective/followup" "$C/poisson2d-rom-objective/deps" \
         "$C/burgers2d-rom-latent-stepping/followup" \
         "$C/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom" \
         "$C/burgers2d-rom-latent-stepping/deps/multistage-precision"

cp "$EXP"/wsf_*.py "$C/rom-warmstart-fom/"
cp "$EXP"/in/*.pkl "$C/rom-warmstart-fom/in/"
cp "$EXPS"/poisson2d-rom-objective/pro_common.py "$C/poisson2d-rom-objective/"
cp "$EXPS"/poisson2d-rom-objective/followup/fu_eq.py "$C/poisson2d-rom-objective/followup/"
cp "$EXPS"/burgers2d-rom-latent-stepping/blat_common.py "$C/burgers2d-rom-latent-stepping/"
cp "$EXPS"/burgers2d-rom-latent-stepping/followup/fu_common.py "$C/burgers2d-rom-latent-stepping/followup/"
MSP="$WT/2026-08-14-multistage-precision/experiments/multistage-precision"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" "$C/poisson2d-rom-objective/deps/"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" "$C/burgers2d-rom-latent-stepping/deps/multistage-precision/"
cp "$WT/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
   "$C/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/"

cat > "$STAGE/run.sbatch" <<EOF
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
cd "$REMOTE/code/rom-warmstart-fom"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$COMMIT  dirty=$DIRTY  cell=$cell"
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
EOF

(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "$STAGE"
