#!/usr/bin/env bash
# make_cell.sh <cell> <memory> <walltime> <env assignments> <command> [gpu type]
set -euo pipefail
cell="$1"
memory="$2"
walltime="$3"
envs="$4"
command="$5"
gpu_type="${6:-h100}"
if [[ "$cell" == ctol_hybb_* ]]; then
  job_name="$cell"
else
  job_name="hybb_$cell"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WORKTREES="$(cd "$EXP/../../.." && pwd)"
REMOTE="/cluster/tufts/paralab/tawal01/hybb1024/$cell"
STAGE="$HERE/stage/$cell"
COMMIT="$(git -C "$EXP" rev-parse HEAD)"
DIRTY="$(git -C "$EXP" status --porcelain -- . | sha256sum | cut -c1-12)"

# This removes only a local, cell-specific staging directory, never results.
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/burgers2d-coord-rom"
cp "$EXP"/bh_*.py "$STAGE/code/"
cp "$WORKTREES/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
  "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$WORKTREES/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/sweep/burgers2d_film_N64.pkl" \
  "$STAGE/code/deps/burgers2d-coord-rom/"

# The final genuine-NM-ROM negative control reuses the audited K=8 FiLM
# checkpoint and weak-EQ implementation.  Stage every dependency by content so
# the manifest, rather than an ancestor git repository, fixes provenance.
FILM_DEP="$STAGE/code/deps/burgers2d-rom-latent-stepping"
mkdir -p "$FILM_DEP/followup" "$FILM_DEP/deps/burgers2d-coord-rom" \
  "$FILM_DEP/deps/multistage-precision"
cp "$EXP/../burgers2d-rom-latent-stepping/blat_common.py" "$FILM_DEP/"
cp "$EXP/../burgers2d-rom-latent-stepping/followup/fu_common.py" "$FILM_DEP/followup/"
cp "$WORKTREES/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
  "$FILM_DEP/deps/burgers2d-coord-rom/"
cp "$WORKTREES/2026-08-14-multistage-precision/experiments/multistage-precision/ms_parametric.py" \
  "$WORKTREES/2026-08-14-multistage-precision/experiments/multistage-precision/ms_autodecoder.py" \
  "$FILM_DEP/deps/multistage-precision/"
cp "$WORKTREES/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/followup/cluster/stage/bt_n/code/in/blat_ad_N64_K8.pkl" \
  "$FILM_DEP/"

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
export BH_COMMIT=$COMMIT
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
