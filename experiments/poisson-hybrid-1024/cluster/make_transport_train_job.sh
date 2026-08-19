#!/usr/bin/env bash
# make_transport_train_job.sh <cell> <walltime> "<env assignments>"
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 <cell> <HH:MM:SS> '<env assignments>'" >&2; exit 2; }
cell="$1"; walltime="$2"; envs="$3"
[[ "$cell" =~ ^[a-z0-9_]+$ ]] || { echo "unsafe cell name: $cell" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
EXPS="$(dirname "$EXP")"
WORKTREES="$(cd "$EXPS/../.." && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/hybp1024/$cell"
COMMIT="$(git -C "$EXP" rev-parse HEAD)"
DIRTY_HASH="$(git -C "$EXP" status --porcelain -- . | sha256sum | cut -c1-12)"

rm -rf "$STAGE"
mkdir -p "$STAGE/code/deps" "$STAGE/out" "$STAGE/logs"
cp "$EXP/train_transport.py" "$EXP/transport_arch.py" "$STAGE/code/"
cp "$WORKTREES/2026-08-14-multistage-precision/experiments/multistage-precision/ms_parametric.py" \
   "$STAGE/code/deps/"
cp "$EXPS/rom-warmstart-fom/wsf_util.py" "$STAGE/code/deps/"

cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J ctol_hybp_${cell}
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t $walltime
#SBATCH -o $REMOTE/logs/%j.out
#SBATCH -e $REMOTE/logs/%j.err
set -euo pipefail
cd "$REMOTE/code"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=$COMMIT
echo "host=\$(hostname) job=\${SLURM_JOB_ID} gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$COMMIT dirty_hash=$DIRTY_HASH cell=$cell"
\$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export $envs
\$PY train_transport.py "$REMOTE/out"
echo ALL-DONE
EOF

(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort \
  > MANIFEST.sha256)
echo "$STAGE"
