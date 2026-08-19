#!/usr/bin/env bash
# make_job.sh <cell> <walltime> "<env assignments>"
# Builds an isolated stage directory for /cluster/tufts/paralab/tawal01/hybp1024/<cell>.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: make_job.sh <cell> <HH:MM:SS> '<env assignments>'" >&2; exit 2; }
cell="$1"; walltime="$2"; envs="$3"
[[ "$cell" =~ ^[a-z0-9_]+$ ]] || { echo "unsafe cell name: $cell" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
EXPS="$(dirname "$EXP")"
WT="$(cd "$EXPS/../.." && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/hybp1024/$cell"
COMMIT="$(git -C "$EXP" rev-parse HEAD)"
DIRTY_HASH="$(git -C "$EXP" status --porcelain -- . | sha256sum | cut -c1-12)"

# This generated local staging directory is owned solely by this cell.
rm -rf "$STAGE"
mkdir -p "$STAGE/code/poisson-hybrid-1024/input" \
         "$STAGE/code/poisson-hybrid-1024/deps" \
         "$STAGE/code/poisson2d-rom-objective/followup" \
         "$STAGE/code/poisson2d-rom-objective/deps" \
         "$STAGE/code/rom-warmstart-fom" "$STAGE/out" "$STAGE/logs"

cp "$EXP/feasibility.py" "$STAGE/code/poisson-hybrid-1024/"
cp "$EXPS/poisson2d-rom-objective/pro_common.py" \
   "$STAGE/code/poisson2d-rom-objective/"
cp "$EXPS/poisson2d-rom-objective/followup/fu_eq.py" \
   "$STAGE/code/poisson2d-rom-objective/followup/"
cp "$EXPS/rom-warmstart-fom/wsf_util.py" "$EXPS/rom-warmstart-fom/wsf_poisson.py" \
   "$STAGE/code/rom-warmstart-fom/"
NDA="$WT/../2026-08-19-nonlinear-decoder-architecture/experiments/nonlinear-decoder-architecture"
cp "$NDA/nda_arch.py" "$STAGE/code/poisson-hybrid-1024/deps/"
MSP="$WT/2026-08-14-multistage-precision/experiments/multistage-precision"
cp "$MSP/ms_parametric.py" "$MSP/ms_autodecoder.py" \
   "$STAGE/code/poisson2d-rom-objective/deps/"
cp "$EXPS/cost-to-tolerance/ckpt_poisson/autodec_K8_N64_hbc_stages.pkl" \
   "$STAGE/code/poisson-hybrid-1024/input/auto.pkl"
cp "$MSP/runs/par512_bw/ms_parametric_stages.pkl" \
   "$STAGE/code/poisson-hybrid-1024/input/param.pkl"
cp "$NDA/runs/nda_pg98l4g2_r6/out/autodec_K16_N64_hbc_stages.pkl" \
   "$STAGE/code/poisson-hybrid-1024/input/groupfilm.pkl"

cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J ctol_hybp_${cell}
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t $walltime
#SBATCH -o $REMOTE/logs/%j.out
#SBATCH -e $REMOTE/logs/%j.err
set -euo pipefail
cd "$REMOTE/code/poisson-hybrid-1024"
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSF_COMMIT=$COMMIT
echo "host=\$(hostname) job=\${SLURM_JOB_ID} gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$COMMIT dirty_hash=$DIRTY_HASH cell=$cell"
\$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
export PKL=input/auto.pkl
export PARAM_PKL=input/param.pkl
export GROUP_PKL=input/groupfilm.pkl
export $envs
\$PY feasibility.py "$REMOTE/out/$cell.json"
echo ALL-DONE
EOF

(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort \
  > MANIFEST.sha256)
echo "$STAGE"
