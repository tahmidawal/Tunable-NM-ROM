#!/usr/bin/env bash
# make_job_2048.sh <cell> <smoke|final>
# Builds an isolated stage for /cluster/tufts/paralab/tawal01/hybp2048/<cell>.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: make_job_2048.sh <cell> <smoke|final>" >&2; exit 2; }
cell="$1"; kind="$2"
[[ "$cell" =~ ^[a-z0-9_]+$ ]] || { echo "unsafe cell name: $cell" >&2; exit 2; }
case "$kind" in
  smoke)
    [[ "$cell" = "n2048smoke1" ]] || { echo "smoke cell must be n2048smoke1" >&2; exit 2; }
    walltime=01:00:00
    envs="POISSON_2048_KIND=smoke SMOKE=1 NS=2048 FOM_TAUS=1e-6,1e-8,1e-10 N_TEST=1 N_TIME=1 TIME_REPS=2 TIME_WARM=1 BURN_S=0 TEST_SEED=13579 M=64 MQ=256 GN_ITERS=60 LM_ROM_TAU=0.01 TR_SCALE=1 CG_MAXITER=50000 EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_CAND_OFF=4096 ARMS=lmtrmean_c64_q0,spectral_q1024,spectral_q2048 NATIVE_ARMS= PAIRWISE_DIAGNOSTIC=0 BALANCED_PAIR_ARM=lmtrmean_c64_q0 BALANCED_CONTROL_REPS=10 BOOTSTRAP_REPS=100"
    ;;
  final)
    [[ "$cell" = "n2048final1" ]] || { echo "final cell must be n2048final1" >&2; exit 2; }
    walltime=04:00:00
    envs="POISSON_2048_KIND=final SMOKE=0 NS=2048 FOM_TAUS=1e-6,1e-8,1e-10 N_TEST=16 N_TIME=8 TIME_REPS=12 TIME_WARM=3 BURN_S=3 TEST_SEED=20260826 M=64 MQ=256 GN_ITERS=60 LM_ROM_TAU=0.01 TR_SCALE=1 CG_MAXITER=50000 EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_CAND_OFF=4096 ARMS=lmtrmean_c64_q0,spectral_q1024,spectral_q2048 NATIVE_ARMS= PAIRWISE_DIAGNOSTIC=0 BALANCED_PAIR_ARM=lmtrmean_c64_q0 BALANCED_CONTROL_REPS=10 BOOTSTRAP_REPS=10000"
    ;;
  *) echo "kind must be smoke or final" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
EXPS="$(dirname "$EXP")"
WT="$(cd "$EXPS/../.." && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/hybp2048/$cell"
COMMIT="$(git -C "$EXP" rev-parse HEAD)"
DIRTY="$(git -C "$EXP" status --porcelain -- .)"
[[ -z "$DIRTY" ]] || { echo "refusing to stage dirty experiment tree" >&2; exit 2; }
DIRTY_HASH="$(printf '%s' "$DIRTY" | sha256sum | cut -c1-12)"
if [[ "$kind" = final ]]; then
  /home/tahmid/Dev/.venv/bin/python "$EXP/audit_2048.py" smoke-license \
    "$EXP/runs/n2048smoke1/out/n2048smoke1.json"
fi

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
NDA="$WT/2026-08-19-nonlinear-decoder-architecture/experiments/nonlinear-decoder-architecture"
cp "$NDA/nda_arch.py" "$STAGE/code/poisson-hybrid-1024/deps/"
cp "$EXP/transport_arch.py" "$STAGE/code/poisson-hybrid-1024/deps/"
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
#SBATCH -J ctol_hybp2_${cell}
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
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
echo "host=\$(hostname) job=\${SLURM_JOB_ID} gpu=\$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "commit=$COMMIT dirty_hash=$DIRTY_HASH cell=$cell namespace=hybp2048"
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
