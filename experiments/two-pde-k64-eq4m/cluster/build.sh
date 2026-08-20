#!/usr/bin/env bash
# Build the one-job, one-GPU stage for the frozen paired k sweep.
set -euo pipefail

cell="ctol_k64_eq4m_r1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exp="$(cd "$here/.." && pwd)"
wt="$(cd "$exp/../.." && pwd)"
wts="$(cd "$wt/.." && pwd)"
stage="$here/stage/$cell"
remote_root="/cluster/tufts/paralab/tawal01/k64-eq4m"
remote="$remote_root/$cell"
cluster_py="/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python"

ctol="$wt/experiments/cost-to-tolerance"
poisson="$wt/experiments/poisson2d-rom-objective"
burgers="$wt/experiments/burgers2d-rom-latent-stepping"
nda="$wt/experiments/nonlinear-decoder-architecture"
msp="$wts/2026-08-14-multistage-precision/experiments/multistage-precision"
b2d="$wts/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom"

[[ ! -e "$stage" ]] || { echo "stage already exists: $stage" >&2; exit 2; }
for f in "$ctol/ctol_poisson.py" "$ctol/ctol_burgers.py" "$ctol/ctol_eq.py" \
         "$ctol/ctol_tol.py" "$poisson/followup/fu_train.py" "$burgers/blat_train_ad.py" \
         "$nda/nda_arch.py" "$msp/ms_parametric.py" "$msp/ms_autodecoder.py" \
         "$b2d/burgers2d_film.py"; do
  [[ -f "$f" ]] || { echo "missing source: $f" >&2; exit 3; }
done

mkdir -p "$stage/code/deps/poisson2d-rom-objective/followup" \
  "$stage/code/deps/poisson2d-rom-objective/deps" \
  "$stage/code/deps/burgers2d-rom-latent-stepping/followup" \
  "$stage/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision" \
  "$stage/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom" \
  "$stage/code/deps/nonlinear-decoder-architecture" \
  "$stage/ckpt/poisson" "$stage/ckpt/burgers" "$stage/out" "$stage/logs"

cp "$ctol"/ctol_*.py "$stage/code/"
cp "$exp/make_configs.py" "$exp/validate_results.py" "$stage/code/"
cp "$exp/EXPERIMENT.md" "$stage/"

cp "$poisson/pro_common.py" "$stage/code/deps/poisson2d-rom-objective/"
cp "$poisson/followup/fu_eq.py" "$poisson/followup/fu_style.py" \
  "$poisson/followup/fu_train.py" \
  "$stage/code/deps/poisson2d-rom-objective/followup/"

cp "$burgers/blat_common.py" "$burgers/blat_train_ad.py" \
  "$stage/code/deps/burgers2d-rom-latent-stepping/"
cp "$burgers/followup/fu_common.py" "$burgers/followup/fu_style.py" \
  "$stage/code/deps/burgers2d-rom-latent-stepping/followup/"

cp "$nda/nda_arch.py" "$stage/code/deps/nonlinear-decoder-architecture/"
cp "$msp/ms_parametric.py" "$msp/ms_autodecoder.py" \
  "$stage/code/deps/poisson2d-rom-objective/deps/"
cp "$msp/ms_parametric.py" "$msp/ms_autodecoder.py" \
  "$stage/code/deps/burgers2d-rom-latent-stepping/deps/multistage-precision/"
cp "$b2d/burgers2d_film.py" \
  "$stage/code/deps/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/"

for k in 4 6 8 12 16 24 32; do
  cp "$ctol/ckpt_poisson/autodec_K${k}_N64_hbc_stages.pkl" "$stage/ckpt/poisson/"
  cp "$ctol/ckpt_burgers/blat_ad_N64_K${k}.pkl" "$stage/ckpt/burgers/"
done

commit="$(git -C "$wt" rev-parse HEAD)"
dirty="$(git -C "$wt" status --porcelain --untracked-files=no | sha256sum | cut -c1-12)"
cat > "$stage/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $cell
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 18:00:00
#SBATCH -o $remote/logs/%j.out
#SBATCH -e $remote/logs/%j.err
set -euo pipefail
root="$remote"
cd "\$root"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY="$cluster_py"
export CTOL_COMMIT="$commit"
export CTOL_SRC_COMMITS="two-pde-k64-eq4m=$commit/$dirty"
echo "cell=$cell commit=$commit dirty=$dirty job=\$SLURM_JOB_ID host=\$(hostname)"
echo "gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
\$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
\$PY code/make_configs.py .

# Train only the two missing latent dimensions with the frozen recipes.
for k in 48 64; do
  (cd code/deps/poisson2d-rom-objective && \
    N=64 N_TRAIN=512 N_VAL=64 K_LAT=\$k HARD_BC=1 STEPS=20000 BATCH=32 \
    P_SUB=1024 HIDDEN=128 N_LAYERS=4 TRAIN_SEED=0 GN_ITERS=60 \
    \$PY -u followup/fu_train.py ../../../ckpt/poisson)
done
for k in 48 64; do
  (cd code/deps/burgers2d-rom-latent-stepping && \
    N=64 N_TRAIN=512 N_VAL=64 K_LAT=\$k BC_MODE=poly AD_HIDDEN=256 AD_LAYERS=5 \
    AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 T_SMOOTH=1e-2 TRAIN_SEED=0 POD_KMAX=64 \
    N_TEST=16 GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100 \
    \$PY -u blat_train_ad.py ../../../ckpt/burgers)
done

test -f ckpt/poisson/autodec_K48_N64_hbc_stages.pkl
test -f ckpt/poisson/autodec_K64_N64_hbc_stages.pkl
test -f ckpt/burgers/blat_ad_N64_K48.pkl
test -f ckpt/burgers/blat_ad_N64_K64.pkl

common="N=64 KS=4,6,8,12,16,24,32,48,64 TAUS=1e-3,1e-2 NS=64 DO_SUPP=0 CAP_CONTROL=0 DO_CEILING=0 TIME_REPS=9 TIME_WARM=3 BURN_IN_S=1.5 DIRECT_COMPONENT_TIMING=1 CTOL_CAND_CAP=4096 CTOL_EQ_SNAPS=64 CTOL_EQ_PERTURB=3 CTOL_EQ_ROWS=3072 CTOL_EQ_SEED=20259"
(cd code && env \$common TR_FACTOR=1 CONFIGS=../poisson2d_configs.json PKL_DIR=../ckpt/poisson \
  POOL_CONTROL=0 DO_POD_DIRECT=0 FOM_LADDER=1e-13 \
  \$PY -u ctol_poisson.py ../out/poisson2d_k64_eq4m.json)
(cd code && env \$common TR_FACTOR=0.01 CONFIGS=../burgers2d_configs.json PKL_DIR=../ckpt/burgers \
  CTOL_N_TEST=16 N_POD_TRAJ=64 POD_SLICE_STRIDE=4 GEN_CHUNK=16 FOM_NEWTON_LADDER=8 \
  \$PY -u ctol_burgers.py ../out/burgers2d_k64_eq4m.json)

\$PY code/validate_results.py out/poisson2d_k64_eq4m.json \
  out/burgers2d_k64_eq4m.json out/validation.json
cp MANIFEST.sha256 out/MANIFEST.sha256
(cd out && find . -type f -not -name RESULTS.sha256 -exec sha256sum {} \; | sort > RESULTS.sha256)
echo ALL-DONE
EOF

(cd "$stage" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; \
  | sort > MANIFEST.sha256)
echo "$stage"
