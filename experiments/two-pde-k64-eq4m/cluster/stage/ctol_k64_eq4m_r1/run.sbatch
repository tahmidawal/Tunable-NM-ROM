#!/bin/bash
#SBATCH -J ctol_k64_eq4m_r1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 18:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/k64-eq4m/ctol_k64_eq4m_r1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/k64-eq4m/ctol_k64_eq4m_r1/logs/%j.err
set -euo pipefail
root="/cluster/tufts/paralab/tawal01/k64-eq4m/ctol_k64_eq4m_r1"
cd "$root"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY="/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python"
export CTOL_COMMIT="ae922522f0edaba825f2d9dc23efcada8cab976f"
export CTOL_SRC_COMMITS="two-pde-k64-eq4m=ae922522f0edaba825f2d9dc23efcada8cab976f/e3b0c44298fc"
echo "cell=ctol_k64_eq4m_r1 commit=ae922522f0edaba825f2d9dc23efcada8cab976f dirty=e3b0c44298fc job=$SLURM_JOB_ID host=$(hostname)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
$PY code/make_configs.py .

# Train only the two missing latent dimensions with the frozen recipes.
for k in 48 64; do
  (cd code/deps/poisson2d-rom-objective &&     N=64 N_TRAIN=512 N_VAL=64 K_LAT=$k HARD_BC=1 STEPS=20000 BATCH=32     P_SUB=1024 HIDDEN=128 N_LAYERS=4 TRAIN_SEED=0 GN_ITERS=60     $PY -u followup/fu_train.py ../../../ckpt/poisson)
done
for k in 48 64; do
  (cd code/deps/burgers2d-rom-latent-stepping &&     N=64 N_TRAIN=512 N_VAL=64 K_LAT=$k BC_MODE=poly AD_HIDDEN=256 AD_LAYERS=5     AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 T_SMOOTH=1e-2 TRAIN_SEED=0 POD_KMAX=64     N_TEST=16 GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100     $PY -u blat_train_ad.py ../../../ckpt/burgers)
done

test -f ckpt/poisson/autodec_K48_N64_hbc_stages.pkl
test -f ckpt/poisson/autodec_K64_N64_hbc_stages.pkl
test -f ckpt/burgers/blat_ad_N64_K48.pkl
test -f ckpt/burgers/blat_ad_N64_K64.pkl

common="N=64 KS=4,6,8,12,16,24,32,48,64 TAUS=1e-3,1e-2 NS=64 DO_SUPP=0 CAP_CONTROL=0 DO_CEILING=0 TIME_REPS=9 TIME_WARM=3 BURN_IN_S=1.5 DIRECT_COMPONENT_TIMING=1 CTOL_CAND_CAP=4096 CTOL_EQ_SNAPS=64 CTOL_EQ_PERTURB=3 CTOL_EQ_ROWS=3072 CTOL_EQ_SEED=20259"
(cd code && env $common TR_FACTOR=1 CONFIGS=../poisson2d_configs.json PKL_DIR=../ckpt/poisson   POOL_CONTROL=0 DO_POD_DIRECT=0 FOM_LADDER=1e-13   $PY -u ctol_poisson.py ../out/poisson2d_k64_eq4m.json)
(cd code && env $common TR_FACTOR=0.01 CONFIGS=../burgers2d_configs.json PKL_DIR=../ckpt/burgers   CTOL_N_TEST=16 N_POD_TRAJ=64 POD_SLICE_STRIDE=4 GEN_CHUNK=16 FOM_NEWTON_LADDER=8   $PY -u ctol_burgers.py ../out/burgers2d_k64_eq4m.json)

$PY code/validate_results.py out/poisson2d_k64_eq4m.json   out/burgers2d_k64_eq4m.json out/validation.json
cp MANIFEST.sha256 out/MANIFEST.sha256
(cd out && find . -type f -not -name RESULTS.sha256 -exec sha256sum {} \; | sort > RESULTS.sha256)
echo ALL-DONE
