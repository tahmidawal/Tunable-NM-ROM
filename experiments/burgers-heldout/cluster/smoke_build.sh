#!/bin/bash
# Local GB10 smoke of the bh1 build chain on a STAGED directory (cluster/stage.py <attempt> build),
# same six steps as run.sbatch with tiny sizes (48-node grid, 30 trajectories, 1200 states, 300 head
# steps, 4 direction trajectories). Plumbing only: no number from it is a result.
set -euo pipefail
D=${1:?staged dir}
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
cd "$D"; mkdir -p output
EXPPATH=$(ls -d "$PWD"/exp/experiments/*/ "$PWD"/exp/experiments/b-panel/speed | tr '\n' ':')
EXPPATH="$EXPPATH$PWD/exp/experiments/burgers-heldout"
L=exp/experiments/burgers-heldout
PYTHONPATH="$EXPPATH" jaxrun $PY $L/bh_compress.py seed1024 --blocks in/burgers2d_cat1024.pkl \
  --incumbent in/sep_hfit_dense_mid_N256_dense.pkl --out in/cat1024_seed.pkl
( cd code && env N=48 K=16 R=1024 MAX_SNAPS=1200 T_EARLY=5 N_TEST=2 SEED0=0 LOOSE=1 N_TRAJ=30 \
  EXTRA_SEED=1000 EXTRA_TRAJ=8 GEN_CHUNK=16 PROJ_CHUNK=256 IDENT_ROWS=16 \
  CKPT="$D/in/cat1024_seed.pkl" OUT_PREFIX="$D/output/x1024_" jaxrun $PY sep_coeff_extract.py )
PYTHONPATH="$EXPPATH" jaxrun $PY $L/bh_compress.py compress --npz output/x1024_sep_coeff_N48_K16_R1024.npz \
  --blocks in/burgers2d_cat1024.pkl --incumbent in/sep_hfit_dense_mid_N256_dense.pkl --out output/compress \
  --floor-mesh 48
( cd code && env N=48 K=16 R=512 MAX_SNAPS=1200 T_EARLY=5 N_TEST=2 SEED0=0 LOOSE=1 N_TRAJ=30 \
  EXTRA_SEED=1000 EXTRA_TRAJ=8 GEN_CHUNK=16 PROJ_CHUNK=256 IDENT_ROWS=16 \
  CKPT="$D/output/compress/cpod512_seed.pkl" OUT_PREFIX="$D/output/x512_" jaxrun $PY sep_coeff_extract.py )
( cd code && env NPZ="$D/output/x512_sep_coeff_N48_K16_R512.npz" CKPT="$D/output/compress/cpod512_seed.pkl" \
  OUT="$D/output/hfit_bh.json" ARMS=mid STEPS=300 BATCH=256 LR=1e-3 TIME_CAP=60 ORACLE_ITERS=20 \
  CODEDIAG_N=64 CODEDIAG_ITERS=10 ENC_STEPS=200 SEED0=0 EMIT=mid EMIT_PATH="$D/output/bh_model.pkl" \
  jaxrun $PY sep_hfit_run.py )
PYTHONPATH="$EXPPATH" jaxrun $PY $L/bh_dirs.py --checkpoint output/bh_model.pkl --out output/dirs --smoke 48
echo SMOKE-DONE
