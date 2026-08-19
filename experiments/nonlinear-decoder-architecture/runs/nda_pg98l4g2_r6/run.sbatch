#!/bin/bash
#SBATCH -J nda_pg98l4g2_r6
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pg98l4g2_r6/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pg98l4g2_r6/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pg98l4g2_r6"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "cell=nda_pg98l4g2_r6 commit=af376d46c68e133187743a446448fca87cfae365 dirty_hash=4b00bb1ca6a8 host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code/poisson
export N=64 N_TRAIN=512 N_VAL=64 STEPS=20000 BATCH=32 P_SUB=1024
export K_LAT=16 HARD_BC=1 GN_ITERS=60 TRAIN_SEED=0
export HIDDEN=98 N_LAYERS=4 DECODER_ARCH=groupfilm FILM_GROUP_SIZE=2 FILM_START=0 Z_WIDTH=64
$PY -u followup/fu_train.py ../../out
export PKL=../../out/autodec_K16_N64_hbc_stages.pkl
export N_TEST=16 OBJECTIVES=weak_a1_M128 MS=512 SCHEMES=full,nnls INITS=nearest
export EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1
$PY -u pro_colloc.py ../../out/rom.json
$PY -c "import json; d=json.load(open('../../out/rom.json')); assert d['complete'] and d['manifest']['backend']=='gpu' and len(d['rows'])==2"
echo ALL-DONE
