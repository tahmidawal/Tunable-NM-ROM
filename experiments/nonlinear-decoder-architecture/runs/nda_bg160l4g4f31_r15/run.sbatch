#!/bin/bash
#SBATCH -J nda_bg160l4g4f31_r15
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 18:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_bg160l4g4f31_r15/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_bg160l4g4f31_r15/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_bg160l4g4f31_r15"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "cell=nda_bg160l4g4f31_r15 commit=ba99cb2210a8966312766b59fe7d393fda03e0f5 dirty_hash=99981e6d08ef host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code/burgers
export N=64 N_TRAIN=512 N_VAL=64 N_TEST=16 K_LAT=16
export AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 T_SMOOTH=1e-2 LAT_REG=1e-4 LAT_LR=5e-3 PEAK_LR=2e-3
export TRAIN_SEED=0
export AD_HIDDEN=160 AD_LAYERS=4 AD_N_FREQ=31
export DECODER_ARCH=groupfilm FILM_GROUP_SIZE=4 FILM_START=0 Z_WIDTH=64
export GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100 FLOOR_BUDGET=60
export VARIANTS=lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64
export POD_KS=16 POD_VARIANTS= DO_TIMING=1 TIME_REPS=7
$PY -u blat_train_ad.py ../../out
$PY -u blat_rom.py ../../out/blat_ad_N64_K16.pkl ../../out
$PY -c "import json; d=json.load(open('../../out/blat_rom_N64_K16.json')); assert d['backend']=='gpu' and set(d['rom'])==set('lspg:full:weak64,lspg:eq256:weak64,lspg:eq512:weak64'.split(','))"
echo ALL-DONE
