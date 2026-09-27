#!/bin/bash
#SBATCH -J nda_be2e_g160_r12
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 18:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_be2e_g160_r12/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_be2e_g160_r12/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_be2e_g160_r12"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python CTOL_COMMIT=7fa70504afdd2513c01ef95c0269c5e699c2453b
echo "cell=nda_be2e_g160_r12 commit=7fa70504afdd2513c01ef95c0269c5e699c2453b host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export N=64 KS=16 NS=64 TAUS=1e-2,1e-3 M=96 MQ=384 M_BIG=128 K_BIG=99 MQ_4M=512
export CTOL_N_TEST=16 N_POD_TRAJ=512 POD_SLICE_STRIDE=4 GEN_CHUNK=16
export TIME_REPS=9 TIME_WARM=3 BURN_IN_S=1.5 DO_SUPP=0 DO_CEILING=0
export CONFIGS=../configs.json ARM_TAG=architecture_e2e FOM_NEWTON_LADDER=3
export AD_HIDDEN=256 AD_LAYERS=5 DECODER_ARCH=film PKL_DIR=../ckpt/control
$PY -u ctol_burgers.py ../out/control.json
export AD_HIDDEN=160 AD_LAYERS=4 DECODER_ARCH=groupfilm FILM_GROUP_SIZE=2 FILM_START=0 Z_WIDTH=64 PKL_DIR=../ckpt/variant
$PY -u ctol_burgers.py ../out/variant.json
$PY - <<'PY'
import json
for name in ('control', 'variant'):
    d = json.load(open(f'../out/{name}.json'))
    assert d['complete'] and d['config']['backend'] == 'gpu'
    assert len(d['rows']) == 4 and all(r['method'] == 'coord' for r in d['rows'])
PY
echo ALL-DONE
