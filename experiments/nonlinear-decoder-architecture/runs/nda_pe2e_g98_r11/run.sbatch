#!/bin/bash
#SBATCH -J nda_pe2e_g98_r11
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pe2e_g98_r11/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pe2e_g98_r11/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_pe2e_g98_r11"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python CTOL_COMMIT=7099d83ccf9c758feaa7e34eba48d94f817ea622
echo "cell=nda_pe2e_g98_r11 commit=7099d83ccf9c758feaa7e34eba48d94f817ea622 host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export KS=16 NS=64 TAUS=1e-2,1e-3 M=128 MQ=512 M_BIG=128 K_BIG=99 MQ_4M=512
export N_TEST=16 GN_ITERS=60 TIME_REPS=9 TIME_WARM=3 BURN_IN_S=1.5
export DO_SUPP=0 POOL_CONTROL=0 CAP_CONTROL=0 DO_POD_DIRECT=0 DO_CEILING=0
export CONFIGS=../configs.json ARM_TAG=architecture_e2e FOM_LADDER=1e-6
export HIDDEN=128 N_LAYERS=4 DECODER_ARCH=film PKL_DIR=../ckpt/control
$PY -u ctol_poisson.py ../out/control.json
export HIDDEN=98 N_LAYERS=4 DECODER_ARCH=groupfilm FILM_GROUP_SIZE=2 FILM_START=0 Z_WIDTH=64 PKL_DIR=../ckpt/variant
$PY -u ctol_poisson.py ../out/variant.json
$PY - <<'PY'
import json
for name in ('control', 'variant'):
    d = json.load(open(f'../out/{name}.json'))
    assert d['complete'] and d['config']['backend'] == 'gpu'
    assert len(d['rows']) == 2 and all(r['method'] == 'coord' for r in d['rows'])
PY
echo ALL-DONE
