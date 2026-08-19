#!/bin/bash
#SBATCH -J nda_btrust160_r13
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_btrust160_r13/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_btrust160_r13/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_btrust160_r13"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "cell=nda_btrust160_r13 commit=886434a685f68426e7970c4ab1d876cf7ba97d0d host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export N_TEST=16 FLOOR_BUDGET=60 GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100
export VARIANTS=lspg:full:weak96,lspg:eq384:weak96,lspg:full:weak128,lspg:eq512:weak128
export POD_KS=16 POD_VARIANTS= DO_TIMING=0 TR_FACTOR=0.01
$PY -u nda_burgers_eval.py ../ckpt/control.pkl ../out/control_tr001
$PY -u nda_burgers_eval.py ../ckpt/variant.pkl ../out/variant_tr001
$PY - <<'PY'
import glob, json
p = glob.glob('../out/*/*.json')
assert len(p) == 2
for f in p:
    d = json.load(open(f))
    assert d['backend'] == 'gpu'
    assert set(d['rom']) == {'lspg:full:weak96', 'lspg:eq384:weak96',
                             'lspg:full:weak128', 'lspg:eq512:weak128'}
    assert d['config']['tr_factor'] == 0.01
PY
echo ALL-DONE
