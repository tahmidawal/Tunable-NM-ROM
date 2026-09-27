#!/bin/bash
#SBATCH -J nda_beq128ref_s0_r25
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_beq128ref_s0_r25/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_beq128ref_s0_r25/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_beq128ref_s0_r25"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "cell=nda_beq128ref_s0_r25 commit=80a36e77612aaa9bd320d1532927928ef0a7734e host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export N_TEST=16 FLOOR_BUDGET=60 GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100
export VARIANTS=lspg:full:weak128,lspg:eq544:weak128,lspg:eq576:weak128,lspg:eq608:weak128
export POD_KS=16 POD_VARIANTS= DO_TIMING=0 TR_FACTOR=0
$PY -u nda_burgers_eval.py ../ckpt/variant.pkl ../out
$PY - <<'PY'
import glob, json
p = glob.glob('../out/*.json')
assert len(p) == 1
d = json.load(open(p[0]))
want = set('lspg:full:weak128,lspg:eq544:weak128,lspg:eq576:weak128,lspg:eq608:weak128'.split(','))
assert d['backend'] == 'gpu' and set(d['rom']) == want
PY
echo ALL-DONE
