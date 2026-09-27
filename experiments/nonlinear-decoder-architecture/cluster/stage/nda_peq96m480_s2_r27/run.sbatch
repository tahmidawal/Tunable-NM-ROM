#!/bin/bash
#SBATCH -J nda_peq96m480_s2_r27
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_peq96m480_s2_r27/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_peq96m480_s2_r27/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/nda_peq96m480_s2_r27"
sha256sum -c MANIFEST.sha256
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "cell=nda_peq96m480_s2_r27 commit=a8867b41cc46ca1498ee920c68f67297fab2e95f host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
df -h /cluster/tufts/paralab/tawal01
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
cd code
export N_TEST=16 GN_ITERS=60 OBJECTIVES=weak_a1_M96 MS=480 SCHEMES=full,nnls INITS=nearest
export EQ_SNAPS=64 EQ_PERTURB=3 EQ_ROWS=3072 EQ_FIXED_SNAPS=1 TR_FACTOR=0
$PY -u nda_poisson_eval.py ../ckpt/variant.pkl ../out/rom.json
$PY - <<'PY'
import json
d = json.load(open('../out/rom.json'))
assert d['complete'] and d['manifest']['backend'] == 'gpu' and len(d['rows']) == 2
assert {r['scheme'] for r in d['rows']} == {'full', 'nnls'}
PY
echo ALL-DONE
