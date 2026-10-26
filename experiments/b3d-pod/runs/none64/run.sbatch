#!/usr/bin/env bash
#SBATCH --job-name=b3dpod_none64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:24:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b3d-pod-20261026/none64/slurm-%j.log
set -euo pipefail
cd /cluster/tufts/paralab/tawal01/b3d-pod-20261026/none64
export PYTHONDONTWRITEBYTECODE=1
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
export TMPDIR=/cluster/tufts/paralab/tawal01/b3d-pod-20261026/none64/tmp
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/b3d-pod-20261026/none64/cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
sha256sum -c source.tar.sha256
tar -xf source.tar
$PY - <<'PY'
from pathlib import Path
import hashlib,json
p=json.loads(Path('PROVENANCE.json').read_text())
for f,h in p['files'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
print('source_provenance=verified',flush=True)
PY
mkdir artifacts
cp /cluster/tufts/paralab/tawal01/burgers-archive/jcp-hr-baselines/t3d64b/jobdir/output/artifacts/Phi.npz artifacts/
cp /cluster/tufts/paralab/tawal01/burgers-archive/jcp-hr-baselines/t3d64b/jobdir/output/artifacts/HR~ecsw~512~k8.npz artifacts/
cp /cluster/tufts/paralab/tawal01/burgers-archive/jcp-hr-baselines/t3d64b/jobdir/output/artifacts/HR~ecsw~512~k16.npz artifacts/
$PY experiments/b3d-pod/validate.py --config experiments/b3d-pod/configs/none64.json --artifacts artifacts --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
