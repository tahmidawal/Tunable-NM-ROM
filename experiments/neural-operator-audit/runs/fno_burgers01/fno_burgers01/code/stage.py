"""Prepare an immutable code bundle; data is generated/copied only on cluster."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
NAME = 'fno_poisson01'
REMOTE = f'/cluster/tufts/paralab/tawal01/no_audit_20260914/{NAME}'
OUT = HERE/'cluster/stage'/NAME
OUT.mkdir(parents=True, exist_ok=False)
(OUT/'code').mkdir()
(OUT/'logs').mkdir()
for path in HERE.glob('*.py'):
    shutil.copy2(path, OUT/'code'/path.name)
shutil.copytree(HERE/'configs', OUT/'code/configs')
shutil.copy2(HERE/'NEURALOPERATOR-LICENSE', OUT/'code/NEURALOPERATOR-LICENSE')
commit = subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'], text=True).strip()
provenance = dict(source_commit=commit, source_worktree=str(WT), job_directory=REMOTE,
                 data_origin='Poisson pilot01 generated on cluster; copied with verified source indices and artifact hashes',
                 gpu_hour_limit=7, sequential_per_model_wall_seconds=7200,
                 precision='float64/complex128, highest matmul precision',
                 comparisons='training capacity screen only; no cross-job speed ratios')
(OUT/'source-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
(OUT/'run.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name=ctol_noa_fno_p01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output={REMOTE}/logs/%j.out
#SBATCH --error={REMOTE}/logs/%j.err
set -euo pipefail
cd {REMOTE}
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TMPDIR={REMOTE}/tmp
export XDG_CACHE_HOME={REMOTE}/cache
export TORCH_HOME={REMOTE}/cache/torch
export MPLCONFIGDIR={REMOTE}/cache/matplotlib
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
sha256sum -c MANIFEST.sha256
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}'); sys.exit(0 if b=='gpu' else 42)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
exec "$PY" code/worker.py
''')
hashes = []
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        hashes.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}')
(OUT/'MANIFEST.sha256').write_text('\n'.join(hashes)+'\n')
print(OUT)
