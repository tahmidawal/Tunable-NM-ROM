"""Prepare the immutable Burgers FNO code bundle.

Data is never staged from this machine. The verified cluster-generated Burgers
cache is copied on the cluster into this job's own directory and re-verified
against its own recorded checksums before training.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
NAME = 'fno_burgers01'
REMOTE = f'/cluster/tufts/paralab/tawal01/no_audit_20260914/{NAME}'
CACHE = '/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01'
OUT = HERE / 'cluster/stage' / NAME
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'code').mkdir()
(OUT / 'logs').mkdir()
for path in HERE.glob('*.py'):
    shutil.copy2(path, OUT / 'code' / path.name)
shutil.copytree(HERE / 'configs', OUT / 'code/configs')
shutil.copy2(HERE / 'NEURALOPERATOR-LICENSE', OUT / 'code/NEURALOPERATOR-LICENSE')
commit = subprocess.check_output(['git', '-C', str(WT), 'rev-parse', 'HEAD'], text=True).strip()
provenance = dict(
    source_commit=commit, source_worktree=str(WT), job_directory=REMOTE,
    pde='burgers', mesh_intervals=256, output_times=[0., .05, .1, .15, .2, .25],
    data_origin=f'Burgers lane cluster-generated cache {CACHE}; copied on the cluster and re-verified',
    model_inputs='sampled initial nodal field, viscosity and coordinates only; no generation descriptors, case ids or truth sidecars',
    time_dependence='direct multi-time output: five evolved fields as output channels; the supplied initial state is returned exactly and no autoregressive rollout is used',
    capacities=['small', 'medium', 'large', 'learning-rate refinement of the best capacity'],
    epoch_budget=200, resumability='none; each training run is bounded inside this one allocation',
    gpu_hour_target=4.25, precision='float64/complex128, highest matmul precision',
    comparisons='same-job FNO timing and validation accuracy only; no cross-job speed ratio and no ROM/FOM panel')
(OUT / 'source-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
(OUT / 'run.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name=ctol_noa_fno_b01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:30:00
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
( cd data && sha256sum -c DATA.sha256 )
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}'); sys.exit(0 if b=='gpu' else 42)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
exec "$PY" code/worker_burgers.py
''')
hashes = []
for path in sorted(OUT.rglob('*')):
    if path.is_file():
        hashes.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUT)}')
(OUT / 'MANIFEST.sha256').write_text('\n'.join(hashes) + '\n')
print(OUT)
