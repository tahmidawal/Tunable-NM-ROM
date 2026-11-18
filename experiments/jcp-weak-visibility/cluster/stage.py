"""Prepare one immutable local cluster payload, without submitting anything."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

lane = Path(__file__).resolve().parents[1]
attempt = sys.argv[1]
assert re.fullmatch('[a-z][a-z0-9]+', attempt)
config_name = sys.argv[2] if len(sys.argv) > 2 else 'pilot.json'
assert re.fullmatch(r'[a-z0-9_]+\.json', config_name) and (lane/'configs'/config_name).is_file()
target = lane/'runs'/attempt
target.mkdir(exist_ok=False)
code = target/'code'
code.mkdir()
for name in ['core.py','run.py','fixture.py','cohort_guard.py','PROTOCOL.md','PROVENANCE.json','OWNER.json',
             'configs/'+config_name,'vendor/engines.py','vendor/sep_common.py',
             'vendor/legacy_engines.py','vendor/legacy_qcore.py',
             'inputs/frozen.pkl','inputs/rotation_R512.npz']:
    dest = code/name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lane/name, dest)
for amendment in lane.glob('AMENDMENT-*.md'):
    shutil.copy2(amendment, code/amendment.name)
commit = subprocess.check_output(['git','-c','gc.auto=0','rev-parse','HEAD'], cwd=lane, text=True).strip()
batch = f'''#!/usr/bin/env bash
#SBATCH --job-name=ctol_jwv_{attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=job.log
#SBATCH --error=job.log
set -euo pipefail
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export SOURCE_COMMIT={commit}
export TMPDIR="$SLURM_SUBMIT_DIR/tmp"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/cache"
export CUDA_CACHE_PATH="$SLURM_SUBMIT_DIR/cache/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd "$SLURM_SUBMIT_DIR/code"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}'); sys.exit(0 if b=='gpu' else 42)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
mkdir -p checks
$PY -u fixture.py
$PY -u run.py --config configs/{config_name} --output ../output
'''
(target/'run.sbatch').write_text(batch)
(target/'reservation.json').write_text(json.dumps(dict(attempt=attempt, ceiling_allocated_gpu_seconds=2700,
    lane_ceiling_allocated_gpu_seconds=10800, source_commit=commit, owner='weak_visibility_run'), indent=2)+'\n')
files = sorted(p for p in target.rglob('*') if p.is_file())
(target/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}\n' for p in files))
print(target)
