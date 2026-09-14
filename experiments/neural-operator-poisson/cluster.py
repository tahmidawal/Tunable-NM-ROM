"""Content-proven staging; one directory and one bounded allocation per label."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from dataset import ROOT, HERE, sha_file
NAMESPACE='/cluster/tufts/paralab/tawal01/no_poisson_20260914'


def stage(label):
    if not label.replace('_','').isalnum(): raise ValueError('Alphanumeric label required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dest=HERE/'stages'/label; dest.mkdir(parents=True,exist_ok=False)
    paths=[HERE/name for name in ['dataset.py','diagnose.py','protocol.json']]
    native=ROOT/'experiments/multiresolution-poisson'
    paths += [native/name for name in ['core.py','correction_core.py','speed_core.py','kernel_solver.py','iterative_core.py']]
    paths += [ROOT/'experiments/separable-decoder/sep_common.py',ROOT/'experiments/cost-to-tolerance/ctol_tol.py',ROOT/'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py']
    cfg=json.loads((HERE/'protocol.json').read_text())
    paths += [ROOT/cfg['checkpoint'],ROOT/cfg['basis']]
    hashes={}
    for source in paths:
        relative=source.relative_to(ROOT)
        payload=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT)
        assert payload==source.read_bytes(),f'Uncommitted source: {relative}'
        target=dest/'source'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload); hashes[str(target.relative_to(dest))]=sha_file(target)
    remote=f'{NAMESPACE}/{label}'
    script=f'''#!/bin/bash
#SBATCH --job-name=ctol_nop_{label}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export CAMPAIGN_SOURCE_COMMIT={commit}
export MPLCONFIGDIR={remote}/cache/matplotlib
export XDG_CACHE_HOME={remote}/cache
cd {remote}
sha256sum -c MANIFEST.sha256 --quiet
finish() {{
 status=$?
 echo "$status" > EXIT_CODE
 find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
 exit "$status"
}}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u source/experiments/neural-operator-poisson/dataset.py calibrate --output out/calibration
"$PY" -u source/experiments/neural-operator-poisson/dataset.py generate --output out/train --split train --count 128 --intervals 256 --calibration out/calibration/calibration.json
"$PY" -u source/experiments/neural-operator-poisson/dataset.py generate --output out/validation --split validation --count 32 --intervals 256 --calibration out/calibration/calibration.json --include-reference
"$PY" -u source/experiments/neural-operator-poisson/diagnose.py --index out/validation/index.json --output out/diagnosis
'''
    (dest/'job.sbatch').write_text(script)
    for name in ['out','logs','cache']: (dest/name).mkdir()
    info=dict(source_commit=commit,source_hashes=hashes,remote=remote,
        staged_at=datetime.now(timezone.utc).isoformat(),time_limit_gpu_hours=1,
        first_block_lane_gpu_hour_cap=8,allocation='one A10080GB, 8CPU,96GB,normal gpu')
    (dest/'ORIGIN.json').write_text(json.dumps(info,indent=2)+'\n')
    (dest/'MANIFEST.sha256').write_text(''.join(f'{sha_file(p)}  {p.relative_to(dest)}\n' for p in sorted(dest.rglob('*')) if p.is_file()))
    record=HERE/'runs'/label; record.mkdir(parents=True,exist_ok=False)
    (record/'submission.json').write_text(json.dumps(info,indent=2)+'\n')
    subprocess.run(['ssh','tufts-login',f'mkdir -p {remote}'],check=True)
    subprocess.run(['scp','-r',str(dest)+'/.',f'tufts-login:{remote}/'],check=True)
    subprocess.run(['ssh','tufts-login',f'cd {remote} && sha256sum -c MANIFEST.sha256 --quiet && squeue -u tawal01 && sbatch job.sbatch && squeue -u tawal01'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');a=p.parse_args();stage(a.label)
