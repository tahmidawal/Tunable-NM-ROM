"""Prepare an immutable Burgers FNO code bundle for one cluster job.

Data is never staged from this machine. The verified cluster-generated Burgers
cache is copied on the cluster into the job's own directory and re-verified
against its own recorded checksums before training.

`fno_burgers01` is the bounded equal-epoch capacity screen; `fno_burgers02` is
the equal-wall-budget long-training job that also scores the matched ROM/FOM
diagnosis cohort. Each gets its own job directory, as required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
NAMESPACE = '/cluster/tufts/paralab/tawal01/no_audit_20260914'
CACHE = '/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01'

JOBS = {
    'fno_burgers01': dict(
        worker='worker_burgers.py', job_name='ctol_noa_fno_b01', time='04:30:00',
        capacities='equal 200-epoch budget for small, medium and large, then a learning-rate '
                   'refinement of the best capacity',
        extra_data=(), gpu_hour_target=4.25),
    'fno_burgers02': dict(
        worker='worker_burgers_long.py', job_name='ctol_noa_fno_b02', time='04:15:00',
        capacities='equal 3000-second wall budget per capacity with early stopping, then a '
                   'learning-rate refinement of the best capacity',
        extra_data=('refinement',), gpu_hour_target=4.0),
}


def main(name):
    spec = JOBS[name]
    remote = f'{NAMESPACE}/{name}'
    out = HERE / 'cluster/stage' / name
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    (out / 'logs').mkdir()
    for path in HERE.glob('*.py'):
        shutil.copy2(path, out / 'code' / path.name)
    shutil.copytree(HERE / 'configs', out / 'code/configs')
    shutil.copy2(HERE / 'NEURALOPERATOR-LICENSE', out / 'code/NEURALOPERATOR-LICENSE')
    commit = subprocess.check_output(['git', '-C', str(WT), 'rev-parse', 'HEAD'], text=True).strip()
    (out / 'source-provenance.json').write_text(json.dumps(dict(
        source_commit=commit, source_worktree=str(WT), job_directory=remote, worker=spec['worker'],
        pde='burgers', mesh_intervals=256, output_times=[0., .05, .1, .15, .2, .25],
        data_origin=f'Burgers lane cluster-generated cache {CACHE}; copied on the cluster and re-verified',
        extra_data=list(spec['extra_data']),
        model_inputs='sampled initial nodal field, viscosity and coordinates only; no generation '
                     'descriptors, case ids or truth sidecars',
        time_dependence='direct multi-time output: five evolved fields as output channels; the supplied '
                        'initial state is returned exactly and no autoregressive rollout is used',
        capacities=spec['capacities'],
        resumability='none; each training run is bounded inside this one allocation',
        gpu_hour_target=spec['gpu_hour_target'], precision='float64/complex128, highest matmul precision',
        comparisons='same-job FNO timing and validation accuracy; matched-cohort accuracy may be compared '
                    'with the Burgers lane diagnosis, timing may not'), indent=2) + '\n')
    (out / 'run.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name={spec['job_name']}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time={spec['time']}
#SBATCH --signal=B:USR1@180
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
cd {remote}
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TMPDIR={remote}/tmp
export XDG_CACHE_HOME={remote}/cache
export TORCH_HOME={remote}/cache/torch
export MPLCONFIGDIR={remote}/cache/matplotlib
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
sha256sum -c MANIFEST.sha256
( cd data && sha256sum -c DATA.sha256 )
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}'); sys.exit(0 if b=='gpu' else 42)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
exec "$PY" code/{spec['worker']}
''')
    hashes = []
    for path in sorted(out.rglob('*')):
        if path.is_file():
            hashes.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}')
    (out / 'MANIFEST.sha256').write_text('\n'.join(hashes) + '\n')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='fno_burgers01', choices=sorted(JOBS))
    main(parser.parse_args().name)
