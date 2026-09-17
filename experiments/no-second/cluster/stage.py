"""Stage one no-second attempt into an isolated paralab-bound directory.

    python cluster/stage.py <attempt> <spec-file-name> [--gpu a100|h100|h200|l40s]

Each attempt gets its OWN submit directory and its OWN remote directory: one job per
directory, never two. Every staged file is checked byte-for-byte against the
committed blob at HEAD before it is copied, so what runs on the cluster is exactly
what is in Git. Data is never staged from this machine: the sbatch preamble copies
the Burgers lane's verified cluster cache into the attempt's own `data/` and
re-verifies it against the cache's `DATA.sha256` before any training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/no-second'
NAMESPACE = '/cluster/tufts/paralab/tawal01/no_second_20260917'
CACHE = '/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01'
CODE = ['families.py', 'model.py', 'train.py', 'dataset.py', 'evaluate_cohort.py', 'timing.py',
        'prepare_diagnosis_cohort.py', 'spectral_conv_f64.py', 'NEURALOPERATOR-LICENSE',
        'smoke_second.py', 'training_smoke_second.py', 'worker_second.py']
# Every config file is staged (pois01 died in its preamble because an explicit list omitted
# the Poisson configs).
CODE += sorted(str(p.relative_to(ROOT / 'experiments/no-second'))
               for p in (ROOT / 'experiments/no-second/configs').rglob('*.json'))
EXCLUDE = 'pax007'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    parser.add_argument('spec')
    parser.add_argument('--gpu', default='a100', choices=('a100', 'h100', 'h200', 'l40s'))
    args = parser.parse_args()
    attempt, spec_name = args.attempt, args.spec
    assert attempt.isalnum(), attempt
    spec = json.loads((ROOT / LANE / 'specs' / spec_name).read_text())
    out = ROOT / LANE / 'runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in CODE + [f'specs/{spec_name}', 'cluster/stage.py']:
        source = f'{LANE}/{name}'
        content = (ROOT / source).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{source}']), source
        dest = out / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=source, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(dict(
        files=proof, source_commit=commit, source_worktree=str(ROOT), job_directory=remote,
        spec=spec, gpu=args.gpu, pde=spec.get('pde', 'burgers'), mesh_intervals=256,
        output_times=[0., .05, .1, .15, .2, .25] if spec.get('pde', 'burgers') == 'burgers' else [0.],
        data_origin=spec.get('data_origin', f'Burgers lane cluster-generated cache {CACHE}; copied on the cluster '
                             'in the sbatch preamble and re-verified against its DATA.sha256; the cache is never modified'),
        model_inputs='sampled initial nodal field, viscosity and coordinates only; no generation '
                     'descriptors, case ids or truth sidecars',
        time_dependence='direct multi-time output: five evolved fields as output channels; the supplied '
                        'initial state is returned exactly and no autoregressive rollout is used',
        resumability='none; each training run is bounded inside this one allocation',
        precision='float64 outside the network; network dtype as declared in each config; TF32 disabled',
        comparisons='same-job timing and validation accuracy; matched-cohort accuracy may be compared with '
                    'the Burgers lane diagnosis and the FNO lane, timing may not'), indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = '''#!/bin/bash
#SBATCH --job-name=__JOBNAME__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
__CONSTRAINT__#SBATCH --exclude=__EXCLUDE__
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=__HOURS__
#SBATCH --signal=B:USR1@180
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export WANDB_MODE=offline WANDB_DISABLED=true
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TORCH_HOME="$TASK_ROOT/cache/torch" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
# Data: copy the verified shared cache into this attempt (never modify the cache), then re-verify.
test ! -e data
mkdir data
cp -r __DATADIRS__ __CACHE__/__MANIFEST__ data/
( cd data && sha256sum -c __MANIFEST__ --quiet && echo "data_verified=$(wc -l < __MANIFEST__)" )
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# Forward Slurm's USR1 (sent to this batch shell 180 s before the limit) to the worker,
# which asks train.py for an epoch-boundary stop; then wait for it.
"$PY" code/worker_second.py code/specs/__SPEC__ &
WORKER=$!
trap 'kill -USR1 "$WORKER" 2>/dev/null || true' USR1
wait "$WORKER"
find out $(test -d data/diagnosis-cohort && echo data/diagnosis-cohort) -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    cache = spec.get('data_source', CACHE)
    data_dirs = ' '.join(f'{cache}/{d}' for d in spec.get('data_dirs', ['train', 'validation', 'refinement']))
    tokens = (('__JOBNAME__', spec['job_name']), ('__REMOTE__', remote), ('__DATADIRS__', data_dirs),
              ('__MANIFEST__', spec.get('data_manifest', 'DATA.sha256')), ('__CACHE__', cache),
              ('__SPEC__', spec_name), ('__HOURS__', spec['time']), ('__EXCLUDE__', EXCLUDE), ('__GPU__', args.gpu),
              ('__CONSTRAINT__', '#SBATCH --constraint=a100-80G\n' if args.gpu == 'a100' else ''))
    for token, value in tokens:
        assert script.count(token) >= 1, token
        script = script.replace(token, value)
    assert not any(token in script for token, _ in tokens), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
