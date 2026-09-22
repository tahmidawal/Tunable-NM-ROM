"""Stage one ops-tune-deeponet attempt into an isolated paralab-bound directory.

Copied from `experiments/ops-deeponet-b2d/cluster/stage.py` with four changes, all forced by
DESIGN.md:

  * the lane path, the cluster namespace and the staged-file list;
  * **several data mounts instead of one** -- this lane trains against the pinned 128-case
    cache and against its own extended cache in the same job (DESIGN section 2.3);
  * mounts are **read-only symlinks, verified through the link**, not copies: the extended
    cache is ~34 GB and copying it into three job directories would spend 100 GB of a share
    that is 91 % full for no isolation gain (nothing in this lane writes into a mount, and
    every mount is `chmod -w` after its generating job);
  * `timing.py` is not staged. No speed number is admissible from this lane (DESIGN section 7).

    python cluster/stage.py <attempt> <spec-file-name> [--gpu a100|h100|h200|l40s]

Each attempt gets its OWN submit directory and its OWN remote directory: one job per
directory, never two. Every staged file is checked byte-for-byte against the committed blob at
HEAD before it is copied, so what runs on the cluster is exactly what is in Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/ops-tune-deeponet'
NAMESPACE = '/cluster/tufts/paralab/tawal01/opstune_don_20260922'
PINNED = '/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01'
CODE = ['families.py', 'model.py', 'train.py', 'dataset.py', 'evaluate_cohort.py',
        'prepare_diagnosis_cohort.py', 'spectral_conv_f64.py', 'NEURALOPERATOR-LICENSE',
        'smoke_second.py', 'training_smoke_second.py', 'worker_tune.py', 'worker_gen.py',
        'gen_more.py', 'build_pool.py', 'compose.py',
        # the pinned generator, byte-identical to what produced the 128-case cache
        'data.py', 'refine.py', 'protocol.json', 'protocol-refined.json',
        'engines.py', 'sep_common.py']
CODE += sorted(str(p.relative_to(ROOT / LANE))
               for p in (ROOT / LANE / 'configs').rglob('*.json'))
EXCLUDE = 'pax007'

REFERENCE = re.compile(r"""['"]code/([^'"\s{}]+)['"]""")


def check_references(out):
    """Every `code/<path>` literal in any staged Python or JSON file must resolve inside the
    staged tree, and every staged JSON must parse."""
    missing = []
    for path in sorted((out / 'code').rglob('*')):
        if path.suffix not in ('.py', '.json'):
            continue
        if path.suffix == '.json':
            json.loads(path.read_text())
        for match in REFERENCE.finditer(path.read_text()):
            if not (out / 'code' / match.group(1)).is_file():
                missing.append(f"{path.relative_to(out)} -> code/{match.group(1)}")
    assert not missing, f'staged tree does not contain every referenced file: {missing}'
    return len(list((out / 'code').rglob('*.json')))


def mount_script(mounts):
    """Read-only symlinks into `data/<name>/`, each verified through the link."""
    lines = []
    for name, mount in mounts.items():
        source, dirs = mount['source'], mount['dirs']
        manifest = mount.get('manifest', 'DATA.sha256')
        lines.append(f'mkdir -p data/{name}')
        for entry in dirs:
            lines.append(f'ln -s {source}/{entry} data/{name}/{entry}')
        lines.append(f'ln -s {source}/{manifest} data/{name}/{manifest}')
        if mount.get('verify', True):
            lines.append(f'( cd data/{name} && sha256sum -c {manifest} --quiet && '
                         f'echo "data_verified_{name}=$(wc -l < {manifest})" )')
        else:
            lines.append(f'echo "data_unverified_{name}=intentional; see spec"')
    return '\n'.join(lines)


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
        output_times=[0., .05, .1, .15, .2, .25],
        data_origin=spec.get('data_origin', 'see spec mounts; the pinned Burgers cache is '
                             f'{PINNED} and is never modified'),
        model_inputs='sampled initial nodal field, viscosity and coordinates only; no generation '
                     'descriptors, case ids or truth sidecars',
        time_dependence='direct multi-time output: five evolved fields as output channels; the supplied '
                        'initial state is returned exactly and no autoregressive rollout is used',
        resumability='none; each training run is bounded inside this one allocation',
        precision='float64 outside the network; network dtype as declared in each config; TF32 disabled',
        comparisons='validation and matched-cohort accuracy only; NO timing block runs in this lane '
                    'and no speed statement from it is admissible'), indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = '''#!/bin/bash
#SBATCH --job-name=__JOBNAME__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
__CONSTRAINT__#SBATCH --exclude=__EXCLUDE__
#SBATCH --cpus-per-task=8
#SBATCH --mem=__MEM__
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
# Data: read-only symlinks to verified caches (never modified), verified through the link.
test ! -e data
mkdir data
__MOUNTS__
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# Forward Slurm's USR1 (sent to this batch shell 180 s before the limit) to the worker,
# which asks train.py for a stop at the next evaluation; then wait for it.
"$PY" code/__WORKER__ code/specs/__SPEC__ &
WORKER=$!
trap 'kill -USR1 "$WORKER" 2>/dev/null || true' USR1
wait "$WORKER"
find out $(test -d data/diagnosis-cohort && echo data/diagnosis-cohort) -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
du -sh out | tail -1
echo ALL-DONE
'''
    tokens = (('__JOBNAME__', spec['job_name']), ('__REMOTE__', remote),
              ('__MOUNTS__', mount_script(spec['mounts'])),
              ('__WORKER__', spec['worker']), ('__SPEC__', spec_name), ('__HOURS__', spec['time']),
              ('__MEM__', spec.get('mem', '180G')),
              ('__EXCLUDE__', EXCLUDE), ('__GPU__', args.gpu),
              ('__CONSTRAINT__', '#SBATCH --constraint=a100-80G\n' if args.gpu == 'a100' else ''))
    for token, value in tokens:
        assert script.count(token) >= 1, token
        script = script.replace(token, value)
    assert not any(token in script for token, _ in tokens), script
    (out / 'run.sbatch').write_text(script)
    print(f'referenced-file check passed; {check_references(out)} staged JSON files parsed')
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
