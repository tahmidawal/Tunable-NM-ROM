"""Private, checksummed wave attempt transport. No account-wide cancellation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

CELL = Path(__file__).resolve().parent
TREE = CELL.parents[1]
NAMESPACE = '/cluster/tufts/paralab/tawal01/modcp_wave2d_20260910'
CONNECTION = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
              '-o', 'Hostname=login-p02.pax.tufts.edu', '-o', 'HostKeyAlias=login-prod.pax.tufts.edu']
SSH = ['ssh', *CONNECTION, 'tufts-login']


def run(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def stage(label, boundary, kind, inputs=None, reuse_eq=None, phase='validation', validation=None, seal=None):
    commit = run(['git', 'rev-parse', 'HEAD'], cwd=TREE).strip()
    dest = CELL/'stages'/label
    dest.mkdir(parents=True, exist_ok=False)
    for folder in ('code/modcp-eq/common', 'code/modcp-eq/wave', 'code/fresh-wave-head', 'logs', 'out', 'inputs'):
        (dest/folder).mkdir(parents=True)
    files = [*CELL.glob('*.py'), *(CELL/'common').glob('*.py'), *(CELL/'wave').glob('*.py'),
             CELL/'wave/config.json', TREE/'experiments/fresh-wave-head/fresh_fom.py',
             TREE/'experiments/fresh-wave-head/test_fresh_fom.py']
    sources = {}
    for path in files:
        rel = path.relative_to(TREE)
        payload = subprocess.check_output(['git', 'show', f'{commit}:{rel}'], cwd=TREE)
        if payload != path.read_bytes():
            raise RuntimeError('Uncommitted source: '+str(path))
        target = dest/'code'/path.relative_to(TREE/'experiments')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        sources[str(target.relative_to(dest))] = sha(target)
    if kind in ('compare', 'smoke'):
        if inputs is None:
            raise ValueError('A checksum-collected trained checkpoint directory is required')
        inputs = Path(inputs).resolve()
        for name in ('cp.pkl', 'modcp.pkl', 'film.pkl'):
            shutil.copy2(inputs/name, dest/'inputs'/name)
        training_result = inputs.parent/'training_result.json'
        shutil.copy2(training_result, dest/'inputs'/'training_result.json')
        write(dest/'inputs'/'ORIGIN.json', {'training_result_sha256': sha(training_result),
                                          'checkpoint_sha256': {name: sha(inputs/name) for name in ('cp.pkl', 'modcp.pkl', 'film.pkl')}})
        if reuse_eq is not None:
            reuse_eq = Path(reuse_eq).resolve()
            target = dest/'inputs'/'eq_reuse'
            target.mkdir()
            shutil.copy2(reuse_eq/'handoff.json', target/'handoff.json')
            for rule in (reuse_eq/'quadrature').glob('*'):
                if (rule/'rule.npz').exists() and (rule/'audit.json').exists():
                    (target/'quadrature'/rule.name).mkdir(parents=True)
                    for name in ('rule.npz', 'audit.json'):
                        shutil.copy2(rule/name, target/'quadrature'/rule.name/name)
        if kind == 'compare' and phase == 'evaluation':
            if validation is None or seal is None:
                raise ValueError('Evaluation requires immutable validation directory and global cohort seal')
            validation, seal = Path(validation).resolve(), Path(seal).resolve()
            target = dest/'inputs'/'validation_bundle'
            target.mkdir()
            for name in ('handoff.json', 'frozen_selection_256.json', 'frozen_selection_512.json'):
                shutil.copy2(validation/name, target/name)
            shutil.copy2(seal, target/'global_seal.json')
            shutil.copytree(validation/'quadrature', target/'quadrature')
            # Host-only integrity guard: refuse to upload/open an unsealed panel.
            import sys
            sys.path.insert(0, str(CELL/'wave'))
            from seal import verify_validation_bundle
            cfg = json.loads((CELL/'wave/config.json').read_text())
            expected = {name: sha(inputs/f'{name}.pkl') for name in ('cp', 'modcp', 'film')}
            verify_validation_bundle(target, cfg, 'wave_reflective' if boundary == 'dirichlet' else 'wave_absorbing', expected)
    remote = NAMESPACE+'/'+label
    commands = '"$PY" -u code/modcp-eq/wave/test_physics.py\n'
    if (CELL/'wave/test_weak.py').exists():
        commands += '"$PY" -u code/modcp-eq/wave/test_weak.py\n'
    if kind == 'train':
        commands += f'"$PY" -u code/modcp-eq/wave/train_wave.py --config code/modcp-eq/wave/config.json --boundary {boundary} --out out/training\n'
    elif kind == 'compare':
        commands += f'"$PY" -u code/modcp-eq/wave/compare.py --config code/modcp-eq/wave/config.json --boundary {boundary} --inputs inputs --out out/comparison --phase {phase}\n'
    else:
        commands += f'"$PY" -u code/modcp-eq/wave/smoke.py --inputs inputs --boundary {boundary} --out out/smoke\n'
    batch = f'''#!/bin/bash
#SBATCH --job-name=ctol_modcp_wave_{label}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT={commit}
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
finish() {{
  task_exit=$?
  cd "$TASK_ROOT"
  printf '%s\\n' "$task_exit" > EXIT_CODE
  find out -type f ! -path '*/data/training_fields.npy' -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
  exit "$task_exit"
}}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
{commands}
'''
    (dest/'job.sbatch').write_text(batch)
    metadata = {'label': label, 'boundary': boundary, 'kind': kind, 'phase': phase if kind == 'compare' else None, 'source_commit': commit,
                'source_sha256': sources, 'remote': remote, 'staged_at': datetime.now(timezone.utc).isoformat()}
    write(dest/'CONFIG.json', metadata)
    (dest/'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(dest)}\n'
        for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'MANIFEST.sha256'))
    print(json.dumps(metadata, indent=2))


def submit(label):
    dest = CELL/'stages'/label
    metadata = json.loads((dest/'CONFIG.json').read_text())
    remote = metadata['remote']
    q = shlex.quote(remote)
    print(run(SSH+['squeue -u tawal01 -o "%.18i %.40j %.8T %.10M %.20R"'], timeout=30))
    print(run(SSH+['df -h /cluster/tufts/paralab/tawal01'], timeout=30))
    exists = run(SSH+[f'if test -d {q}; then echo exists; fi'], timeout=30).strip()
    if exists:
        # Recovery from an SSH disconnect must not overwrite staged/running code.
        run(SSH+[f'cd {q} && sha256sum -c MANIFEST.sha256 --quiet && test ! -e EXIT_CODE'], timeout=30)
    else:
        run(SSH+[f'mkdir -p {q}'], timeout=30)
        subprocess.run(['scp', *CONNECTION, '-q', '-r', str(dest)+'/.', 'tufts-login:'+remote+'/'], check=True, timeout=120)
    run(SSH+[f'cd {q} && sha256sum -c MANIFEST.sha256 --quiet'], timeout=30)
    jobname = shlex.quote('ctol_modcp_wave_'+label)
    queued = run(SSH+[f'squeue -h -n {jobname} -o %i'], timeout=30).split()
    if len(queued) > 1:
        raise RuntimeError('Unexpected duplicate attempt jobs; refuse to submit')
    job = queued[0] if queued else run(SSH+[f'cd {q} && sbatch --parsable job.sbatch'], timeout=30).strip()
    metadata.update(job_id=job, submitted_at=datetime.now(timezone.utc).isoformat())
    record = CELL/'runs'/label
    record.mkdir(parents=True, exist_ok=True)
    write(record/'submission.json', metadata)
    print(run(SSH+['squeue -u tawal01 -o "%.18i %.40j %.8T %.10M %.20R"'], timeout=30))
    print(json.dumps(metadata, indent=2))


def collect(label):
    record = CELL/'runs'/label
    metadata = json.loads((record/'submission.json').read_text())
    remote = metadata['remote']
    if remote != NAMESPACE+'/'+label:
        raise RuntimeError('Attempt namespace mismatch')
    q = shlex.quote(remote)
    queued = run(SSH+['squeue -h -u tawal01 -o %i'], timeout=30).split()
    if str(metadata['job_id']) in queued:
        raise RuntimeError('Job is still queued/running')
    run(SSH+[f'cd {q} && test -f EXIT_CODE && sha256sum -c RESULTS.sha256 --quiet'], timeout=120)
    # Reproducible data is not transferred. Checkpoints, optimizer states,
    # provenance, source and actual requested result fields all remain durable.
    # Numerical field NPZs are already compressed. Recompressing a large panel
    # wastes host time and used to exceed the collection timeout.
    run(SSH+[f'cd {q} && tar --exclude=./out/training/data/training_fields.npy -cf ../{label}.tar . && sha256sum ../{label}.tar'], timeout=3600)
    checksum = run(SSH+[f'sha256sum {shlex.quote(NAMESPACE+"/"+label+".tar")}'], timeout=1800).split()[0]
    archive = record/'verified-cluster.tar'
    subprocess.run(['scp', *CONNECTION, '-q', 'tufts-login:'+NAMESPACE+'/'+label+'.tar', str(archive)], check=True, timeout=7200)
    if sha(archive) != checksum:
        raise RuntimeError('Archive checksum mismatch')
    extracted = record/'cluster'
    extracted.mkdir(exist_ok=False)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(extracted)], check=True)
    subprocess.run(['sha256sum', '-c', 'RESULTS.sha256', '--quiet'], cwd=extracted, check=True)
    size = archive.stat().st_size
    parts = []
    if size > 90*1024**2:
        joined = hashlib.sha256()
        with archive.open('rb') as stream:
            while payload := stream.read(90*1024**2):
                part = record/f'{archive.name}.part-{len(parts):04d}'
                part.write_bytes(payload)
                joined.update(payload)
                parts.append({'path': part.name, 'sha256': sha(part)})
        if joined.hexdigest() != checksum:
            raise RuntimeError('Split archive hash mismatch')
        archive.unlink()
    else:
        parts.append({'path': archive.name, 'sha256': checksum})
    write(record/'ARCHIVE.json', {'sha256': checksum, 'bytes': size, 'ordered_parts': parts,
                                'format': 'tar', 'compression': 'none; contained NPZ field artifacts are compressed',
                                'source_commit': metadata['source_commit'], 'job_id': metadata['job_id'],
                                'excluded_regenerated_data': 'out/training/data/training_fields.npy',
                                'restore': 'Concatenate ordered_parts in their recorded order; verify sha256 of resulting uncompressed tar; extract. RESULTS.sha256 authenticates each result.'})
    # Exact directory and exact archive, only after BOTH independent checks pass.
    run(SSH+[f'rm -rf -- {q} {shlex.quote(NAMESPACE+"/"+label+".tar")} && test ! -e {q}'], timeout=60)
    metadata['remote_cleanup_complete'] = True
    write(record/'submission.json', metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('stage', 'submit', 'collect'))
    ap.add_argument('label')
    ap.add_argument('--boundary', choices=('dirichlet', 'absorbing'), default='dirichlet')
    ap.add_argument('--kind', choices=('train', 'compare', 'smoke'), default='train')
    ap.add_argument('--inputs', type=Path)
    ap.add_argument('--reuse-eq', type=Path)
    ap.add_argument('--phase', choices=('validation', 'evaluation'), default='validation')
    ap.add_argument('--validation', type=Path)
    ap.add_argument('--seal', type=Path)
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{1,30}', args.label):
        ap.error('Use a private simple attempt label')
    if args.action == 'stage':
        stage(args.label, args.boundary, args.kind, args.inputs, args.reuse_eq, args.phase, args.validation, args.seal)
    else:
        globals()[args.action](args.label)
