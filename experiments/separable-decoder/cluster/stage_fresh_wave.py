"""Stage committed fresh-wave sources into one immutable cluster attempt.

Coordinator owns local stage/records in the repair worktree. The approved wave
worktree is read-only here. No old wave source or solution data is staged.
Submission, checked result pulls and cleanup remain explicit separate actions.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude')
APPROVED = ROOT/'worktrees/2026-09-06-wave-head-transfer'
NAMESPACE = '/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('label')
    p.add_argument('--entry', required=True)
    p.add_argument('--arg', action='append', default=[])
    p.add_argument('--gpu', choices=['a100', 'h100', 'h200', 'l40s'], default='a100')
    p.add_argument('--constraint', default='a100-40G')
    p.add_argument('--hours', type=int, default=2)
    p.add_argument('--input', action='append', default=[], help='name=/absolute/checkpoint.npz; no solution data')
    a = p.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,30}', a.label):
        p.error('invalid label')
    if not 1 <= a.hours <= 20:
        p.error('hours outside [1,20]')
    if a.constraint and not re.fullmatch(r'[A-Za-z0-9_-]+', a.constraint):
        p.error('invalid feature constraint')
    coordinator = Path(__file__).resolve().parents[1]
    cell = APPROVED/'experiments/fresh-wave-head'
    entry = Path(a.entry)
    if entry.is_absolute() or '..' in entry.parts or entry.suffix != '.py':
        p.error('entry must be a relative Python source inside fresh cell')
    commit = subprocess.check_output(['git', '-C', str(APPROVED), 'rev-parse', 'HEAD'], text=True).strip()
    source = {}
    tracked_names = subprocess.check_output([
        'git', '-C', str(APPROVED), 'ls-tree', '-r', '--name-only', commit,
        '--', 'experiments/fresh-wave-head'], text=True).splitlines()
    for f in [APPROVED/name for name in tracked_names]:
        if not f.is_file() or any(part in {'runs', 'cache', 'data', '__pycache__', '.pytest_cache'} for part in f.relative_to(cell).parts):
            continue
        if f.suffix not in {'.py', '.json', '.md'}:
            continue
        rel = f.relative_to(APPROVED).as_posix()
        payload = f.read_bytes()
        tracked = subprocess.check_output(['git', '-C', str(APPROVED), 'show', f'{commit}:{rel}'])
        if payload != tracked:
            raise RuntimeError(f'commit source before staging: {rel}')
        if f.suffix == '.py':
            for node in ast.walk(ast.parse(payload)):
                imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                    [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                if any(name.startswith(('wav2d', 'wave2d', 'stk2d')) for name in imports):
                    raise RuntimeError(f'legacy dependency in {rel}: {imports}')
        source[f.relative_to(cell).as_posix()] = payload
    if entry.as_posix() not in source:
        raise RuntimeError('entry not among committed fresh sources')
    stage = coordinator/'cluster/stage'/f'fresh_{a.label}'
    record = coordinator/'runs/fresh_wave_campaign'/a.label
    if stage.exists() or record.exists():
        raise RuntimeError('immutable attempt exists; choose another label')
    inputs = {}
    for item in a.input:
        name, rawpath = item.split('=', 1)
        path = Path(rawpath)
        if not re.fullmatch(r'[a-z][a-z0-9_]*\.(npz|pkl|json)', name) or not path.is_absolute() or not path.is_file():
            raise RuntimeError('invalid named checkpoint input')
        if name in inputs:
            raise RuntimeError('duplicate input')
        inputs[name] = path
    for name in ('code', 'in', 'out', 'logs'):
        (stage/name).mkdir(parents=True)
    record.mkdir(parents=True)
    for rel, payload in source.items():
        dest = stage/'code'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    for name, path in inputs.items():
        (stage/'in'/name).write_bytes(path.read_bytes())
    remote = f'{NAMESPACE}/{a.label}'
    cfg = dict(label=a.label, source_commit=commit, source_worktree=str(APPROVED),
               source_hashes={name: hashlib.sha256(payload).hexdigest() for name,payload in source.items()},
               remote=remote, entry=a.entry, arguments=a.arg, gpu=a.gpu, constraint=a.constraint,
               inputs={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in inputs.items()})
    (stage/'COMMIT.txt').write_text(commit+'\n')
    (stage/'CONFIG.json').write_text(json.dumps(cfg, indent=2)+'\n')
    feature = f'#SBATCH --constraint={a.constraint}' if a.constraint else ''
    command = shlex.join([a.entry, *a.arg])
    batch = f'''#!/bin/bash
#SBATCH --job-name=ctol_fw_{a.label}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{a.gpu}:1
{feature}
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time={a.hours:02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={shlex.quote(remote)}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT={commit}
export OUT="$TASK_ROOT/out"
export INPUT="$TASK_ROOT/in"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
finish() {{
    task_exit=$?
    cd "$TASK_ROOT"
    printf '%s\\n' "$task_exit" > EXIT_CODE
    find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
    exit "$task_exit"
}}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
"$PY" -u {command}
'''
    (stage/'job.sbatch').write_text(batch)
    manifest = ''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(stage).as_posix()}\n' for f in sorted(stage.rglob('*')) if f.is_file())
    (stage/'MANIFEST.sha256').write_text(manifest)
    (record/'INPUTS.sha256').write_text(manifest)
    (record/'job.sbatch').write_text(batch)
    (record/'submission.json').write_text(json.dumps(cfg, indent=2)+'\n')
    print(json.dumps(dict(stage=str(stage), record=str(record), remote=remote), indent=2))


if __name__ == '__main__':
    main()
