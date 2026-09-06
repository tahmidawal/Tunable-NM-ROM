"""Stage one immutable Burgers-3D repair job; submission is a separate action.

Invoke with the absolute project Python. Stages no solution data. Optional
archived parameter metadata is only a provenance reference; actual parameters
and truth are regenerated. Sources must equal committed content; each job has
a unique output and log path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('label')
    parser.add_argument('--script', choices=['b3d_repair.py', 'sep_b3d_tensor.py'], default='b3d_repair.py')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-table', type=Path)
    parser.add_argument('--env', action='append', default=[])
    parser.add_argument('--hours', type=int, default=2)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z][a-z0-9_]{0,30}', args.label), 'unsafe job label'
    assert 1 <= args.hours <= 20
    src = Path(__file__).resolve().parent.parent
    repo = src.parents[1]
    stage = src / 'cluster/stage' / args.label
    record = src / 'runs/b3d_repair' / args.label
    assert not stage.exists() and not record.exists(), 'use a new label for every attempt'
    assert args.checkpoint.is_file()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    names = ['b3d_common.py', 'b3d_tensor_common.py', 'sep_b3d_tensor.py',
             'b3d_repair.py', 'sep_hfit.py']
    source_content = {}
    for name in names:
        content = (src / name).read_bytes()
        tracked = subprocess.check_output(['git', '-C', str(repo), 'show',
                                          f'{commit}:experiments/separable-decoder/{name}'])
        assert content == tracked, f'commit {name} before staging'
        source_content[name] = content
    env = {}
    for item in args.env:
        key, value = item.split('=', 1)
        assert re.fullmatch(r'[A-Z][A-Z0-9_]*', key), key
        assert key not in {'HOME', 'CODEX_HOME', 'CKPT', 'OUT', 'TABLE_DIR', 'COMMIT',
                           'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_ENABLE_X64'}
        env[key] = value
    remote = f'/cluster/tufts/paralab/tawal01/b3d_repair_20260906/{args.label}'
    for path in ['code', 'in', 'out', 'logs']:
        (stage / path).mkdir(parents=True)
    record.mkdir(parents=True)
    for name, content in source_content.items():
        (stage / 'code' / name).write_bytes(content)
    shutil.copy2(args.checkpoint, stage / 'in/checkpoint.pkl')
    reference_assignment = ''
    if args.reference_table is not None:
        assert args.reference_table.is_file()
        shutil.copy2(args.reference_table, stage / 'in/reference_table.npz')
        reference_assignment = 'REFERENCE_TABLE="$TASK_ROOT/in/reference_table.npz"'
    elif args.script == 'b3d_repair.py':
        raise ValueError('repair diagnostics require --reference-table for provenance')
    (stage / 'COMMIT.txt').write_text(commit + '\n')
    assignments = ' '.join(shlex.quote(f'{key}={value}') for key, value in env.items())
    batch = f'''#!/bin/bash
#SBATCH --job-name=ctol_b3dr_{args.label}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time={args.hours:02}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
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
export COMMIT={commit}
cd "$TASK_ROOT/code"
env CKPT="$TASK_ROOT/in/checkpoint.pkl" OUT="$TASK_ROOT/out/result.json" TABLE_DIR="$TASK_ROOT/code/runs/b3dtensor/tables" {reference_assignment} {assignments} "$PY" -u {args.script}
'''
    (stage / 'job.sbatch').write_text(batch)
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    manifest = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(stage)}\n' for p in files)
    (stage / 'MANIFEST.sha256').write_text(manifest)
    (record / 'INPUTS.sha256').write_text(manifest)
    (record / 'job.sbatch').write_text(batch)
    (record / 'submission.json').write_text(json.dumps(dict(
        label=args.label, script=args.script, source_commit=commit, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        remote=remote, env=env), indent=2) + '\n')
    print(stage)
    print(remote)


if __name__ == '__main__':
    main()
