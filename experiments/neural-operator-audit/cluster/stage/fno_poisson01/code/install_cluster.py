"""Add the baseline stack without replacing any existing distribution."""
from pathlib import Path
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

HERE = Path(__file__).resolve().parent
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
assert sys.executable == PY
assert str(HERE).startswith('/cluster/tufts/paralab/tawal01/no_audit_20260914/')
before = {canonicalize_name(d.metadata['Name']): d.version for d in metadata.distributions()}
(HERE / 'packages-before.json').write_text(json.dumps(before, indent=2))
requirements = (HERE / 'requirements-cluster-additions.txt').read_text().splitlines()
assert 'torch' not in before
assert all(canonicalize_name(Requirement(r).name) not in before for r in requirements)
(HERE / 'tmp').mkdir()
env = dict(os.environ, TMPDIR=str(HERE / 'tmp'), PIP_NO_CACHE_DIR='1')
subprocess.run([PY, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                '--report', str(HERE / 'support-install.json'), '-r',
                str(HERE / 'requirements-cluster-additions.txt')], check=True, env=env)
subprocess.run([PY, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                '--report', str(HERE / 'torch-install.json'), 'torch==2.11.0+cu128',
                '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True, env=env)
after = {canonicalize_name(d.metadata['Name']): d.version for d in metadata.distributions()}
changed = {n: (v, after.get(n)) for n, v in before.items() if after.get(n) != v}
(HERE / 'packages-after.json').write_text(json.dumps(after, indent=2))
(HERE / 'preservation.json').write_text(json.dumps({'existing_distributions_unchanged': not changed,
    'changed': changed, 'added': {n:v for n,v in after.items() if n not in before},
    'limitation': 'Torch CUDA dependency pins differ from preserved newer JAX CUDA libraries; '
                  'compatibility must pass allocated-GPU JAX and Torch numerical checks.'}, indent=2))
assert not changed, changed
check = subprocess.run([PY, '-m', 'pip', 'check'], text=True, capture_output=True)
(HERE / 'pip-check.txt').write_text(check.stdout + check.stderr)
print('Existing package versions preserved. GPU compatibility verification remains required.', flush=True)
