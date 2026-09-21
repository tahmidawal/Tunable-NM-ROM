"""Checksum-verified pull of one finished attempt; large files go to the ignored ckpt/ dir.

usage: collect.py ATTEMPT          (does NOT delete the remote dir; do that after the audit)
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
NS = '/cluster/tufts/paralab/tawal01/bankfloor_20260920'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''):
            h.update(c)
    return h.hexdigest()


def main():
    att = sys.argv[1]
    run = LANE / 'runs' / att
    for sub in ('output/', 'logs/', 'OUTPUTS.sha256'):
        subprocess.check_call(['rsync', '-a', f'tufts-login:{NS}/{att}/{sub}', str(run / sub)])
    bad = []
    lines = (run / 'OUTPUTS.sha256').read_text().splitlines()
    for line in lines:
        digest, name = line.split(None, 1)
        if sha(run / name.strip()) != digest:
            bad.append(name)
    assert not bad and lines, f'checksum mismatch: {bad}'
    log = ''.join(p.read_text() for p in (run / 'logs').glob('*.out'))
    assert 'jax_backend=gpu' in log and 'ALL-DONE' in log, 'log markers missing'
    shutil.copy(run / 'output/result.json', run / 'result.json')
    ck = LANE / 'ckpt'
    ck.mkdir(exist_ok=True)
    man_path = LANE / 'CKPT-MANIFEST.json'
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for p in sorted((run / 'output').rglob('*')):
        if p.is_file() and p.parent.name in ('ckpt', 'fields'):
            dest = ck / att / p.parent.name / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            if p.parent.name == 'ckpt':
                man[f'{att}/{p.name}'] = dict(
                    path=str(dest), bytes=dest.stat().st_size, sha256=sha(dest), attempt=att,
                    remote_origin=f'{NS}/{att}/output/ckpt/{p.name}')
    man_path.write_text(json.dumps(man, indent=2) + '\n')
    print(f'collected {att}: {len(lines)} files verified')


if __name__ == '__main__':
    main()
