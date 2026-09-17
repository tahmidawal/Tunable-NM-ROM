"""Checksum-collect one exact completed attempt; remote cleanup stays explicit.

Copied from `experiments/q-ridge/cluster/collect.py`. This lane writes no `bank_G.npz`,
so nothing is excluded: everything under `output/` is archived byte for byte.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_qxm_20260917'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/b-qxm/runs' / a.attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    members = ['COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'run.sbatch', 'logs',
               'OUTPUTS.sha256', 'output', 'experiments']
    cmd = (f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && '
           f'sha256sum -c MANIFEST.sha256 --quiet && tar -czf collection.tar.gz '
           + ' '.join(map(shlex.quote, members))
           + ' && sha256sum collection.tar.gz > collection.tar.gz.sha256')
    subprocess.run(['ssh', 'tufts-login', cmd], check=True)
    for name in ['collection.tar.gz', 'collection.tar.gz.sha256']:
        subprocess.run(['scp', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.gz.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xzf', 'collection.tar.gz'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'MANIFEST.sha256', '--quiet'], cwd=out, check=True)
    print(out)
    print('Checksums verified; exact remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
