"""Checksum-collect one exact completed ns2d attempt; remote cleanup stays explicit.

    python experiments/ns2d/cluster/collect.py <attempt> [--exclude path ...]

Verifies OUTPUTS.sha256 and MANIFEST.sha256 on the cluster, tars the attempt, copies it,
verifies the tarball's own checksum locally, extracts, and re-verifies OUTPUTS.sha256.
Large deterministic artifacts can be excluded from the tarball and copied separately.
"""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/ns_20260917'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--exclude', nargs='*', default=[])
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/ns2d/runs' / a.attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    members = ['COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'run.sbatch', 'logs',
               'OUTPUTS.sha256', 'experiments']
    cmd = (f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && '
           f'sha256sum -c MANIFEST.sha256 --quiet && tar -czf collection.tar.gz '
           + ' '.join(f'--exclude={shlex.quote(x)}' for x in a.exclude) + ' '
           + ' '.join(map(shlex.quote, members))
           + ' && sha256sum collection.tar.gz > collection.tar.gz.sha256')
    subprocess.run(['ssh', 'tufts-login', cmd], check=True)
    for name in ['collection.tar.gz', 'collection.tar.gz.sha256']:
        subprocess.run(['scp', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.gz.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xzf', 'collection.tar.gz'], cwd=out, check=True)
    for x in a.exclude:
        (out / x).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['scp', f'tufts-login:{remote}/{x}', str(out / x)], check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    if a.exclude:
        (out / 'EXCLUDED-FROM-ARCHIVE.txt').write_text('\n'.join(a.exclude) + '\n')
    print(out)
    print('Checksums verified; exact remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
