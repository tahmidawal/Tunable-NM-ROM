"""Checksum-collect one completed attempt into runs/<attempt>/archive (git-ignored except what is committed
explicitly). Remote cleanup is a separate explicit step (cluster/cleanup.sh)."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/quad2d_20261001'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/quadrature-study/runs' / a.attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    pre = (f'cd {shlex.quote(remote)} && ' + ('find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && '
                                              if a.partial else '') +
           'sha256sum -c OUTPUTS.sha256 --quiet && tar -cf collection.tar COMMIT.txt PROVENANCE.json MANIFEST.sha256 '
           'run.sbatch logs OUTPUTS.sha256 output && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(['ssh', 'tufts-login', pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['scp', '-q', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    (out / 'collection.tar').unlink()
    print(out)
    print('Checksums verified; remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
