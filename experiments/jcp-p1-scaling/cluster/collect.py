"""Checksum-collect one completed attempt into runs/<attempt>/archive; then (separately) clean the remote dir.

    python cluster/collect.py <attempt> [--partial]
    python cluster/collect.py <attempt> --cleanup     (after the archive is verified and committed/manifested)
"""
import argparse
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-p1-scaling'
SSH = ['timeout', '600', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--cleanup', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = HERE / 'runs' / a.attempt / 'archive'
    if a.cleanup:
        assert (out / 'OUTPUTS.sha256').exists(), 'collect first'
        subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
        subprocess.run(SSH + [f'rm -rf {shlex.quote(remote)}'], check=True)
        print('removed', remote)
        return
    out.mkdir(parents=True, exist_ok=False)
    pre = (f'cd {shlex.quote(remote)} && ' + ('find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && '
                                              if a.partial else '') +
           'sha256sum -c OUTPUTS.sha256 --quiet && tar -cf collection.tar COMMIT.txt PROVENANCE.json MANIFEST.sha256 '
           'run.sbatch logs OUTPUTS.sha256 output && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(SSH + [pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['timeout', '3600', 'scp', '-q', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    (out / 'collection.tar').unlink()
    print(out)


if __name__ == '__main__':
    main()
