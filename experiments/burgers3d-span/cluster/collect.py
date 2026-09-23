"""Checksum-collect one completed attempt into runs/<attempt>/archive; remote removal is a separate explicit step
(`--remove`, only after local verification)."""
import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b3dspan_20260923'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--remove', action='store_true', help='after a verified collection: delete the remote dir')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/burgers3d-span/runs' / a.attempt / 'archive'
    if a.remove:
        assert (out / 'OUTPUTS.sha256').exists()
        subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
        subprocess.run(['ssh', 'tufts-login', f'rm -rf {shlex.quote(remote)} && test ! -e {shlex.quote(remote)} && echo removed {remote}'], check=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    pre = (f'cd {shlex.quote(remote)} && ' + ('find code/output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && '
                                              if a.partial else '') +
           'sha256sum -c OUTPUTS.sha256 --quiet && tar -cf collection.tar COMMIT.txt PROVENANCE.json MANIFEST.sha256 '
           'run.sbatch logs OUTPUTS.sha256 code/output && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(['ssh', 'tufts-login', pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['rsync', '-a', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    (out / 'collection.tar').unlink()
    print(out)


if __name__ == '__main__':
    main()
