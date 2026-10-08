"""Checksum-collect one completed attempt into runs/<attempt>/archive, then (with --cleanup) delete the remote job
directory after the local checksums verified (pattern of vendor/quad2d/cluster/collect.py)."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcpwide'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--cleanup', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    assert not (a.partial and a.cleanup), 'never clean up after a partial collection'
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/jcp-wide-bank/runs' / a.attempt / 'archive'
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
    print(out, 'checksums verified')
    if a.cleanup:
        jobs = subprocess.check_output(['ssh', 'tufts-login', f'squeue -u $USER -h -o %j | grep -cx jw_{a.attempt} || true'],
                                       text=True).strip()
        assert jobs == '0', f'jw_{a.attempt} still in the queue: no cleanup'
        assert (out / 'output' / 'COMPLETE').exists() or (out / 'output' / 'training.json').exists(), 'no completion marker'
        subprocess.run(['ssh', 'tufts-login', f'rm -rf {shlex.quote(remote)}'], check=True)
        print('remote removed:', remote)


if __name__ == '__main__':
    main()
