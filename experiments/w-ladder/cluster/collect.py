"""Checksum-collect one exact completed attempt; remote cleanup is an explicit separate step.

    python experiments/w-ladder/cluster/collect.py <attempt>          # collect + verify
    python experiments/w-ladder/cluster/collect.py <attempt> --delete # after local verification only

Everything under the remote attempt is archived byte for byte (OUTPUTS.sha256 and
MANIFEST.sha256 are checked on the cluster before the tar and again locally after extraction).
Pattern copied from experiments/q-ridge/cluster/collect.py.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/w_ladder_20260917'


def ssh(command):
    return subprocess.check_output(['ssh', 'tufts-login', command], text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt'); p.add_argument('--delete', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    run = ROOT / 'experiments/w-ladder/runs' / a.attempt
    out = run / 'archive'
    if a.delete:
        assert (out / 'VERIFIED.txt').exists(), 'verify before deleting'
        print(ssh(f'test -f {shlex.quote(remote)}/collection.tar.gz.sha256 && rm -rf -- {shlex.quote(remote)} && test ! -e {shlex.quote(remote)} && echo deleted'))
        print(ssh('squeue -u tawal01 -o "%.10i %.30j %.8T"'))
        (run / 'cleanup.txt').write_text(f'{remote} deleted after verified collection\n')
        return
    out.mkdir(parents=True, exist_ok=False)
    queue = ssh('squeue -h -u tawal01 -o "%i %T %j"')
    assert f'wl_{a.attempt}' not in queue, 'still queued: ' + queue
    accounting = ssh(f'sacct --name=wl_{a.attempt} --format=JobID,JobName,State%30,ExitCode,Elapsed,NodeList,AllocTRES -P')
    (run / 'accounting.txt').write_text(accounting)
    members = ['COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'run.sbatch', 'logs', 'OUTPUTS.sha256', 'output', 'experiments']
    cmd = (f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && sha256sum -c MANIFEST.sha256 --quiet && '
           f'tar -czf collection.tar.gz ' + ' '.join(map(shlex.quote, members)) + ' && sha256sum collection.tar.gz > collection.tar.gz.sha256')
    subprocess.run(['ssh', 'tufts-login', cmd], check=True)
    for name in ['collection.tar.gz', 'collection.tar.gz.sha256']:
        subprocess.run(['scp', '-q', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.gz.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xzf', 'collection.tar.gz'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'MANIFEST.sha256', '--quiet'], cwd=out, check=True)
    log = sorted((out / 'logs').glob('*.out'))[-1].read_text()
    assert 'jax_backend=gpu' in log and 'ALL-DONE' in log and 'w_ladder_complete' in log, 'log incomplete'
    (out / 'VERIFIED.txt').write_text('collection.tar.gz sha256, OUTPUTS.sha256 and MANIFEST.sha256 verified locally; log has jax_backend=gpu, w_ladder_complete, ALL-DONE\n')
    print(out)
    print('Checksums verified; exact remote cleanup remains an explicit separate step (--delete).')


if __name__ == '__main__':
    main()
