"""Checksum-collect one exact completed attempt; remote cleanup stays an explicit separate step.

The nine `opsckpt/*.pt` files (1.2 GB) are NOT collected: they are out-of-band INPUTS whose
SHA256 is in PROVENANCE.json and whose source copies are retained in the lanes they came from
(b-ladder-top's rule). The remote MANIFEST is verified in full before the archive is made, so
they are checked; they are simply not copied back.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/opstime_20260922'
LANE = 'experiments/ops-timing-panel'
EXCLUDED = None   # filled from operators.json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    global EXCLUDED
    import json
    EXCLUDED = [f"opsckpt/{c['name']}.pt"
                for c in json.loads((ROOT / LANE / 'operators.json').read_text())['checkpoints']]
    out = ROOT / LANE / 'runs' / a.attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    members = ['COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'run.sbatch', 'logs', 'OUTPUTS.sha256', 'output']
    for extra in ('experiments',):
        if subprocess.run(['ssh', 'tufts-login', f'test -e {shlex.quote(remote + "/" + extra)}']).returncode == 0:
            members.append(extra)
    cmd = (f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && '
           f'sha256sum -c MANIFEST.sha256 --quiet && tar -czf collection.tar.gz '
           + ' '.join(map(shlex.quote, members)) + ' && sha256sum collection.tar.gz > collection.tar.gz.sha256')
    subprocess.run(['ssh', 'tufts-login', cmd], check=True)
    for name in ['collection.tar.gz', 'collection.tar.gz.sha256']:
        subprocess.run(['scp', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.gz.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xzf', 'collection.tar.gz'], cwd=out, check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    manifest = [line for line in (out / 'MANIFEST.sha256').read_text().splitlines()
                if line.split('  ', 1)[-1] not in EXCLUDED]
    (out / 'MANIFEST.collected.sha256').write_text('\n'.join(manifest) + '\n')
    subprocess.run(['sha256sum', '-c', 'MANIFEST.collected.sha256', '--quiet'], cwd=out, check=True)
    (out / 'EXCLUDED-FROM-COLLECTION.txt').write_text(
        '\n'.join(EXCLUDED) + '\n\nVerified on the remote side by the full MANIFEST.sha256 before the archive '
        'was made; SHA256 recorded in PROVENANCE.json; source copy retained in the worktree it was staged from.\n')
    print(out)
    print('Checksums verified; exact remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
