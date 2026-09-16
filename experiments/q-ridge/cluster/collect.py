"""Checksum-collect one exact completed attempt; remote cleanup stays explicit.

`output/bank_G.npz` (266 MB) is collected SEPARATELY and kept out of the compressed
archive, exactly as `b-ladder-top`'s collector treats the 429 MB FNO checkpoint: it is a
deterministic function of the frozen checkpoint rather than a result, its SHA256 is
recorded in `result.json` and verified on both sides against `OUTPUTS.sha256`, and it is
needed only by the local R3 audit. Everything else is archived byte for byte.
"""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/q_ridge_20260916'
EXCLUDED = ['output/bank_G.npz']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/q-ridge/runs' / a.attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    members = ['COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'run.sbatch', 'logs',
               'OUTPUTS.sha256', 'output', 'experiments']
    cmd = (f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && '
           f'sha256sum -c MANIFEST.sha256 --quiet && tar -czf collection.tar.gz '
           + ' '.join(f'--exclude={shlex.quote(x)}' for x in EXCLUDED) + ' '
           + ' '.join(map(shlex.quote, members))
           + ' && sha256sum collection.tar.gz > collection.tar.gz.sha256')
    subprocess.run(['ssh', 'tufts-login', cmd], check=True)
    for name in ['collection.tar.gz', 'collection.tar.gz.sha256']:
        subprocess.run(['scp', f'{remote}/{name}'.join(['tufts-login:', '']), str(out / name)],
                       check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.gz.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xzf', 'collection.tar.gz'], cwd=out, check=True)
    (out / 'output').mkdir(exist_ok=True)
    subprocess.run(['scp', f'tufts-login:{remote}/output/bank_G.npz',
                    str(out / 'output/bank_G.npz')], check=True)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    (out / 'EXCLUDED-FROM-ARCHIVE.txt').write_text(
        '\n'.join(EXCLUDED) + '\n\nCollected separately and verified on BOTH sides by the full '
        'OUTPUTS.sha256; its SHA256 is also recorded in result.json under "bank_G". It is the '
        'frozen coordinate bank on the query grid, a deterministic function of the checkpoint, '
        'needed only by the local R3 audit, so it is not duplicated into the Git-tracked '
        'archive chunks.\n')
    digest = hashlib.sha256((out / 'output/bank_G.npz').read_bytes()).hexdigest()
    (out / 'BANK-SHA256.txt').write_text(f'{digest}  output/bank_G.npz\n')
    print(out)
    print('Checksums verified; exact remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
