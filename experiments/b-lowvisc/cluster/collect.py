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
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_lowvisc_20260917'
# The extraction npz (131072 x 512 f64 coefficients, ~540 MB) and the intermediate bank
# checkpoint are collected SEPARATELY and kept out of the compressed archive; their
# SHA256s are verified on both sides against OUTPUTS.sha256 and recorded beside the chunks.
# Nothing is excluded from the b-lowvisc gate archive: its largest arrays are the POD audit
# artifacts (tens of MB), which the independent audit needs and which the chunked archive holds.
EXCLUDED = []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/b-lowvisc/runs' / a.attempt / 'archive'
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
        subprocess.run(['scp', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.gz.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xzf', 'collection.tar.gz'], cwd=out, check=True)
    for x in EXCLUDED:
        (out / x).parent.mkdir(parents=True, exist_ok=True)
        # an excluded file that the job did not produce (e.g. a skipped stage) is simply absent
        subprocess.run(['scp', f'tufts-login:{remote}/{x}', str(out / x)], check=False)
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet'], cwd=out, check=True)
    (out / 'EXCLUDED-FROM-ARCHIVE.txt').write_text(
        '\n'.join(EXCLUDED) + '\n\nCollected separately and verified on BOTH sides by the full '
        'OUTPUTS.sha256. The extraction npz holds the whitened span coefficients of the 131072 '
        'picked training states in the seed bank (a deterministic function of the bank and the '
        'seed-regenerated data); its SHA256 is recorded beside the chunks and is needed only by '
        'the local training audit, so it is not duplicated into the Git-tracked archive chunks.\n')
    with (out / 'EXCLUDED-SHA256.txt').open('w') as fh:
        for x in EXCLUDED:
            if (out / x).exists():
                fh.write(f'{hashlib.sha256((out / x).read_bytes()).hexdigest()}  {x}\n')
    ck = sorted((out / 'output/train').glob('sep_hfit_*.pkl')) if (out / 'output/train').exists() else []
    if ck:
        # the seed checkpoint enters the sealed job from experiments/b-lowvisc/checkpoints/, with
        # its SHA256 taken from the job's own TRAIN-SHA256.txt (never recomputed silently)
        cdir = ROOT / 'experiments/b-lowvisc/checkpoints'
        cdir.mkdir(exist_ok=True)
        recorded = {ln.split()[1].split('/')[-1]: ln.split()[0] for ln in
                    (out / 'output/train/TRAIN-SHA256.txt').read_text().splitlines() if ln.strip()}
        for p in ck:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest == recorded[p.name], (p.name, digest, recorded.get(p.name))
            dest = cdir / p.name
            assert not dest.exists() or dest.read_bytes() == p.read_bytes(), f'{dest} exists and differs'
            dest.write_bytes(p.read_bytes())
            (cdir / (p.stem + '.sha256')).write_text(f'{digest}  {p.name}  attempt={a.attempt}\n')
            print('checkpoint ->', dest, digest[:16])
    print(out)
    print('Checksums verified; exact remote cleanup remains an explicit separate step.')


if __name__ == '__main__':
    main()
