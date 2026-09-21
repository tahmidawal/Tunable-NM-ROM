"""Checksum-collect one completed burgers-heldout attempt into runs/<attempt>/archive (git-ignored).

The two extraction npz of bh1 (training coefficients, 0.5-1 GB, regenerable from the seed by the same
job) are left out of the pull; their SHA256 stays in the pulled OUTPUTS.sha256. Every pulled file is
verified against OUTPUTS.sha256. Remote cleanup is an explicit separate step.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/bheld_20260921'
SKIP = ('output/x1024_sep_coeff_N256_K16_R1024.npz', 'output/x512_sep_coeff_N256_K16_R512.npz')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    out = ROOT / 'experiments/burgers-heldout/runs' / a.attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    excl = ' '.join(f'--exclude={shlex.quote(s)}' for s in SKIP)
    pre = (f'cd {shlex.quote(remote)} && ' + ('find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 && '
                                              if a.partial else '') +
           f'sha256sum -c OUTPUTS.sha256 --quiet && tar {excl} -cf collection.tar COMMIT.txt PROVENANCE.json '
           'MANIFEST.sha256 run.sbatch logs OUTPUTS.sha256 output && sha256sum collection.tar > collection.tar.sha256')
    subprocess.run(['ssh', 'tufts-login', pre], check=True)
    for name in ['collection.tar', 'collection.tar.sha256']:
        subprocess.run(['scp', f'tufts-login:{remote}/{name}', str(out / name)], check=True)
    subprocess.run(['sha256sum', '-c', 'collection.tar.sha256'], cwd=out, check=True)
    subprocess.run(['tar', '-xf', 'collection.tar'], cwd=out, check=True)
    lines = [l for l in (out / 'OUTPUTS.sha256').read_text().splitlines() if l.split()[1] not in SKIP]
    (out / 'OUTPUTS.pulled.sha256').write_text('\n'.join(lines) + '\n')
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.pulled.sha256', '--quiet'], cwd=out, check=True)
    (out / 'collection.tar').unlink()
    print(out)
    print('Checksums verified (skipped, hash kept:', ', '.join(SKIP), '); remote cleanup is a separate step.')


if __name__ == '__main__':
    main()
