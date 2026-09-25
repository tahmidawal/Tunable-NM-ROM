"""Checksum-collect one completed t2-burgers-test attempt into runs/<attempt>/archive (git-ignored).

    python cluster/collect.py <attempt>

Verifies output/ remotely against the job's OUTPUTS.sha256, rsyncs output/, logs/ and the job's provenance files,
then re-verifies every file locally. Remote cleanup is a separate, explicit step (rm -rf of the attempt directory).
"""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/t2-burgers-test'
NAMESPACE = '/cluster/tufts/paralab/tawal01/t2btest_20260925'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    a = p.parse_args()
    assert a.attempt.isalnum()
    remote = f'{NAMESPACE}/{a.attempt}'
    lane_r = f'{remote}/{LANE}'
    local = ROOT / LANE / 'runs' / a.attempt / 'archive'
    local.mkdir(parents=True, exist_ok=False)
    subprocess.run(['ssh', 'tufts-login', f'cd {shlex.quote(lane_r)} && sha256sum -c OUTPUTS.sha256 --quiet'], check=True)
    subprocess.run(['rsync', '-a', f'tufts-login:{lane_r}/output', f'tufts-login:{lane_r}/OUTPUTS.sha256', str(local)], check=True)
    (local / 'job').mkdir()
    for f in ('logs', 'run.sbatch', 'COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256'):
        subprocess.run(['rsync', '-a', f'tufts-login:{remote}/{f}', str(local / 'job')], check=True)
    bad = []
    n = 0
    for line in (local / 'OUTPUTS.sha256').read_text().splitlines():
        h, name = line.split(None, 1)
        n += 1
        if sha(local / name.strip()) != h:
            bad.append(name)
    assert not bad, bad[:10]
    print('collected', a.attempt, n, 'files verified ->', local)


if __name__ == '__main__':
    main()
