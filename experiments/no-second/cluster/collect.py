"""Checksum-collect one exact completed no-second attempt; remote cleanup stays explicit.

    python cluster/collect.py <attempt>            # verify remotely, tar, scp, verify, extract
    python cluster/collect.py <attempt> --cleanup  # ONLY after audit.py passed: rm the remote dir

Excluded from the archive, exactly as the parent lane excluded them: the final-epoch
optimiser state (`last.pt`; training is not resumable and `best.pt` carries the
selected weights, normalisation and its own hash), package caches, the training
cases and refinement anchors (already archived and hash-linked by the Burgers lane)
and the solver-audit sidecars. Everything else — every `best.pt`, every saved
validation and cohort prediction, every history, the timing repetition arrays, the
model-facing validation cases and the rebuilt cohort — is archived byte for byte
and then chunked into 64 MiB Git-tracked parts by `preserve` below.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / 'experiments/no-second'
NAMESPACE = '/cluster/tufts/paralab/tawal01/no_second_20260917'
EXCLUDES = ['--exclude=*/last.pt', '--exclude=*.solver.npz', '--exclude=cache', '--exclude=tmp',
            '--exclude=data/train', '--exclude=data/refinement', '--exclude=code/__pycache__']


def ssh(command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'tufts-login', command],
                                   text=True, timeout=3600)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def collect(attempt):
    remote = f'{NAMESPACE}/{attempt}'
    out = LANE / 'runs' / attempt / 'archive'
    out.mkdir(parents=True, exist_ok=False)
    log = ssh(f'cd {shlex.quote(remote)} && tail -1 logs/*.out')
    assert 'ALL-DONE' in log, log
    ssh(f'cd {shlex.quote(remote)} && sha256sum -c OUTPUTS.sha256 --quiet && sha256sum -c MANIFEST.sha256 --quiet && '
        f'tar -czf {shlex.quote(remote + ".tar.gz")} {" ".join(EXCLUDES)} -C {shlex.quote(NAMESPACE)} {attempt} && '
        f'sha256sum {shlex.quote(remote + ".tar.gz")} > {shlex.quote(remote + ".tar.gz.sha256")}')
    for name in (attempt + '.tar.gz', attempt + '.tar.gz.sha256'):
        subprocess.run(['scp', '-q', f'tufts-login:{NAMESPACE}/{name}', str(out / name)], check=True)
    expected = (out / (attempt + '.tar.gz.sha256')).read_text().split()[0]
    assert sha(out / (attempt + '.tar.gz')) == expected
    subprocess.run(['tar', '-xzf', attempt + '.tar.gz'], cwd=out, check=True)
    root = out / attempt
    for line in (root / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha(root / name) == digest, name
    missing = []
    for line in (root / 'OUTPUTS.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if (root / name).exists():
            assert sha(root / name) == digest, name
        else:
            missing.append(name)
    assert all(n.endswith('last.pt') or '/cache/' in n for n in missing), missing
    (out / 'ARCHIVE.sha256').write_text(f'{expected}  {attempt}.tar.gz\n')
    (out / 'EXCLUDED-FROM-ARCHIVE.txt').write_text('\n'.join(EXCLUDES) + '\n\nOUTPUTS.sha256 entries not archived:\n'
                                                    + '\n'.join(missing) + '\n')
    print(out)
    print('Checksums verified; run audit.py, then preserve, then --cleanup explicitly.')


def preserve(attempt):
    out = LANE / 'runs' / attempt / 'archive'
    archive = out / (attempt + '.tar.gz')
    expected = (out / 'ARCHIVE.sha256').read_text().split()[0]
    assert sha(archive) == expected
    chunks = LANE / 'runs' / attempt / 'archive-parts'
    chunks.mkdir()
    manifest = dict(archive_sha256=expected, exclusions=EXCLUDES, parts=[])
    with archive.open('rb') as stream:
        index = 0
        while data := stream.read(64 * 1024 * 1024):
            path = chunks / f'part-{index:04d}'
            path.write_bytes(data)
            manifest['parts'].append(dict(path=path.name, sha256=sha(path), bytes=len(data)))
            index += 1
    (chunks / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive.unlink()
    print(chunks, len(manifest['parts']), 'parts')


def cleanup(attempt):
    remote = f'{NAMESPACE}/{attempt}'
    audit = LANE / 'runs' / attempt / 'audit.json'
    assert json.loads(audit.read_text())['passed'] is True, 'audit must pass before remote cleanup'
    assert (LANE / 'runs' / attempt / 'archive-parts/manifest.json').exists(), 'preserve first'
    ssh(f'rm -rf -- {shlex.quote(remote)} {shlex.quote(remote + ".tar.gz")} {shlex.quote(remote + ".tar.gz.sha256")}')
    print('removed', remote)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt')
    parser.add_argument('--preserve', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    assert args.attempt.isalnum()
    if args.cleanup:
        cleanup(args.attempt)
    elif args.preserve:
        preserve(args.attempt)
    else:
        collect(args.attempt)
