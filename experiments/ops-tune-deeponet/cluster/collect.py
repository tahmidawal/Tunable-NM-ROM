"""Checksum-collect one exact completed ops-tune-deeponet attempt; remote cleanup stays explicit.

Inherited from `experiments/ops-deeponet-b2d/cluster/collect.py` with the lane path, the
namespace, and **an explicit archive inventory** (design audit finding 5/M15): this lane mounts
its data as read-only symlinks, and `tar -czf` preserves a symlink rather than its target, so a
plain copy of the parent would archive `data/pinned/validation` as a dangling absolute cluster
path and local recomputation could not read it. `-h` dereferences, and the exclusions then
decide exactly what that pulls in.

    python cluster/collect.py <attempt>            # verify remotely, tar, scp, verify, extract
    python cluster/collect.py <attempt> --cleanup  # ONLY after audit.py passed: rm the remote dir

**Archived** (what an independent local recomputation of every reported number needs): every
`best.pt`, every saved validation and cohort prediction, every `result.json`, `history.json`,
`provenance.json` and merged arm config, the worker and selection records, the **materialised**
32 validation cases and their index, the rebuilt 8-case cohort, the job logs, `code/` and the
manifests. Plus, for the extended training data, its **index and generation report** — its
fields are not archived because they are 17 GB and are regenerable from the recorded case
seeds, commit and reference setting, which the index carries.

**Excluded**: `last.pt` (training is not resumable; `best.pt` carries the selected weights,
normalisation and its own hash), the 34 GB extended cache's fields and packed pool, the pinned
training cases and refinement anchors (archived and hash-linked by the Burgers lane), the
solver-audit sidecars of the pinned cache, and package caches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / 'experiments/ops-tune-deeponet'
NAMESPACE = '/cluster/tufts/paralab/tawal01/opstune_don_20260922'
EXCLUDES = ['--exclude=*/last.pt', '--exclude=*.solver.npz', '--exclude=cache', '--exclude=tmp',
            '--exclude=data/pinned/train', '--exclude=data/pinned/refinement',
            '--exclude=data/ext/pool', '--exclude=data/ext/train/*.npz',
            '--exclude=code/__pycache__']


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
        # -h: the data mounts are symlinks; without it the archive carries dangling links to
        # cluster-only paths and nothing can be recomputed locally (design audit finding 5).
        f'tar -czhf {shlex.quote(remote + ".tar.gz")} {" ".join(EXCLUDES)} -C {shlex.quote(NAMESPACE)} {attempt} && '
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
    allowed = ('last.pt', '.solver.npz')
    assert all(n.endswith(allowed) or '/cache/' in n or '/data/ext/' in n or '/data/pinned/train/' in n
               or '/data/pinned/refinement/' in n for n in missing), missing
    # What local recomputation needs must actually be here, dereferenced.
    for required in ('data/pinned/validation/index.json', 'out/selection.json', 'out/worker.json'):
        assert (root / required).is_file() and not (root / required).is_symlink(), required
    assert len(list((root / 'data/pinned/validation').glob('*.npz'))) == 32, 'validation cases not materialised'
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
        while data := stream.read(45 * 1024 * 1024):
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
