"""Independent read-only recovery of a complete archive retained in Git histories."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

p = argparse.ArgumentParser()
p.add_argument('--artifact', type=Path, required=True)
p.add_argument('--cold-source', required=True)
p.add_argument('--cold-mirror', required=True)
p.add_argument('--retention-sha256', required=True)
a = p.parse_args()
repo = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle-rom')
root = a.artifact.resolve()
assert root == repo / 'experiments/ns-obstacle-rom/artifacts/neural-pod-guided-diagnostic09'
rel = root.relative_to(repo).as_posix()

def git(*args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', '-C', str(repo), *args])

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load(name):
    return json.loads((root / name).read_bytes())

r, cold, measurement = load('RETENTION.json'), load('COLD-RECOVERY.json'), load('MEASUREMENT.json')
release = load('RELEASE.json')
assert sha((root / 'RETENTION.json').read_bytes()) == a.retention_sha256
assert sha((root / 'RELEASE.json').read_bytes()) == r['collection_release_sha256']
assert sha((root / 'MEASUREMENT.json').read_bytes()) == release['measurement_sha256']
assert release['authorized'] is True and release['single_execution'] is True
assert release['automatic_retry'] is False
assert r['schema'] == 'ns-obstacle-git-object-first-retention-v1'
assert r['working_archive_or_chunks_created'] is False
assert cold['working_chunks_never_created'] is True
assert cold['chunks'] == r['chunks']
assert cold['artifact_relative_path'] == rel
assert cold['archive_sha256'] == r['archive_sha256'] == measurement['archive_sha256']
assert cold['archive_bytes'] == r['archive_bytes'] == measurement['archive_bytes']
assert r['job_id'] == measurement['job_id'] == 6661438
assert release['job_id'] == r['job_id']
assert release['archive_sha256'] == r['archive_sha256']
assert release['archive_bytes'] == r['archive_bytes']
terminal = [row.split('|') for row in measurement['terminal_status'].splitlines()
            if row.split('|')[0] == str(r['job_id'])]
assert len(terminal) == 1 and terminal[0][1] == 'COMPLETED' and terminal[0][3] == '0:0'
source, mirror = cold['archive_source_commit'], cold['advertised_archive_mirror_commit']
advertised = git('ls-remote', 'origin', 'refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom').split()[0].decode()
for older, newer in [(source, a.cold_source), (mirror, a.cold_mirror), (a.cold_mirror, advertised)]:
    subprocess.run(['git', '-c', 'gc.auto=0', '-C', str(repo), 'merge-base', '--is-ancestor', older, newer], check=True)
for rev in (a.cold_source, a.cold_mirror):
    assert not git('ls-tree', '-r', '--name-only', rev, '--', rel + '/archive-chunks')

# All warm metadata/receipts are bound to both final histories; no archive expansion.
for f in root.rglob('*'):
    if f.is_file():
        assert not f.is_symlink()
        name = f.relative_to(repo).as_posix()
        raw = f.read_bytes()
        for rev in (a.cold_source, a.cold_mirror):
            assert git('show', rev + ':' + name) == raw

rows = r['chunks']
assert rows and sum(x['bytes'] for x in rows) == r['archive_bytes']
assert all(type(x['bytes']) is int and 0 < x['bytes'] <= 8 * 2**20 for x in rows)
assert all(x['bytes'] == 8 * 2**20 for x in rows[:-1])
for rev in (source, mirror):
    whole, count = hashlib.sha256(), 0
    for i, row in enumerate(rows):
        name = rel + '/archive-chunks/collection.tar.gz.part-%03d' % i
        oid = git('rev-parse', rev + ':' + name).strip().decode()
        assert oid == row['oid']
        raw = git('cat-file', 'blob', oid)
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
        whole.update(raw)
        count += len(raw)
    assert count == r['archive_bytes'] and whole.hexdigest() == r['archive_sha256']

class Reader(io.RawIOBase):
    def __init__(self):
        self.remaining = iter(rows)
        self.buffer = b''
        self.offset = 0

    def readable(self):
        return True

    def readinto(self, target):
        if self.offset == len(self.buffer):
            row = next(self.remaining, None)
            if row is None:
                return 0
            self.buffer = git('cat-file', 'blob', row['oid'])
            assert len(self.buffer) == row['bytes'] and sha(self.buffer) == row['sha256']
            self.offset = 0
        n = min(len(target), len(self.buffer) - self.offset)
        target[:n] = self.buffer[self.offset:self.offset + n]
        self.offset += n
        return n

seen, ledger = {}, None
with io.BufferedReader(Reader()) as stream, tarfile.open(fileobj=stream, mode='r|gz') as archive:
    for member in archive:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        assert path.parts and path.parts[0] == root.name
        if member.isdir():
            continue
        assert member.isfile()
        name = PurePosixPath(*path.parts[1:]).as_posix()
        assert name != '.' and name not in seen
        h, count, pieces = hashlib.sha256(), 0, []
        with archive.extractfile(member) as data:
            for block in iter(lambda: data.read(2**20), b''):
                count += len(block)
                h.update(block)
                if name == 'COLLECTION-SHA256SUMS':
                    assert count <= 2**20
                    pieces.append(block)
        assert count == member.size
        seen[name] = {'bytes': count, 'sha256': h.hexdigest()}
        if name == 'COLLECTION-SHA256SUMS':
            ledger = b''.join(pieces)

assert seen == load('ORIGINAL-FILE-INVENTORY.json')
assert len(seen) == r['original_files'] == measurement['regular_files']
assert sum(x['bytes'] for x in seen.values()) == measurement['expanded_bytes']
assert ledger is not None and sha(ledger) == r['ledger_sha256'] == measurement['ledger_sha256']
pairs = [line.split('  ', 1) for line in ledger.decode().splitlines()]
assert len(pairs) == len({name for _, name in pairs})
expected = {name: digest for digest, name in pairs}
assert set(expected) == set(seen) - {'COLLECTION-SHA256SUMS'}
assert all(seen[name]['sha256'] == digest for name, digest in expected.items())
metadata_names = [row['path'] for row in measurement['metadata_files']]
assert len(metadata_names) == len(set(metadata_names))
assert set(metadata_names) == {f.relative_to(root / 'metadata').as_posix()
                               for f in (root / 'metadata').rglob('*') if f.is_file()}
assert sum(row['bytes'] for row in measurement['metadata_files']) == measurement['metadata_extraction_bytes'] == r['metadata_extracted_bytes']
assert r['metadata_extracted_bytes'] <= release['metadata_limit_bytes']
for row in measurement['metadata_files']:
    name = PurePosixPath(row['path'])
    assert not name.is_absolute() and '..' not in name.parts
    f = root / 'metadata' / row['path']
    assert f.resolve().is_relative_to(root / 'metadata')
    assert f.stat().st_size == row['bytes'] == seen[row['path']]['bytes']
    assert sha(f.read_bytes()) == seen[row['path']]['sha256']
assert not (root / 'archive-chunks').exists()
print(json.dumps({'passed': True, 'archive_source_commit': source,
                  'archive_mirror_commit': mirror, 'cold_source_commit': a.cold_source,
                  'cold_mirror_commit': a.cold_mirror, 'archive_sha256': r['archive_sha256'],
                  'verified_regular_files': len(seen), 'chunk_count': len(rows),
                  'both_histories_complete_compressed_stream_verified': True,
                  'retention_sha256': a.retention_sha256,
                  'bulk_scientific_extraction': False, 'remote_deletion_performed': False}, indent=2))
