"""Retain complete collected files in Git; split only oversized binaries.

Ordinary files are their own worktree archive. Every original is retained and
independently hashed from committed Git bytes before remote cleanup is allowed.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

LIMIT = 64 * 1024 * 1024


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def safe_path(base, name):
    path = base / name
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert path.resolve().is_relative_to(base.resolve())
    return path


def inventory(archive):
    return [p for p in sorted(archive.rglob('*')) if p.is_file()
            and not any(v in {'cache', 'tmp', '__pycache__'} for v in p.relative_to(archive).parts)]


def prepare(root, run, limit=LIMIT):
    manifest_path = run / 'DIRECT-RETENTION.json'
    assert not manifest_path.exists(), 'Do not overwrite a retention manifest.'
    rows = []
    for source in inventory(run / 'archive'):
        assert not source.is_symlink()
        relative = source.relative_to(run).as_posix()
        row = dict(path=relative, bytes=source.stat().st_size, sha256=sha(source), chunks=[])
        if row['bytes'] > limit:
            directory = run / 'large-artifacts' / hashlib.sha256(relative.encode()).hexdigest()
            directory.mkdir(parents=True, exist_ok=False)
            with source.open('rb') as handle:
                while data := handle.read(limit):
                    chunk = directory / f'part{len(row["chunks"]):05d}'
                    chunk.write_bytes(data)
                    row['chunks'].append(dict(path=chunk.relative_to(run).as_posix(), bytes=len(data),
                                               sha256=hashlib.sha256(data).hexdigest()))
        rows.append(row)
    assert rows
    value = dict(schema='heat3d-direct-git-retention-v1', original_files_unchanged=True,
                 scope='Every collected archive file except cache/tmp/__pycache__; no scientific files omitted.',
                 files=rows, bytes=sum(r['bytes'] for r in rows), chunk_limit=limit,
                 root_relative=run.relative_to(root).as_posix(),
                 restore_command='Use direct_retention.py ATTEMPT --restore COMMIT --member archive/RELATIVE_PATH')
    manifest_path.write_text(json.dumps(value, indent=2) + '\n')
    return value


class GitReader:
    def __init__(self, root, commit):
        self.root = root
        self.commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--verify', commit+'^{commit}'], text=True).strip()
        self.process = subprocess.Popen(['git', '-C', str(root), 'cat-file', '--batch'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read(self, path, sink=None, combined=None):
        key = f'{self.commit}:{path.relative_to(self.root).as_posix()}'
        assert '\n' not in key
        self.process.stdin.write((key+'\n').encode()); self.process.stdin.flush()
        header = self.process.stdout.readline().decode().split()
        assert len(header) == 3 and header[1] == 'blob', (key, header)
        remaining = size = int(header[2]); digest = hashlib.sha256()
        while remaining:
            block = self.process.stdout.read(min(8*1024*1024, remaining))
            assert block
            remaining -= len(block); digest.update(block)
            if sink is not None: sink.write(block)
            if combined is not None: combined.update(block)
        assert self.process.stdout.read(1) == b'\n'
        return digest.hexdigest(), size

    def close(self):
        self.process.stdin.close(); self.process.wait()
        assert self.process.returncode == 0


def verify(root, run, commit, restore_member=None):
    manifest_path = run / 'DIRECT-RETENTION.json'
    value = json.loads(manifest_path.read_text())
    assert value['root_relative'] == run.relative_to(root).as_posix()
    reader = GitReader(root, commit)
    checked = []
    try:
        assert reader.read(manifest_path) == (sha(manifest_path), manifest_path.stat().st_size)
        rows = value['files']
        if restore_member is not None:
            rows = [r for r in rows if r['path'] == restore_member]
            assert len(rows) == 1
        for row in rows:
            source = safe_path(run, row['path'])
            digest = hashlib.sha256(); size = 0; handle = None
            if restore_member is not None:
                assert not source.exists(), 'Refuse to overwrite a materialized file.'
                source.parent.mkdir(parents=True, exist_ok=True)
                handle = source.open('xb')
            try:
                if row['chunks']:
                    for chunk in row['chunks']:
                        h, n = reader.read(safe_path(run, chunk['path']), sink=handle, combined=digest)
                        assert (h, n) == (chunk['sha256'], chunk['bytes'])
                        size += n
                    actual = digest.hexdigest(), size
                else:
                    actual = reader.read(source, sink=handle)
                assert actual == (row['sha256'], row['bytes']), row['path']
            finally:
                if handle is not None: handle.close()
            if source.exists():
                assert (sha(source), source.stat().st_size) == actual
            checked.append(row['path'])
        if restore_member is None:
            assert set(checked) == {p.relative_to(run).as_posix() for p in inventory(run/'archive')}
    finally:
        reader.close()
    result = dict(passed=True, actual_git_blob_bytes_verified=True, commit=reader.commit,
                  files=len(checked), bytes=sum(r['bytes'] for r in rows), restored_member=restore_member,
                  complete_manifest_sha256=sha(manifest_path),
                  proof='Every ordinary Git blob and reconstructed large-file chunk stream independently matches the original SHA256 and size.')
    return result


def stage(root, run):
    value = json.loads((run/'DIRECT-RETENTION.json').read_text())
    paths = [run/'DIRECT-RETENTION.json']
    for row in value['files']:
        paths.extend(safe_path(run, c['path']) for c in row['chunks']) if row['chunks'] else paths.append(safe_path(run, row['path']))
    for start in range(0, len(paths), 100):
        subprocess.run(['git', '-c', 'gc.auto=0', '-C', str(root), 'add', '-f', '--'] +
                       [str(p.relative_to(root)) for p in paths[start:start+100]], check=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('attempt')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--stage', action='store_true')
    action.add_argument('--verify'); action.add_argument('--restore')
    parser.add_argument('--member'); args = parser.parse_args()
    assert args.attempt.isalnum()
    lane = Path(__file__).resolve().parent; root = lane.parents[1]; run = lane/'runs'/args.attempt
    if args.prepare or args.stage:
        assert json.loads((run/'COLLECTED.json').read_text())['checksums_verified']
        assert json.loads((run/'audit-local.json').read_text())['passed']
    if args.prepare:
        value = prepare(root, run)
        print(json.dumps(dict(files=len(value['files']), bytes=value['bytes'],
                              large_files=sum(bool(r['chunks']) for r in value['files']))))
    elif args.stage: stage(root, run)
    else:
        assert bool(args.restore) == bool(args.member)
        result = verify(root, run, args.verify or args.restore, args.member)
        destination = run/('DIRECT-RETENTION-AUDIT.json' if args.verify else 'DIRECT-RESTORE-AUDIT.json')
        destination.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result))


if __name__ == '__main__': main()
