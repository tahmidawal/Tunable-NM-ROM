"""Retain and verify exact run files in mirrorable chunks without a second large archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

CHUNK = 32 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


class Writer:
    def __init__(self, dest):
        self.dest, self.f, self.size = dest, None, 0
        self.parts, self.whole = [], hashlib.sha256()

    def finish_part(self):
        if self.f is not None:
            path = Path(self.f.name); self.f.close()
            self.parts.append(dict(file=path.name, bytes=path.stat().st_size, sha256=digest(path)))
            self.f = None

    def write(self, data):
        self.whole.update(data); size = len(data)
        while data:
            if self.f is None:
                self.f = (self.dest/f'run.tar.gz.part{len(self.parts):04d}').open('xb')
                self.size = 0
            n = min(len(data), CHUNK-self.size)
            self.f.write(data[:n]); self.size += n; data = data[n:]
            if self.size == CHUNK: self.finish_part()
        return size


class Reader(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.current = iter(paths), None

    def readable(self): return True

    def readinto(self, view):
        while True:
            if self.current is None:
                try: self.current = next(self.paths).open('rb')
                except StopIteration: return 0
            n = self.current.readinto(view)
            if n: return n
            self.current.close(); self.current = None

    def close(self):
        if self.current is not None: self.current.close()
        super().close()


def verify(dest):
    meta = json.loads((dest/'RETENTION.json').read_text())
    whole = hashlib.sha256()
    for item in meta['parts']:
        path = dest/item['file']
        assert path.stat().st_size == item['bytes'] <= CHUNK
        assert digest(path) == item['sha256']
        with path.open('rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''): whole.update(b)
    assert whole.hexdigest() == meta['archive_sha256']
    remaining = {item['path']: item for item in meta['files']}
    with io.BufferedReader(Reader([dest/x['file'] for x in meta['parts']])) as stream:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                assert member.isfile() and member.name in remaining, member.name
                want = remaining.pop(member.name)
                h = hashlib.sha256(); count = 0
                with archive.extractfile(member) as f:
                    for b in iter(lambda: f.read(1 << 20), b''): h.update(b); count += len(b)
                assert count == want['bytes'] and h.hexdigest() == want['sha256'], member.name
    assert not remaining
    return dict(files=len(meta['files']), parts=len(meta['parts']), archive_sha256=meta['archive_sha256'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source'); ap.add_argument('--dest', required=True)
    ap.add_argument('--verify-only', action='store_true'); args = ap.parse_args(); dest = Path(args.dest)
    if not args.verify_only:
        source = Path(args.source).resolve(); dest = dest.resolve()
        assert not dest.is_relative_to(source), 'Archive must be outside its source'
        files = sorted(p for p in source.rglob('*') if p.is_file()
                       and not {'cache', 'tmp', '__pycache__'}.intersection(p.relative_to(source).parts)
                       and p.name != 'input.tgz')
        assert files and all(not p.is_symlink() for p in files)
        dest.mkdir(parents=True, exist_ok=False)
        listing = [dict(path=str(p.relative_to(source)), bytes=p.stat().st_size, sha256=digest(p)) for p in files]
        writer = Writer(dest)
        with tarfile.open(fileobj=writer, mode='w|gz') as archive:
            for p in files: archive.add(p, arcname=str(p.relative_to(source)), recursive=False)
        writer.finish_part()
        meta = dict(schema=1, files=listing, parts=writer.parts, archive_sha256=writer.whole.hexdigest())
        (dest/'RETENTION.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(verify(dest), indent=2))


if __name__ == '__main__': main()
