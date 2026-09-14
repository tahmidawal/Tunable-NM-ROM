"""Verify ordered archive chunks and restore exact scientific bytes safely."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024*1024): h.update(chunk)
    return h.hexdigest()


class PartsReader(io.RawIOBase):
    def __init__(self, paths): self.paths, self.current = iter(paths), None
    def readable(self): return True
    def readinto(self, target):
        while True:
            if self.current is None:
                try: self.current = next(self.paths).open("rb")
                except StopIteration: return 0
            n = self.current.readinto(target)
            if n: return n
            self.current.close(); self.current = None
    def close(self):
        if self.current is not None: self.current.close()
        super().close()


def restore(record):
    metadata = json.loads((record/"ARCHIVE.json").read_text())
    joined = hashlib.sha256(); parts = []
    for part in metadata["parts"]:
        path = record/part["path"]
        assert path.resolve().is_relative_to(record.resolve())
        assert sha256(path) == part["sha256"] and path.stat().st_size == part["bytes"]
        with path.open("rb") as handle:
            while chunk := handle.read(1024*1024): joined.update(chunk)
        parts.append(path)
    assert joined.hexdigest() == metadata["joined_sha256"]
    with io.BufferedReader(PartsReader(parts)) as raw, tarfile.open(fileobj=raw, mode="r|*") as archive:
        for member in archive:
            target = record/"archive"/member.name
            if not target.resolve().is_relative_to((record/"archive").resolve()): raise RuntimeError("Unsafe archive member")
            if member.isdir(): target.mkdir(parents=True, exist_ok=True); continue
            if not member.isfile(): raise RuntimeError("Nonregular archive member")
            source = archive.extractfile(member)
            if target.exists():
                expected = hashlib.sha256()
                while chunk := source.read(1024*1024): expected.update(chunk)
                if sha256(target) != expected.hexdigest(): raise RuntimeError("Existing file differs; refusing overwrite: "+str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as dest: shutil.copyfileobj(source, dest)
    manifest = record/"archive"/"ARCHIVE.sha256"
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = record/"archive"/relative
        assert path.resolve().is_relative_to((record/"archive").resolve())
        assert sha256(path) == expected, relative
    return dict(archive_verified=True, members_verified=len(manifest.read_text().splitlines()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path)
    print(json.dumps(restore(parser.parse_args().record)))
