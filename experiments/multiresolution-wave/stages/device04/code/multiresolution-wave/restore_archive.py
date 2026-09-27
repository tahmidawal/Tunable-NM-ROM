"""Verify ordered archive parts and restore a missing extracted cluster directory."""
import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path
import tarfile

class PartsReader(io.RawIOBase):
    def __init__(self,paths):
        self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,target):
        while True:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:return 0
            n=self.current.readinto(target)
            if n:return n
            self.current.close();self.current=None

def main(record):
    manifest=json.loads((record/'ARCHIVE.json').read_text())
    parts=[record/p for p in manifest['ordered_parts']]
    expected={line.split(maxsplit=1)[1].strip():line.split()[0] for line in (record/'ARCHIVE.sha256').read_text().splitlines()}
    joined=hashlib.sha256()
    for p in parts:
        one=hashlib.sha256()
        with p.open('rb') as f:
            while block:=f.read(1024*1024):one.update(block);joined.update(block)
        assert one.hexdigest()==expected[p.name],p
    assert joined.hexdigest()==manifest['original_sha256']
    with io.BufferedReader(PartsReader(parts)) as raw,tarfile.open(fileobj=raw,mode='r|gz') as archive:
        for member in archive:
            target=record/member.name
            if not target.resolve().is_relative_to(record.resolve()):raise RuntimeError('Unsafe archive member')
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            if not member.isfile():raise RuntimeError('Nonregular archive member')
            source=archive.extractfile(member)
            if target.exists():
                existing=hashlib.sha256(target.read_bytes()).digest();expected_member=hashlib.sha256()
                while block:=source.read(1024*1024):expected_member.update(block)
                if existing!=expected_member.digest():raise RuntimeError('Existing file differs; refusing overwrite: '+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as dest:shutil.copyfileobj(source,dest)
    for manifest in ('PULL.sha256','MANIFEST.sha256','RESULTS.sha256'):
        for line in (record/'cluster'/manifest).read_text().splitlines():
            sha,relative=line.split(maxsplit=1)
            p=record/'cluster'/relative.strip()
            h=hashlib.sha256()
            with p.open('rb') as f:
                while block:=f.read(1024*1024):h.update(block)
            assert h.hexdigest()==sha,p
    print('archive_and_extracted_manifests_verified')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);main(ap.parse_args().record)
