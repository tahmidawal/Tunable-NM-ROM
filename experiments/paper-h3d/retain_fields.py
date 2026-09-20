"""Byte-exact split-tar retention and streaming restoration audit of saved fields."""
import argparse, hashlib, io, json, tarfile
from pathlib import Path

BLOCK=64*1024*1024
class SplitWriter:
    def __init__(self,path):self.path=path;self.index=0;self.handle=None;self.used=0;self.total=hashlib.sha256()
    def write(self,data):
        self.total.update(data);remaining=memoryview(data);count=len(data)
        while remaining:
            if self.handle is None:
                self.handle=(self.path/f'part{self.index:04d}.bin').open('wb');self.index+=1;self.used=0
            n=min(len(remaining),BLOCK-self.used);self.handle.write(remaining[:n]);remaining=remaining[n:];self.used+=n
            if self.used==BLOCK:self.handle.close();self.handle=None
        return count
    def close(self):
        if self.handle:self.handle.close();self.handle=None
class JoinReader:
    def __init__(self,paths):self.paths=iter(paths);self.handle=None
    def read(self,n):
        data=[];total=0
        while total<n:
            if self.handle is None:
                path=next(self.paths,None)
                if path is None:break
                self.handle=path.open('rb')
            chunk=self.handle.read(n-total)
            if not chunk:self.handle.close();self.handle=None;continue
            data.append(chunk);total+=len(chunk)
        return b''.join(data)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(destination):
    manifest=json.loads((destination/'manifest.json').read_text());parts=[destination/row['path'] for row in manifest['chunks']]
    total=hashlib.sha256()
    for path,row in zip(parts,manifest['chunks']):
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):total.update(block)
    assert total.hexdigest()==manifest['tar_sha256']
    files={row['path']:row for row in manifest['files']};checked=[]
    with tarfile.open(fileobj=JoinReader(parts),mode='r|') as archive:
        for member in archive:
            assert member.isfile() and member.name in files
            stream=archive.extractfile(member);h=hashlib.sha256();size=0
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block);size+=len(block)
            assert size==files[member.name]['bytes'] and h.hexdigest()==files[member.name]['sha256']
            checked.append(member.name)
    assert sorted(checked)==sorted(files)
    audit=dict(passed=True,files=len(checked),bytes=sum(f['bytes'] for f in files.values()),
        chunks=len(parts),chunk_hashes_verified=True,full_stream_hash_verified=True,every_restored_member_hash_verified=True)
    (destination/'restoration-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    assert args.attempt.isalnum();root=Path(__file__).resolve().parent
    source=root/'runs'/args.attempt/'archive/out/fields';destination=root/'retained-fields'/args.attempt
    if not args.verify_only:
        assert source.is_dir();destination.mkdir(parents=True,exist_ok=False);rows=[]
        writer=SplitWriter(destination)
        with tarfile.open(fileobj=writer,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(source.rglob('*')):
                if not path.is_file():continue
                relative=path.relative_to(source).as_posix();rows.append(dict(path=relative,bytes=path.stat().st_size,sha256=sha(path)))
                archive.add(path,arcname=relative,recursive=False)
        writer.close();chunks=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(destination.glob('part*.bin'))]
        manifest=dict(schema='heat3d-retained-fields-v1',attempt=args.attempt,source_relative=f'runs/{args.attempt}/archive/out/fields',
            format='concatenate parts in manifest order to obtain a POSIX tar archive; all members are relative field paths',
            original_outputs_manifest_sha256=sha(source.parent.parent/'OUTPUTS.sha256'),files=rows,chunks=chunks,tar_sha256=writer.total.hexdigest())
        (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify(destination)
if __name__=='__main__':main()
