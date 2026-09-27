"""Lossless split archives of paper evidence with verification from Git bytes.

Every prediction/reference array is retained. Explicitly listed regenerable
training caches and optimizer states are excluded, while remaining untouched
in the original local checksum-covered collection. Duplicate byte streams are
stored as ordinary tar hard links. No cluster operation is performed here.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[1]
CHUNK=64*1024*1024


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


class SplitWriter:
    def __init__(self,path):
        self.path=path;self.parts=[];self.stream=None;self.total=hashlib.sha256();self.position=0
    def write(self,data):
        length=len(data);self.total.update(data);self.position+=length
        while data:
            if self.stream is None:
                self.name=f'scientific.tar.part{len(self.parts):04d}'
                self.stream=(self.path/self.name).open('wb');self.hash=hashlib.sha256();self.count=0
            take=min(CHUNK-self.count,len(data));block=data[:take];data=data[take:]
            self.stream.write(block);self.hash.update(block);self.count+=take
            if self.count==CHUNK:self.finish_part()
        return length
    def tell(self):return self.position
    def flush(self):
        if self.stream:self.stream.flush()
    def finish_part(self):
        self.stream.flush();os.fdatasync(self.stream.fileno())
        os.posix_fadvise(self.stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        self.stream.close();self.parts.append(dict(path=self.name,bytes=self.count,sha256=self.hash.hexdigest()));self.stream=None
    def close(self):
        if self.stream:self.finish_part()


def excluded(path):
    if '__pycache__' in path.parts:return 'Python bytecode cache'
    if path.name=='training_fields.npy':return 'regenerable dense training cache; retained in original local collection'
    if path.name.endswith('latest.pkl') or path.name in ('optimizer.pkl','optimizer_state.pkl'):
        return 'optimizer continuation state; selected trained checkpoints are retained separately'
    return None


def create(attempt):
    run=EXP/'runs'/attempt;source=run/'collected';target=run/'scientific-archive'
    assert (source/'OUTPUTS.sha256').exists()
    source_manifest='COLLECTION.sha256' if (source/'COLLECTION.sha256').exists() else 'OUTPUTS.sha256'
    target.mkdir(exist_ok=False);files=[];omitted=[];seen={};writer=SplitWriter(target)
    with tarfile.open(fileobj=writer,mode='w|',format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source.rglob('*')):
            if not path.is_file():continue
            relative=str(path.relative_to(source));reason=excluded(path.relative_to(source))
            if reason:
                omitted.append(dict(path=relative,bytes=path.stat().st_size,reason=reason));continue
            sha=digest(path);size=path.stat().st_size;key=(sha,size)
            info=tarfile.TarInfo(relative);info.size=size;info.mode=0o644;info.mtime=0
            record=dict(path=relative,bytes=size,sha256=sha)
            if key in seen:
                info.type=tarfile.LNKTYPE;info.linkname=seen[key];info.size=0;record['hardlink_to']=seen[key];archive.addfile(info)
            else:
                with path.open('rb') as stream:
                    archive.addfile(info,stream)
                    os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                seen[key]=relative
            files.append(record)
            if len(files)%100==0:print('ARCHIVE',attempt,len(files),writer.tell(),flush=True)
    writer.close()
    manifest=dict(schema='lossless-scientific-tar-split-v1',attempt=attempt,source_collection=str(source),
        source_collection_manifest=source_manifest,source_collection_sha256=digest(source/source_manifest),files=files,excluded=omitted,
        parts=writer.parts,archive_sha256=writer.total.hexdigest(),archive_bytes=writer.tell(),
        retained_uncompressed_file_bytes=sum(v['bytes'] for v in files),
        scope='all original timed predictions, references, raw invocation records, scientific metadata, source and selected checkpoints; listed training/optimizer caches remain local-only')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','parts','excluded')},indent=2))


class GitReader:
    def __init__(self,commit,directory,parts):
        self.commit=commit;self.directory=directory;self.parts=parts;self.index=0;self.proc=None;self.total=hashlib.sha256();self.bytes=0;self.verified=[]
    def read(self,size=-1):
        if size<0:size=8*1024*1024
        chunks=[];remaining=size
        while remaining:
            if self.proc is None:
                if self.index==len(self.parts):break
                self.part=self.parts[self.index];path=self.directory/self.part['path']
                self.proc=subprocess.Popen(['git','-C',str(ROOT),'show',self.commit+':'+str(path.relative_to(ROOT))],stdout=subprocess.PIPE)
                self.part_hash=hashlib.sha256();self.part_bytes=0
            data=self.proc.stdout.read(remaining)
            if not data:
                assert self.proc.wait()==0
                assert self.part_hash.hexdigest()==self.part['sha256'] and self.part_bytes==self.part['bytes']
                self.verified.append(self.part['path']);self.proc=None;self.index+=1;continue
            self.part_hash.update(data);self.part_bytes+=len(data);self.total.update(data);self.bytes+=len(data)
            chunks.append(data);remaining-=len(data)
        return b''.join(chunks)


def verify(attempt,commit):
    directory=EXP/'runs'/attempt/'scientific-archive';relative=str((directory/'manifest.json').relative_to(ROOT))
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+relative]);manifest=json.loads(raw)
    expected={v['path']:v for v in manifest['files']};seen={};reader=GitReader(commit,directory,manifest['parts'])
    with tarfile.open(fileobj=reader,mode='r|') as archive:
        for member in archive:
            assert member.name in expected and member.name not in seen
            record=expected[member.name]
            if member.islnk():
                assert member.linkname in seen and member.linkname==record['hardlink_to']
                sha,size=seen[member.linkname]
            else:
                assert member.isfile();stream=archive.extractfile(member);value=hashlib.sha256();size=0
                for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block);size+=len(block)
                sha=value.hexdigest()
            assert sha==record['sha256'] and size==record['bytes'],member.name
            seen[member.name]=(sha,size)
    while reader.read(8*1024*1024):pass
    assert len(seen)==len(expected) and reader.total.hexdigest()==manifest['archive_sha256'] and reader.bytes==manifest['archive_bytes']
    answer=dict(passed=True,commit=commit,manifest_sha256=hashlib.sha256(raw).hexdigest(),files=len(seen),
        parts=len(reader.verified),archive_bytes=reader.bytes,restored_file_bytes=sum(v[1] for v in seen.values()),
        scope='every scientific file restored and hashed from the concatenated actual committed Git blobs, including lossless hard-link duplicates; no local array used as proof')
    (directory/'git-restore-audit.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['create','verify']);parser.add_argument('attempt');parser.add_argument('--commit',default='HEAD');args=parser.parse_args()
    if args.mode=='create':create(args.attempt)
    else:verify(args.attempt,args.commit)


if __name__=='__main__':main()
