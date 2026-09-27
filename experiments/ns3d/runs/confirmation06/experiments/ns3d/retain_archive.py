"""Durably retain one collected NS attempt, then verify a complete tar roundtrip.

Does not remove local or remote data. The caller separately checks scheduler
quiescence and all scientific audits before deleting an exact remote attempt.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,subprocess,tarfile
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


class Joined(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.current is None:
                path=next(self.paths,None)
                if path is None:return 0
                self.current=path.open('rb')
            count=self.current.readinto(buffer)
            if count:return count
            self.current.close();self.current=None
    def close(self):
        if self.current is not None:self.current.close()
        super().close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('collected');parser.add_argument('archive');parser.add_argument('--job',type=int,required=True);a=parser.parse_args()
    source=Path(a.collected).resolve();archive=Path(a.archive).resolve();archive.mkdir(parents=True,exist_ok=False)
    expected={str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    tar=subprocess.Popen(['tar','-cf','-','-C',str(source),'.'],stdout=subprocess.PIPE)
    split=subprocess.run(['split','-b','64M','-',str(archive/'collection.tar.part')],stdin=tar.stdout)
    tar.stdout.close();status=tar.wait()
    if status or split.returncode:raise RuntimeError('archive creation failed')
    paths=sorted(archive.glob('collection.tar.part*'));parts=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
    observed={}
    with io.BufferedReader(Joined(paths),buffer_size=16*1024*1024) as stream,tarfile.open(fileobj=stream,mode='r|') as bundle:
        for member in bundle:
            if not member.isfile():continue
            name=member.name.removeprefix('./');h=hashlib.sha256();f=bundle.extractfile(member)
            for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
            observed[name]=h.hexdigest()
    passed=expected==observed
    report=dict(format='concatenated POSIX tar,64MiB split parts',source_job=a.job,
        passed=passed,parts=parts,retained_files=len(expected),retained_bytes=sum(p['bytes'] for p in parts),
        roundtrip_verified_files=len(observed),files=expected)
    (archive/'archive.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise RuntimeError('archive roundtrip mismatch')
    # These immutable files have been checksum-read multiple times. Releasing
    # only their page-cache entries leaves GPU smoke memory available to others.
    released=0
    for path in [*paths,*[p for p in source.rglob('*') if p.is_file() and p.stat().st_size>=64*1024*1024]]:
        with path.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        released+=path.stat().st_size
    print(json.dumps(dict(passed=passed,files=len(expected),parts=len(parts),retained_bytes=report['retained_bytes'],page_cache_advised_bytes=released)))


if __name__=='__main__':main()
