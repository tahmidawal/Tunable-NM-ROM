"""Read exact committed inputs, including explicitly declared recoverable archive members.

Archive recovery uses historical Git blobs and a reviewed full-archive certificate;
it never silently falls back from a missing working file to unrelated local bytes.
"""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import hashlib, json, subprocess, tarfile, tempfile
import git_archive_recovery

def relative(value):
    path=PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('input path must be confined and relative')
    return str(path)

class SourceInputs:
    def __init__(self, repo, prefix, commit, reserve_bytes=10*1024**3):
        self.repo=Path(repo);self.prefix=relative(prefix);self.commit=commit
        self.reserve_bytes=reserve_bytes;self.temporary=None;self.archives={}
    def __enter__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='source-recovery-',dir=self.repo)
        return self
    def __exit__(self,*args):
        self.temporary.cleanup()
    def committed(self,path):
        return subprocess.check_output(['git','-C',str(self.repo),'show',self.commit+':'+self.prefix+'/'+relative(path)])
    def read(self,row):
        if 'archive_source' not in row:
            data=self.committed(row['source'])
        else:
            declaration=row['archive_source'];raw=self.committed(declaration['manifest_source'])
            digest=hashlib.sha256(raw).hexdigest()
            if digest!=declaration['manifest_sha256']:raise ValueError('archive recovery manifest changed')
            if digest not in self.archives:
                root=Path(self.temporary.name);manifest=root/(digest+'.json');manifest.write_bytes(raw)
                output=root/(digest+'.tar.gz')
                git_archive_recovery.recover(SimpleNamespace(repo=self.repo,manifest=manifest,output=output,reserve_bytes=self.reserve_bytes))
                self.archives[digest]=output
            with tarfile.open(self.archives[digest],'r:gz') as archive:
                member=archive.getmember(relative(declaration['member']))
                if not (member.isfile() or member.islnk()):raise ValueError('input is not a regular archived member')
                if member.islnk():relative(member.linkname)
                stream=archive.extractfile(member)
                data=stream.read(int(row['bytes'])+1)
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('exact input length/hash mismatch: '+row['source'])
        return data
