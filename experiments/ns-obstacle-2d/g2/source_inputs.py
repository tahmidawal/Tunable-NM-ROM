"""Read exact committed inputs, including explicitly declared recoverable archive members.

Archive recovery uses historical Git blobs and a reviewed full-archive certificate;
it never silently falls back from a missing working file to unrelated local bytes.
"""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import hashlib, json, subprocess, tarfile, tempfile,shutil
import git_archive_recovery

def relative(value):
    path=PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('input path must be confined and relative')
    return str(path)

class SourceInputs:
    def __init__(self, repo, prefix, commit, reserve_bytes=10*1024**3, verify_warm_archives=False):
        self.repo=Path(repo);self.prefix=relative(prefix);self.commit=commit
        self.reserve_bytes=reserve_bytes;self.temporary=None;self.archives={};self.handles={}
        self.verify_warm_archives=verify_warm_archives;self.verified_warm_inputs=[]
    def __enter__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='source-recovery-',dir=self.repo)
        return self
    def __exit__(self,*args):
        for archive in self.handles.values():archive.close()
        self.temporary.cleanup()
    def committed(self,path):
        return subprocess.check_output(['git','-C',str(self.repo),'show',self.commit+':'+self.prefix+'/'+relative(path)])
    def warm_archive(self,row):
        if not self.verify_warm_archives or 'archive_source' not in row:return None
        declaration=row['archive_source'];raw=self.committed(declaration['manifest_source'])
        if hashlib.sha256(raw).hexdigest()!=declaration['manifest_sha256']:raise ValueError('archive recovery manifest changed')
        relative(declaration['member'])
        base=(self.repo/self.prefix).resolve();path=base/relative(row['source'])
        if path.is_symlink() or not path.resolve().is_relative_to(base) or not path.is_file() or path.stat().st_size!=row['bytes']:return None
        with path.open('rb') as stream:data=stream.read(row['bytes']+1)
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:return None
        self.verified_warm_inputs.append({'source':row['source'],'sha256':row['sha256'],'bytes':row['bytes'],'archive_manifest_sha256':declaration['manifest_sha256'],'source_kind':'exact_verified_working_duplicate'})
        return data
    def archive(self,declaration):
        raw=self.committed(declaration['manifest_source']);digest=hashlib.sha256(raw).hexdigest()
        if digest!=declaration['manifest_sha256']:raise ValueError('archive recovery manifest changed')
        if digest not in self.archives:
            root=Path(self.temporary.name);manifest=root/(digest+'.json');manifest.write_bytes(raw);output=root/(digest+'.tar.gz')
            git_archive_recovery.recover(SimpleNamespace(repo=self.repo,manifest=manifest,output=output,reserve_bytes=self.reserve_bytes))
            self.archives[digest]=output
            self.handles[digest]=tarfile.open(output,'r:gz');self.handles[digest].getmembers()
        return digest,self.handles[digest]
    @staticmethod
    def member(archive,name):
        member=archive.getmember(relative(name));seen=set()
        while member.islnk():
            if member.name in seen:raise ValueError('archive hardlink cycle')
            seen.add(member.name);member=archive.getmember(relative(member.linkname))
        if not member.isfile():raise ValueError('input is not a regular archived member')
        return member
    @staticmethod
    def verify(row,data):
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('exact input length/hash mismatch: '+row['source'])
        return data
    def materialize(self,rows,destination):
        """Index each archive once, then read requested members in forward data order."""
        destination=Path(destination).resolve();groups={}
        def write(row,data):
            self.verify(row,data);path=destination/relative(row['destination'])
            if not path.resolve().is_relative_to(destination) or path.exists():raise ValueError('unsafe or duplicate destination')
            if shutil.disk_usage(destination).free<len(data)+self.reserve_bytes:raise MemoryError('input materialization would violate reserve')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        for row in rows:
            if 'archive_source' not in row:write(row,self.committed(row['source']));continue
            warm=self.warm_archive(row)
            if warm is not None:write(row,warm);continue
            digest,archive=self.archive(row['archive_source']);member=self.member(archive,row['archive_source']['member'])
            if member.size!=row['bytes']:raise ValueError('archived member length differs')
            groups.setdefault(digest,{}).setdefault(member.name,(member,[]))[1].append(row)
        for digest,members in groups.items():
            archive=self.handles[digest]
            for member,requests in sorted(members.values(),key=lambda item:item[0].offset_data):
                data=archive.extractfile(member).read(member.size+1)
                for row in requests:write(row,data)
    def read(self,row):
        if 'archive_source' not in row:
            data=self.committed(row['source'])
        else:
            warm=self.warm_archive(row)
            if warm is not None:return self.verify(row,warm)
            _,archive=self.archive(row['archive_source']);member=self.member(archive,row['archive_source']['member'])
            data=archive.extractfile(member).read(int(row['bytes'])+1)
        return self.verify(row,data)
