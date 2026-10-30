"""Bounded archive ingestion into Git blobs, with no working archive copy.

Pure retention helper. Callers own release/space/remote/mirror/branch guards.
The virtual commit helper CAS-updates the selected branch. The real index, remote
and working files are not changed by that helper; callers must complete recovery.
"""
from pathlib import Path, PurePosixPath
import hashlib,io,json,os,subprocess,tarfile,tempfile

CHUNK=8*2**20
def git(repo,args,**kw):
    return subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false','-C',str(repo)]+args,check=True,**kw)

def compressed_bound(n):
    n+=len(('blob '+str(n)+'\0').encode())
    return n+(n>>12)+(n>>14)+(n>>25)+13

def tree_allocation_bound(repo,parent,paths,block_size=4096):
    """Only ancestors of changed paths need new trees; no compression credit."""
    additions={}
    for name in paths:
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe tree path')
        for i in range(len(p.parts)):
            directory='/'.join(p.parts[:i]);additions.setdefault(directory,set()).add(p.parts[i])
    total=0
    for directory,children in additions.items():
        spec=parent+(':'+directory if directory else '^{tree}')
        q=subprocess.run(['git','-C',str(repo),'cat-file','-s',spec],capture_output=True,text=True)
        old=int(q.stdout) if q.returncode==0 else 0
        upper=compressed_bound(old+sum(len(x.encode())+64 for x in children))
        total+=((upper+block_size-1)//block_size)*block_size
    if total>4*2**20:raise ValueError('changed ancestor tree allocation cap exceeded')
    return total

def storage_bound(archive_bytes,metadata_files,index_bytes,block_size=4096):
    """No compression credit; extra chunk temp and three bounded index copies."""
    assert type(archive_bytes) is int and archive_bytes>0
    assert metadata_files and all(type(x['bytes']) is int and x['bytes']>=0 for x in metadata_files)
    assert len({x['path'] for x in metadata_files})==len(metadata_files)
    assert all(not PurePosixPath(x['path']).is_absolute() and '..' not in PurePosixPath(x['path']).parts for x in metadata_files)
    assert 0<index_bytes<=8*2**20 and block_size>0
    sizes=[min(CHUNK,archive_bytes-i) for i in range(0,archive_bytes,CHUNK)]
    rounded=lambda n:((n+block_size-1)//block_size)*block_size
    objects=sum(rounded(compressed_bound(n)) for n in sizes)
    directories={str(p) for x in metadata_files for p in PurePosixPath(x['path']).parents if str(p)!='.'}
    # Account each warm file and each loose Git blob separately, plus directory
    # blocks. Existing-object deduplication receives no credit.
    metadata=sum(rounded(x['bytes'])+rounded(compressed_bound(x['bytes'])) for x in metadata_files)+block_size*(len(directories)+len(metadata_files)+16)
    # New entries/metadata paths must fit within1MiB index growth. Trees/commits
    # are separately capped at4MiB by the caller's exact tree inventory.
    helper=3*rounded(index_bytes+2**20)+4*2**20
    return {'archive_objects_bound':objects,'temporary_chunk_bound':rounded(compressed_bound(CHUNK)),
            'metadata_bound':metadata,
            'helper_index_tree_bound':helper,'chunk_bytes':CHUNK,
            'required_free_bytes':10*2**30+256*2**20+objects+rounded(compressed_bound(CHUNK))+metadata+helper}

def ingest(repo,stream,expected_bytes,expected_sha,record):
    """Persist each returned chunk OID before advancing; partial objects survive."""
    rows=[];h=hashlib.sha256();total=0
    while total<expected_bytes:
        need=min(CHUNK,expected_bytes-total);buf=bytearray()
        while len(buf)<need:
            b=stream.read(need-len(buf))
            if not b:raise ValueError('truncated archive; completed chunk objects retained')
            buf.extend(b)
        blob=bytes(buf);h.update(blob)
        oid=git(repo,['hash-object','-w','--stdin'],input=blob,stdout=subprocess.PIPE).stdout.decode().strip()
        raw=git(repo,['cat-file','blob',oid],stdout=subprocess.PIPE).stdout
        digest=hashlib.sha256(blob).hexdigest()
        if len(raw)!=len(blob) or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('written Git blob differs')
        rows.append({'oid':oid,'bytes':len(blob),'sha256':digest});total+=len(blob)
        record({'phase':'archive_ingest','complete_bytes':total,'chunks':rows.copy()})
    if stream.read(1):raise ValueError('archive exceeded exact released size')
    if h.hexdigest()!=expected_sha:raise ValueError('whole archive SHA mismatch')
    record({'phase':'archive_ingest_complete','complete_bytes':total,'archive_sha256':h.hexdigest(),'chunks':rows})
    return rows

class BlobReader(io.RawIOBase):
    """At most one8MiB Git blob buffered; SHA checked on every recovery read."""
    def __init__(self,repo,rows):self.repo=repo;self.rows=iter(rows);self.buf=memoryview(b'');self.pos=0
    def readable(self):return True
    def readinto(self,target):
        if self.pos==len(self.buf):
            try:r=next(self.rows)
            except StopIteration:return 0
            b=git(self.repo,['cat-file','blob',r['oid']],stdout=subprocess.PIPE).stdout
            if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('cold blob differs')
            self.buf=memoryview(b);self.pos=0
        n=min(len(target),len(self.buf)-self.pos);target[:n]=self.buf[self.pos:self.pos+n];self.pos+=n;return n

def verify_complete_stream(repo,chunks,expected_bytes,expected_sha):
    h=hashlib.sha256();n=0
    with io.BufferedReader(BlobReader(repo,chunks)) as stream:
        for b in iter(lambda:stream.read(2**20),b''):h.update(b);n+=len(b)
    if n!=expected_bytes or h.hexdigest()!=expected_sha:raise ValueError('complete compressed archive recovery differs')
    return {'bytes':n,'sha256':h.hexdigest()}

def verify_archive(repo,chunks,measurement,metadata_root,metadata_cap):
    """Verify every original including JSONL; expand only bounded metadata."""
    verify_complete_stream(repo,chunks,measurement['archive_bytes'],measurement['archive_sha256'])
    root=Path(metadata_root);root.mkdir(exist_ok=False);rows={};ledger=None;metadata=0
    expected_metadata={x['path']:x['bytes'] for x in measurement['metadata_files']}
    if len(expected_metadata)!=len(measurement['metadata_files']):raise ValueError('duplicate measured metadata')
    kept=set()
    archive_only=measurement.get('metadata_policy','').startswith('JSONL retained only')
    suffixes={'.json','.py','.sh','.txt','.out','.err','.stdout','.stderr'}
    if not archive_only:suffixes.add('.jsonl')
    stream=io.BufferedReader(BlobReader(repo,chunks),buffer_size=2**20)
    with tarfile.open(fileobj=stream,mode='r|gz') as tf:
        for m in tf:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!=measurement['archive_top_directory']:raise ValueError('unsafe/wrong tar path')
            if m.isdir():continue
            if not m.isfile():raise ValueError('nonregular archive member')
            name=str(PurePosixPath(*p.parts[1:]))
            if not name or name in rows:raise ValueError('duplicate archive member')
            keep=p.suffix in suffixes or p.name in {'COLLECTION-SHA256SUMS','SOURCE-SHA256SUMS'}
            if keep:
                if expected_metadata.get(name)!=m.size or metadata+m.size>metadata_cap:raise ValueError('metadata identity/cap differs')
                kept.add(name)
            elif name in expected_metadata:raise ValueError('metadata keep policy differs')
            h=hashlib.sha256();n=0;parts=[]
            member=tf.extractfile(m)
            for b in iter(lambda:member.read(2**20),b''):
                h.update(b);n+=len(b)
                if keep:parts.append(b)
            if n!=m.size:raise ValueError('member size differs')
            rows[name]={'bytes':n,'sha256':h.hexdigest()}
            if keep:
                f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b''.join(parts));metadata+=n
            if name=='COLLECTION-SHA256SUMS':ledger=b''.join(parts)
    if len(rows)!=measurement['regular_files'] or sum(x['bytes'] for x in rows.values())!=measurement['expanded_bytes']:raise ValueError('complete member count/bytes differ')
    if ledger is None or hashlib.sha256(ledger).hexdigest()!=measurement['ledger_sha256']:raise ValueError('ledger SHA differs')
    pairs=[line.split('  ',1) for line in ledger.decode().splitlines()]
    if len({x[1] for x in pairs})!=len(pairs):raise ValueError('duplicate ledger name')
    expected={n:s for s,n in pairs}
    if set(expected)!=set(rows)-{'COLLECTION-SHA256SUMS'} or any(rows[n]['sha256']!=s for n,s in expected.items()):raise ValueError('original ledger mismatch')
    if kept!=set(expected_metadata) or metadata!=measurement['metadata_extraction_bytes']:raise ValueError('metadata measurement differs')
    return {'files':rows,'metadata_extracted_bytes':metadata,'all_originals_verified':True}

def commit_virtual_chunks(repo,parent,branch,artifact,chunk_rows,metadata_paths,message,index_cap):
    """CAS branch commit through isolated index; caller index untouched.

    The resulting commit retains chunks even if subsequent mirror/cold lifecycle
    fails. Caller MUST finish reviewed index synchronization or report interruption.
    """
    repo=Path(repo);artifact=PurePosixPath(artifact)
    if artifact.is_absolute() or '..' in artifact.parts:raise ValueError('artifact must be repository relative')
    actual=git(repo,['rev-parse',branch],stdout=subprocess.PIPE).stdout.decode().strip()
    if actual!=parent:raise ValueError('branch moved before collection commit')
    with tempfile.TemporaryDirectory(prefix='git-archive-index-',dir=repo) as tmp:
        idx=Path(tmp)/'index';env=dict(os.environ,GIT_INDEX_FILE=str(idx))
        git(repo,['read-tree',parent],env=env)
        git(repo,['add','-f','--']+list(metadata_paths),env=env)
        for i,row in enumerate(chunk_rows):
            path=str(artifact/'archive-chunks'/('collection.tar.gz.part-%03d'%i))
            if (repo/path).exists():raise ValueError('chunk working copy unexpectedly exists')
            git(repo,['update-index','--add','--cacheinfo','100644',row['oid'],path],env=env)
        if idx.stat().st_size>index_cap:raise ValueError('temporary index cap exceeded')
        tree=git(repo,['write-tree'],env=env,stdout=subprocess.PIPE).stdout.decode().strip()
        commit=git(repo,['commit-tree',tree,'-p',parent],input=(message+'\n').encode(),stdout=subprocess.PIPE).stdout.decode().strip()
        git(repo,['update-ref',branch,commit,parent])
        return commit
