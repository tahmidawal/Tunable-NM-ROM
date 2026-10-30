"""Read-only proof of one obsolete isolated dependency prefix; no extraction/deletion."""
from pathlib import Path
import hashlib, importlib.util, io, json, os, stat, subprocess, tarfile, time

OUT=Path(__file__).resolve().parent
W=OUT.parents[3]
ROM=W.parent/'2026-10-27-ns-obstacle-rom'
BASE=ROM/'experiments/ns-obstacle-rom'
REL='experiments/ns-obstacle-rom/artifacts/training-materialize01'
A=ROM/REL
PREFIX=A/'deps'
start=time.monotonic()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROM),*a])
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
source=git('rev-parse','859cf77af').decode().strip()
mirror=git('rev-parse','6967aa98f').decode().strip()
cold=json.loads((A/'GIT-COLD-ARCHIVE.json').read_text())
ref=cold['mirror_ref'];tip=git('ls-remote','--exit-code','origin',ref).decode().split()[0]
head=git('rev-parse','HEAD').decode().strip()
for old,new in ((source,head),(mirror,tip),(cold['source_commit'],head),(cold['mirror_commit'],tip)):
    git('merge-base','--is-ancestor',old,new)
ret=json.loads((A/'RETENTION.json').read_text())
assert ret['archive_sha256']==cold['archive_sha256']=='490ade93edb5c65b3fd7f3000987bec7588fc4e693fd049a5f7e9166e994e30f'
assert ret['archive_bytes']==cold['archive_bytes']==1330363646
ledger_raw=(A/'COLLECTION-SHA256SUMS').read_bytes()
ledger={}
for line in ledger_raw.decode().splitlines():
    h,p=line.split('  ',1);p=p.removeprefix('./');assert p not in ledger;ledger[p]=h
assert len(ledger)==1113
for commit in (source,mirror):
    for name in ('RETENTION.json','COLLECTION-SHA256SUMS','WARM-RECOVERY.json'):
        assert git('show',commit+':'+REL+'/'+name)==(A/name).read_bytes(),name
chunks=cold['chunks']
for row in chunks:
    for commit in (source,mirror,cold['source_commit'],cold['mirror_commit']):
        assert git('rev-parse',commit+':'+row['repository_path']).decode().strip()==row['git_blob']

# Reject any local link/special file; proposal addresses regular files only.
local={};directories=[]
assert PREFIX.is_dir() and not PREFIX.is_symlink()
for root,dirs,files in os.walk(PREFIX,followlinks=False):
    for name in dirs:
        p=Path(root)/name;assert not p.is_symlink();directories.append(str(p.relative_to(A)))
    for name in files:
        p=Path(root)/name;s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
        k=str(p.relative_to(A));h=sha(p);assert h==ledger[k]
        local[k]={'path':str(p),'repository_path':str(p.relative_to(ROM)),
            'archive_member':'training-materialize01/'+k,'sha256':h,'bytes':s.st_size,
            'allocated_bytes':s.st_blocks*512,'inode':s.st_ino,'device':s.st_dev,
            'link_count':s.st_nlink,'mode':stat.S_IMODE(s.st_mode)}
assert len(local)==745 and sum(r['bytes'] for r in local.values())==261511620

class Stream(io.RawIOBase):
    def __init__(self):self.i=0;self.buf=b'';self.pos=0;self.h=hashlib.sha256();self.n=0
    def readable(self):return True
    def read(self,n=-1):
        assert n>=0;parts=[]
        while n:
            if self.pos==len(self.buf):
                if self.i==len(chunks):break
                row=chunks[self.i];self.buf=git('cat-file','blob',row['git_blob'])
                assert len(self.buf)==row['bytes'] and hashlib.sha256(self.buf).hexdigest()==row['sha256']
                self.h.update(self.buf);self.n+=len(self.buf);self.i+=1;self.pos=0
            take=min(n,len(self.buf)-self.pos);parts.append(self.buf[self.pos:self.pos+take]);self.pos+=take;n-=take
        return b''.join(parts)
stream=Stream();seen={};links=[];extra={}
with tarfile.open(fileobj=stream,mode='r|gz') as ar:
    for m in ar:
        name=m.name.removeprefix('./');p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        if p==Path('training-materialize01'):continue
        key=str(p.relative_to('training-materialize01'))
        if m.isfile():
            assert key not in seen
            h=hashlib.file_digest(ar.extractfile(m),'sha256').hexdigest()
            if key in ledger:assert h==ledger[key];seen[key]={'sha256':h,'bytes':m.size,'mode':m.mode}
            else:assert key=='COLLECTION-SHA256SUMS' and h==hashlib.sha256(ledger_raw).hexdigest();extra[key]=h
        elif m.islnk():
            target=str(Path(m.linkname.removeprefix('./')).relative_to('training-materialize01'))
            assert target in seen and key not in seen and seen[target]['sha256']==ledger[key]
            seen[key]=dict(seen[target]);seen[key]['mode']=m.mode;links.append({'path':key,'target':target,'type':'hardlink'})
        elif m.issym():
            assert not key.startswith('deps/');links.append({'path':key,'target':m.linkname,'type':'symlink'})
        else:assert m.isdir()
while stream.read(2**20):pass
assert set(seen)==set(ledger)
assert stream.n==cold['archive_bytes'] and stream.h.hexdigest()==cold['archive_sha256']
assert {k for k in seen if k.startswith('deps/')}==set(local)
for k,row in local.items():
    assert row['sha256']==seen[k]['sha256'] and row['bytes']==seen[k]['bytes'] and row['mode']==seen[k]['mode']

# Inspect declared input inventories and source builders without reading numerical arrays.
inventory_checks=[];hits=[]
for p in sorted(BASE.rglob('INPUT-INVENTORY.json')):
    if '/deps/' in str(p):continue
    text=p.read_text();inventory_checks.append({'path':str(p),'sha256':sha(p)})
    if 'training-materialize01/deps' in text:hits.append(str(p))
assert not hits,hits
source_checks=[]
for p in sorted(BASE.glob('build_*launch*.py'))+sorted((BASE/'accuracy_v2').glob('*.py')):
    text=p.read_text();source_checks.append({'path':str(p),'sha256':sha(p)})
    assert 'artifacts/training-materialize01/deps' not in text,str(p)
reader=W/'experiments/ns-obstacle-2d/g2/git_archive_recovery.py'
assert sha(reader)=='da6d150859f688ad7fba8767d249d9c7fd8f4f237af8c20e80e93351777365f2'
spec=importlib.util.spec_from_file_location('reader',reader);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reader_result=module.verify(ROM,cold,output=None)
dump('GIT-COLD-ARCHIVE.json',cold)
dump('READER-CHECK.json',{'reader_sha256':sha(reader),'result':reader_result,'materialized_archive':False})
proposal={'schema':'ns-obstacle-isolated-old-dependency-prefix-cold-proposal-v1',
 'proof_passed':True,'deletion_authorized':False,'deletions_performed':False,
 'scope':str(PREFIX),'source_commit':source,'mirror_commit':mirror,'current_source_head':head,'advertised_mirror_tip':tip,
 'cold_source_commit':cold['source_commit'],'cold_mirror_commit':cold['mirror_commit'],
 'archive_sha256':cold['archive_sha256'],'archive_bytes':cold['archive_bytes'],
 'original_regular_files_verified':len(seen),'ledger_sha256':hashlib.sha256(ledger_raw).hexdigest(),
 'retention_sha256':sha(A/'RETENTION.json'),'historical_retention_status_unchanged':ret['status'],
 'archive_links':links,'exact_paths':list(local.values()),'directory_inventory':directories,
 'logical_bytes':sum(r['bytes'] for r in local.values()),
 'conservative_allocated_reclaim_bytes':sum(r['allocated_bytes'] for r in local.values() if r['link_count']==1),
 'input_inventory_checks':inventory_checks,'source_builder_checks':source_checks,
 'dependency_clearance':'ROM owner confirms no local reader/current or prospective job uses this old isolated prefix. Current jobs use remote immutable prepared/parent data and cluster venv. Native builders use preserved wheels and separately installed job-local deps; no alias to this prefix.',
 'preserve':'All wheels/source/original/results/scalar logs/metadata/plots, system and shared environments, archive blobs and both historical chains. Only exact745 regular working duplicates inside this isolated old prefix proposed; no recursive deletion authorization.',
 'recovery':{'descriptor':'GIT-COLD-ARCHIVE.json','descriptor_sha256':sha(OUT/'GIT-COLD-ARCHIVE.json'),'reader_sha256':sha(reader),'procedure':'Recover full archive in reserved fresh workspace using unchanged reader; verify archive and original COLLECTION-SHA256SUMS; restore exact listed members and recorded modes. Preserve archive hardlink semantics; no symlinks exist in proposed prefix.'},
 'elapsed_seconds':time.monotonic()-start}
dump('PROPOSAL.json',proposal)
print(json.dumps({k:proposal[k] for k in ('proof_passed','original_regular_files_verified','logical_bytes','conservative_allocated_reclaim_bytes','elapsed_seconds')}))
