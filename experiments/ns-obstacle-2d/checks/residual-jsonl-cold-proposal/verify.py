"""Read-only complete-history proof for three completed residual local JSONL duplicates."""
from pathlib import Path
import hashlib,importlib.util,io,json,stat,subprocess,tarfile,time
O=Path(__file__).resolve().parent;W=O.parents[3];R=W.parent/'2026-10-27-ns-obstacle-rom';B=R/'experiments/ns-obstacle-rom';start=time.monotonic()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(R),*a])
ref='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom';tip=git('ls-remote','--exit-code','origin',ref).decode().split()[0];head=git('rev-parse','HEAD').decode().strip()
reader=W/'experiments/ns-obstacle-2d/g2/git_archive_recovery.py';assert sha(reader)=='da6d150859f688ad7fba8767d249d9c7fd8f4f237af8c20e80e93351777365f2'
s=importlib.util.spec_from_file_location('reader',reader);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
groups=[];paths=[]
for suffix,proofname,expected in [('residual01','guided-residual01-chunk-cold-proposal',3)]:
 name='neural-pod-guided-'+suffix;folder=B/'artifacts'/name;rel=str(folder.relative_to(R));cp=O.parent/proofname/'GIT-COLD-ARCHIVE-PROPOSED.json';cold=json.loads(cp.read_text());source=cold['source_commit'];mirror=cold['mirror_commit']
 for old,new in ((source,head),(mirror,tip)):git('merge-base','--is-ancestor',old,new)
 ret=(folder/'RETENTION.json').read_bytes();assert hashlib.sha256(ret).hexdigest()==cold['source_retention_sha256']
 raw=(folder/'COLLECTION-SHA256SUMS').read_bytes();ledger={}
 for line in raw.decode().splitlines():
  h,p=line.split('  ',1);p=p.removeprefix('./');assert p not in ledger;ledger[p]=h
 inv=json.loads((folder/'ORIGINAL-FILE-INVENTORY.json').read_text());assert set(inv)==set(ledger)|{'COLLECTION-SHA256SUMS'}
 for c in (source,mirror):
  for n in ('RETENTION.json','COLLECTION-SHA256SUMS','ORIGINAL-FILE-INVENTORY.json'):assert git('show',c+':'+rel+'/'+n)==(folder/n).read_bytes()
 chunks=cold['chunks']
 for row in chunks:
  for c in (source,mirror):assert git('rev-parse',c+':'+row['repository_path']).decode().strip()==row['git_blob']
 rows=[]
 for p in sorted((folder/'metadata/results/fit').glob('*.jsonl')):
  st=p.lstat();assert stat.S_ISREG(st.st_mode) and not p.is_symlink();k=str(p.relative_to(folder/'metadata'));h=sha(p);assert h==ledger[k]==inv[k]['sha256'] and st.st_size==inv[k]['bytes'];rp=str(p.relative_to(R));ids={c:git('rev-parse',c+':'+rp).decode().strip() for c in (source,mirror)};assert len(set(ids.values()))==1
  blob=next(iter(ids.values()));data=git('cat-file','blob',blob);assert len(data)==st.st_size and hashlib.sha256(data).hexdigest()==h
  rows.append({'path':str(p),'repository_path':rp,'archive_member':name+'/'+k,'sha256':h,'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'device':st.st_dev,'inode':st.st_ino,'link_count':st.st_nlink,'mode':stat.S_IMODE(st.st_mode),'direct_git_blob':blob,'direct_source_mirror_blobs':ids})
 assert len(rows)==expected
 class Stream(io.RawIOBase):
  def __init__(self):self.i=0;self.buf=b'';self.pos=0;self.h=hashlib.sha256();self.n=0
  def readable(self):return True
  def read(self,n=-1):
   assert n>=0;out=[]
   while n:
    if self.pos==len(self.buf):
     if self.i==len(chunks):break
     row=chunks[self.i];self.buf=git('cat-file','blob',row['git_blob']);assert len(self.buf)==row['bytes'] and hashlib.sha256(self.buf).hexdigest()==row['sha256'];self.h.update(self.buf);self.n+=len(self.buf);self.i+=1;self.pos=0
    take=min(n,len(self.buf)-self.pos);out.append(self.buf[self.pos:self.pos+take]);self.pos+=take;n-=take
   return b''.join(out)
 stream=Stream();seen={};links=[]
 with tarfile.open(fileobj=stream,mode='r|gz') as ar:
  for m in ar:
   if m.isfile():
    k=str(Path(m.name).relative_to(name));assert k not in seen and k in inv;h=hashlib.file_digest(ar.extractfile(m),'sha256').hexdigest();assert h==inv[k]['sha256'] and m.size==inv[k]['bytes'];assert h==(hashlib.sha256(raw).hexdigest() if k=='COLLECTION-SHA256SUMS' else ledger[k]);seen[k]={'sha256':h,'bytes':m.size}
   elif m.islnk() or m.issym():links.append({'path':m.name,'target':m.linkname})
 while stream.read(2**20):pass
 assert set(seen)==set(inv) and stream.n==cold['archive_bytes'] and stream.h.hexdigest()==cold['archive_sha256']
 check=module.verify(R,cold,output=None)
 target=O/(name+'-RECOVERY.json');target.write_bytes(cp.read_bytes())
 groups.append({'group':name,'source_commit':source,'mirror_commit':mirror,'archive_sha256':cold['archive_sha256'],'archive_bytes':cold['archive_bytes'],'original_regular_files_verified':len(seen),'archive_links':links,'retention_sha256':sha(folder/'RETENTION.json'),'historical_retention_status_preserved':json.loads(ret)['status'],'original_ledger_sha256':hashlib.sha256(raw).hexdigest(),'inventory_sha256':sha(folder/'ORIGINAL-FILE-INVENTORY.json'),'recovery_descriptor':str(target),'recovery_descriptor_sha256':sha(target),'reader_check':check,'exact_paths':rows});paths+=rows
assert len(paths)==3
result={'schema':'ns-obstacle-completed-jsonl-working-duplicate-cold-proposal-v1','proof_passed':True,'deletion_authorized':False,'deletions_performed':False,'current_source_head':head,'advertised_mirror_tip':tip,'mirror_ref':ref,'reader_sha256':sha(reader),'groups':groups,'exact_paths':paths,'logical_bytes':sum(x['bytes'] for x in paths),'conservative_allocated_reclaim_bytes':sum(x['allocated_bytes'] for x in paths if x['link_count']==1),'dependency_clearance':'Root completed old training plots; ROM owner confirms no active/pending local reader. All model/teacher/launch references use separate preserved immutable remote inputs. Fresh clearance required before exact owner unlink.','preserve':'All current coordinate JSONL and every JSON metadata/model/plot/source, all raw archives and Git histories. Only these three completed-run JSONL working duplicates proposed. Histories are preserved byte-identically in direct source/mirror Git blobs AND full original archives.','recovery':'Restore exact direct historical Git blobs to same relative paths and verify SHA/bytes, or use unchanged reader with retained full archive descriptor and verify original ledger before selective restoration. No current-tip-only recovery or history pruning.','elapsed_seconds':time.monotonic()-start}
(O/'PROPOSAL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('proof_passed','logical_bytes','conservative_allocated_reclaim_bytes','elapsed_seconds')}))
