"""Read-only historical full-archive proof; writes only reference-lane metadata."""
from pathlib import Path
import subprocess,hashlib,json,io,tarfile,time
start=time.monotonic();OUT=Path(__file__).resolve().parent;W=OUT.parents[3];ROM=W.parent/'2026-10-27-ns-obstacle-rom';C=W.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle';rel='experiments/ns-obstacle-rom/artifacts/neural-pod-guided-scanned-coordinate01';folder=ROM/rel;ref='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROM),*args])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=git('rev-parse','93bf8c14f').decode().strip();mirror=git('rev-parse','b8d45b9a6').decode().strip();head=git('rev-parse','HEAD').decode().strip();tip=git('ls-remote','origin',ref).decode().split()[0]
for old,new in ((source,head),(mirror,tip)):subprocess.run(['git','-c','gc.auto=0','-C',str(ROM),'merge-base','--is-ancestor',old,new],check=True)
ret=json.loads((folder/'RETENTION.json').read_text());inventory=json.loads((folder/'ORIGINAL-FILE-INVENTORY.json').read_text());raw=(folder/'COLLECTION-SHA256SUMS').read_bytes();ledger={}
for line in raw.decode().splitlines():
 h,k=line.split('  ',1);assert k not in ledger;ledger[k.removeprefix('./')]=h
assert len(ledger)==216 and len(inventory)==217 and set(inventory)==set(ledger)|{'COLLECTION-SHA256SUMS'}
assert sha(folder/'RETENTION.json')=='c49a97bc2fbd64f2121040e01b0108b365758e439fe4ecdcb4837cdf710e550a'
assert ret['state']=='COMPLETED' and ret['job_id']==6660421
assert ret['status']=='checksum_verified_pending_independent_review_and_mirror' and ret['archive_bytes']==60351616 and ret['archive_sha256']=='ee9c70e8922755b5d55836e70c9a0a87368690b5d1cd073c0f65a658f2c1c8aa'
for c in (source,mirror):
 for n in ('RETENTION.json','ORIGINAL-FILE-INVENTORY.json','COLLECTION-SHA256SUMS'):assert git('show',c+':'+rel+'/'+n)==(folder/n).read_bytes()
assert sha(folder/'ORIGINAL-FILE-INVENTORY.json')==ret['original_file_inventory_sha256'] and sha(folder/'COLLECTION-SHA256SUMS')==ret['collection_ledger_sha256']
chunks=[];paths=[]
for row in ret['chunks']:
 rp=rel+'/'+row['path'];blobs={c:git('rev-parse',c+':'+rp).decode().strip() for c in (source,mirror)};assert len(set(blobs.values()))==1;chunks.append(dict(row,repository_path=rp,git_blob=blobs[mirror],git_blobs=blobs));p=folder/row['path'];st=p.stat();assert p.is_file() and not p.is_symlink() and st.st_size==row['bytes'] and sha(p)==row['sha256'];paths.append({'path':str(p),'repository_path':rp,'git_blob':blobs[mirror],'sha256':row['sha256'],'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'link_count':st.st_nlink,'inode':st.st_ino,'device':st.st_dev})
assert len(paths)==2 and {x.name for x in (folder/'archive-chunks').iterdir()}=={Path(x['path']).name for x in paths}
class Stream(io.RawIOBase):
 def __init__(self):self.i=0;self.buf=b'';self.pos=0;self.h=hashlib.sha256();self.count=0
 def readable(self):return True
 def read(self,n=-1):
  assert n>=0;out=[]
  while n:
   if self.pos==len(self.buf):
    if self.i==len(chunks):break
    r=chunks[self.i];self.buf=git('cat-file','blob',r['git_blob']);assert len(self.buf)==r['bytes'] and hashlib.sha256(self.buf).hexdigest()==r['sha256'];self.h.update(self.buf);self.count+=len(self.buf);self.i+=1;self.pos=0
   take=min(n,len(self.buf)-self.pos);out.append(self.buf[self.pos:self.pos+take]);self.pos+=take;n-=take
  return b''.join(out)
stream=Stream();seen={};extra={};links=[]
with tarfile.open(fileobj=stream,mode='r|gz') as ar:
 for member in ar:
  if member.isfile():
   key=str(Path(member.name).relative_to('neural-pod-guided-scanned-coordinate01'));assert key not in seen and key not in extra;h=hashlib.file_digest(ar.extractfile(member),'sha256').hexdigest()
   if key in ledger:assert h==ledger[key]==inventory[key]['sha256'] and member.size==inventory[key]['bytes'];seen[key]={'sha256':h,'bytes':member.size}
   else:assert key=='COLLECTION-SHA256SUMS' and h==hashlib.sha256(raw).hexdigest();assert h==inventory[key]['sha256'] and member.size==inventory[key]['bytes'];extra[key]={'sha256':h,'bytes':member.size}
  elif member.issym() or member.islnk():links.append({'member':member.name,'target':member.linkname})
assert set(seen)==set(ledger) and stream.i==2 and stream.count==ret['archive_bytes'] and stream.h.hexdigest()==ret['archive_sha256']
reviews={}
for n in ('SCANNED-COORDINATE-RETENTION-REVIEW.json','SCANNED-COORDINATE-MODEL-REVIEW.json'):
 p=C/'reviews/accuracy-improvement01'/n;reviews[str(p)]={'sha256':sha(p),'record':json.loads(p.read_text())}
cold={'schema':'ns-obstacle-git-cold-archive-v1','source_commit':source,'mirror_commit':mirror,'mirror_ref':ref,'verified_advertised_remote_tip':tip,'source_retention_path':rel+'/RETENTION.json','source_retention_sha256':sha(folder/'RETENTION.json'),'archive_bytes':ret['archive_bytes'],'archive_sha256':ret['archive_sha256'],'chunks':[{k:r[k] for k in ('repository_path','git_blob','bytes','sha256')} for r in chunks],'working_duplicates_verified':True,'unique_bytes_deleted':0,'recovery_needs_only_mirror_history':True,'historical_retention_status_unchanged':ret['status'],'policy':'Proposed archive representation only; preserve historical pending RETENTION and separate independent reviews; no success-status rewrite, pruning or deletion.'}
cp=OUT/'GIT-COLD-ARCHIVE-PROPOSED.json';cp.write_text(json.dumps(cold,indent=2)+'\n')
out={'schema':'ns-obstacle-ROM-guided-working-archive-working-chunk-cold-proposal-v1','proof_passed':True,'deletion_authorized':False,'deletions_performed':False,'source_commit':source,'mirror_commit':mirror,'current_source_head':head,'advertised_mirror_tip':tip,'mirror_ref':ref,'archive_sha256':ret['archive_sha256'],'archive_bytes':ret['archive_bytes'],'original_regular_files_verified':len(seen)+len(extra),'extra_ledger_members':extra,'archive_links':links,'original_ledger_sha256':sha(folder/'COLLECTION-SHA256SUMS'),'retention_sha256':sha(folder/'RETENTION.json'),'historical_status_preserved':ret['status'],'original_job_state_preserved':ret['state'],'chunks':chunks,'exact_paths':paths,'logical_bytes':sum(p['bytes'] for p in paths),'conservative_allocated_reclaim_bytes':sum(p['allocated_bytes'] for p in paths if p['link_count']==1),'proposed_cold_descriptor_sha256':sha(cp),'independent_reviews':reviews,'dependency_clearance':'ROM owner confirms no active local consumer needs working chunks; all future fit inputs independent remote prepared trees. Fresh owner check required before unlink.','preserve':'All ROM metadata/source/arrays/Git objects/history and original pending RETENTION; only2 exact working chunk duplicates may be proposed for later owner/root-authorized unlink.','recovery':'Use unchanged git_archive_recovery.verify with this exact descriptor, stream historical mirrored blobs, verify full archive and original217-file ledger in reserved fresh workspace; no shallow tip-only history.','elapsed_seconds':time.monotonic()-start}
(OUT/'PROPOSAL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('proof_passed','original_regular_files_verified','archive_bytes','conservative_allocated_reclaim_bytes','elapsed_seconds')}))
