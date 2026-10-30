"""Exact archive-backed duplicate proposal; no deletion/extraction/array decoding."""
from pathlib import Path
import subprocess,hashlib,json,io,tarfile,time
start=time.monotonic();W=Path(__file__).resolve().parents[4];B=W/'experiments/ns-obstacle-2d';ref='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-2d'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*args])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
head=git('rev-parse','HEAD').decode().strip();tip=git('ls-remote','origin',ref).decode().split()[0]
sets={'g2-solverquad01': ['results/quadrature_plus4/jacobian_arrays.npz', 'results/solver_100x/jacobian_arrays.npz'], 'g2-h3': ['results/manufactured_h3/jacobian_arrays.npz', 'results/dfg_h3/jacobian_arrays.npz']};reports=[];proposed=[]
for name,targets in sets.items():
 folder=B/'artifacts'/name;relative=str(folder.relative_to(W));ret=json.loads((folder/'RETENTION.json').read_text());coldpath=folder/'GIT-COLD-ARCHIVE.json';cold=json.loads(coldpath.read_text()) if coldpath.exists() else None;source=cold['source_commit'] if cold else head;mirror=cold['mirror_commit'] if cold else tip
 for old,new in ((source,head),(mirror,tip)):subprocess.run(['git','-c','gc.auto=0','-C',str(W),'merge-base','--is-ancestor',old,new],check=True)
 ledgerpath=folder/'COLLECTION-SHA256SUMS';ledgerraw=ledgerpath.read_bytes();ledger={}
 for line in ledgerraw.decode().splitlines():
  digest,path=line.split('  ',1);path=path.removeprefix('./');assert len(digest)==64 and path not in ledger;ledger[path]=digest
 ledger['COLLECTION-SHA256SUMS']=hashlib.sha256(ledgerraw).hexdigest()
 historic_retention={}
 for commit in (source,mirror):
  if name!='g2-population-re20-02':assert git('show',commit+':'+relative+'/COLLECTION-SHA256SUMS')==ledgerraw
  # Re20 ledger is archive-backed only at these historical pins; full streamed member below must equal local ledger byte hash.
  raw=git('show',commit+':'+relative+'/RETENTION.json');hr=json.loads(raw);assert hr['archive_sha256']==ret['archive_sha256'] and hr['archive_bytes']==ret['archive_bytes'];historic_retention[commit]=hashlib.sha256(raw).hexdigest()
 chunks=[]
 for row in ret['chunks']:
  path=relative+'/'+row['path'];objects={c:git('rev-parse',c+':'+path).decode().strip() for c in (source,mirror)};assert len(set(objects.values()))==1;chunks.append(dict(row,repository_path=path,git_blob=objects[mirror],git_blobs=objects))
 class Stream(io.RawIOBase):
  def __init__(self):self.index=0;self.buffer=b'';self.offset=0;self.digest=hashlib.sha256();self.bytes=0
  def readable(self):return True
  def read(self,n=-1):
   if n<0:raise ValueError('bounded streaming only')
   parts=[];remaining=n
   while remaining:
    if self.offset==len(self.buffer):
     if self.index==len(chunks):break
     row=chunks[self.index];self.buffer=git('cat-file','blob',row['git_blob']);assert len(self.buffer)==row['bytes'] and hashlib.sha256(self.buffer).hexdigest()==row['sha256'];self.digest.update(self.buffer);self.bytes+=len(self.buffer);self.index+=1;self.offset=0
    take=min(remaining,len(self.buffer)-self.offset);parts.append(self.buffer[self.offset:self.offset+take]);self.offset+=take;remaining-=take
   return b''.join(parts)
 stream=Stream();seen={};links=[]
 with tarfile.open(fileobj=stream,mode='r|gz') as archive:
  for member in archive:
   if member.isfile():
    key=str(Path(member.name).relative_to(name));assert key not in seen and key in ledger,(name,key);digest=hashlib.file_digest(archive.extractfile(member),'sha256').hexdigest();assert digest==ledger[key],(name,key);seen[key]={'bytes':member.size,'sha256':digest}
   elif member.issym() or member.islnk():links.append({'member':member.name,'target':member.linkname,'type':member.type.decode()})
 # Tar hardlinks are logical original files: resolve only to already hash-verified archive members.
 pending=[r for r in links if r['type']=='1']
 while pending:
  progress=False
  for link in pending[:]:
   key=str(Path(link['member']).relative_to(name));target=str(Path(link['target']).relative_to(name))
   if target in seen:
    assert key not in seen and key in ledger and ledger[key]==seen[target]['sha256'];seen[key]=dict(seen[target]);pending.remove(link);progress=True
  assert progress,'Unresolved archive hardlink'
 assert set(seen)==set(ledger) and stream.index==len(chunks) and stream.bytes==ret['archive_bytes'] and stream.digest.hexdigest()==ret['archive_sha256']
 rows=[]
 if targets is None:
  for row in chunks:
   p=folder/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];st=p.stat();rows.append({'path':str(p),'kind':'working_archive_chunk_duplicate','sha256':row['sha256'],'bytes':row['bytes'],'git_blob':row['git_blob'],'repository_path':row['repository_path'],'allocated_bytes':st.st_blocks*512,'link_count':st.st_nlink,'device':st.st_dev,'inode':st.st_ino})
 else:
  for key in targets:
   p=folder/key;r=seen[key];assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];st=p.stat();rows.append({'path':str(p),'kind':'archived_numerical_npz_cache','sha256':r['sha256'],'bytes':r['bytes'],'archive_member':name+'/'+key,'allocated_bytes':st.st_blocks*512,'link_count':st.st_nlink,'device':st.st_dev,'inode':st.st_ino})
 report={'group':name,'source_commit':source,'mirror_commit':mirror,'archive_sha256':ret['archive_sha256'],'archive_bytes':ret['archive_bytes'],'retention_sha256':sha(folder/'RETENTION.json'),'historical_retention_sha256':historic_retention,'original_status_preserved':ret['status'],'original_ledger_sha256':sha(ledgerpath),'ledger_retention':('archive_member_verified_from_both_identical_git_blob_histories' if name=='g2-population-re20-02' else 'individual_git_file_and_archive_member'),'original_regular_files_and_ledger_verified':len(seen),'original_archive_links':links,'chunks':chunks,'proposed_paths':rows,'cold_manifest_sha256':sha(coldpath) if cold else None}
 if cold is None:
  # Proposal only: failed scientific status is retained, not relabeled as success.
  manifest={'schema':'ns-obstacle-git-cold-archive-v1','source_commit':source,'mirror_commit':mirror,'mirror_ref':ref,'verified_advertised_remote_tip':tip,'source_retention_path':relative+'/RETENTION.json','source_retention_sha256':historic_retention[source],'archive_bytes':ret['archive_bytes'],'archive_sha256':ret['archive_sha256'],'chunks':[{k:r[k] for k in ('repository_path','git_blob','bytes','sha256')} for r in chunks],'working_duplicates_verified':True,'unique_bytes_deleted':0,'recovery_needs_only_mirror_history':True,'policy':'proposed only; original failed-run status preserved; no prepare-status bypass or deletion performed'}
  mp=Path(__file__).with_name(name+'-RECOVERY-PROPOSED.json');mp.write_text(json.dumps(manifest,indent=2)+'\n');report['proposed_recovery_manifest']={'path':str(mp),'sha256':sha(mp)}
 reports.append(report);proposed.extend(rows)
out={'schema':'ns-obstacle-reference-exact-headroom-proposal-v1','proof_passed':True,'deletion_authorized':False,'deletions_performed':False,'source_head':head,'advertised_mirror_ref':ref,'advertised_mirror_tip':tip,'groups':reports,'exact_path_count':len(proposed),'logical_bytes':sum(r['bytes'] for r in proposed),'conservative_allocated_reclaim_bytes':sum(r['allocated_bytes'] for r in proposed if r['link_count']==1),'preserve':'all metadata/code/mesh/solution/config/reference admissions and original failure records; all Git objects/histories; every unlisted path','active_dependencies':'ROM owner and numerical reviewer confirm no current read dependency; root final exact-path authorization still required. All full_rule/native mesh/solution/config/runtime/lift metadata excluded.','recovery':'Existing historical cold manifests plus exact archived original-Re20 comparison descriptors preserve complete original bytes and statuses. Stream ordered historical chunks, verify each/full SHA, then original collection ledger before recovery into reserved fresh workspace. No shallow-tip-only checkout or history pruning.','large_files_written':False,'numerical_arrays_decoded':False,'validation_accessed':False,'elapsed_seconds':time.monotonic()-start};Path(__file__).with_name('PROPOSAL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('proof_passed','exact_path_count','logical_bytes','conservative_allocated_reclaim_bytes','elapsed_seconds')}))
