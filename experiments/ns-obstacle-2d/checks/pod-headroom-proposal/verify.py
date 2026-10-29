"""Exact archive-backed duplicate proposal; no deletion/extraction/array decoding."""
from pathlib import Path
import subprocess,hashlib,json,io,tarfile,time
start=time.monotonic();W=Path(__file__).resolve().parents[4];B=W/'experiments/ns-obstacle-2d';ref='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-2d'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*args])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
head=git('rev-parse','HEAD').decode().strip();tip=git('ls-remote','origin',ref).decode().split()[0]
sets={'g2-ladder01': None, 'g2-ladder02': None, 'g2-domain-downstream': ['results/case/full_rule.npz', 'results/case/jacobian_arrays.npz'], 'g2-domain-upstream': ['results/case/full_rule.npz', 'results/case/jacobian_arrays.npz'], 'g2-domain-symmetric': ['results/case/full_rule.npz', 'results/case/jacobian_arrays.npz'], 'g2-domain-comparison01': ['results/comparison_original_domain/common_rule.npz', 'results/comparison_original_domain/downstream.npz', 'results/comparison_original_domain/original.npz', 'results/comparison_original_domain/symmetric.npz', 'results/comparison_original_domain/upstream.npz'], 'g2-crossp02': ['results/comparison_p_geometry/common_rule.npz', 'results/comparison_p_geometry/p3_g3_h3.npz', 'results/comparison_p_geometry/p3_g4_h3.npz', 'results/comparison_p_geometry/p4_g4_h3.npz'], 'g2-comparison01': ['results/comparison_dfg/common_rule.npz', 'results/comparison_dfg/dfg_h0.npz', 'results/comparison_dfg/dfg_h1.npz', 'results/comparison_dfg/dfg_h2.npz', 'results/comparison_manufactured/common_rule.npz', 'results/comparison_manufactured/manufactured_h0.npz', 'results/comparison_manufactured/manufactured_h1.npz', 'results/comparison_manufactured/manufactured_h2.npz'], 'g2-p4controls': ['results/manufactured_p4_g4_h3/full_rule.npz', 'results/manufactured_p4_g4_h3/jacobian_arrays.npz'], 'g2-export-batch01': ['results/export_batch_control/arrays/00422aa4d1229f46fe5c80b34c52e325a79bfef6ee60209c79a57ebdea13bcc6.npz', 'results/export_batch_control/arrays/034726803a959c7e5de2a8a2ceb5b6cd4008c8577221ab9871416b9febf1b926.npz', 'results/export_batch_control/arrays/09792edc76a5eeb9bc825e962426a55d9a541b3c410dbe0af30c0213447bdc8c.npz', 'results/export_batch_control/arrays/0d2db76cc4fdac0022d91e852a9e0ffdbc614a03d69a34fe85e5a0421a36c10d.npz', 'results/export_batch_control/arrays/27e81f87fe3f32ea86c7560e3edc28e512122b0fff4166a70c9f4e81115e0224.npz', 'results/export_batch_control/arrays/3f0e7a9a55b0d4a3b65eff61051e2a2712e0bc26f377f0e97a31ea2312f7af86.npz', 'results/export_batch_control/arrays/4aeb88eb4c62c934dc288de64ec6e004bc110db6112708032c5055b6d800e4fe.npz', 'results/export_batch_control/arrays/5123872b1bfb4fcd107eab726d40e7ecb1d402e22c5cbb481bbec47390372d69.npz', 'results/export_batch_control/arrays/57052132d9bd3cc8a6d041b6b10b1c1a0b2f7e92a1471d0ac1b7395b6716aeda.npz', 'results/export_batch_control/arrays/5dd52d05ffb8911cdc0429971500b8fa61d57f61c471f05e0ef3303c43899e03.npz', 'results/export_batch_control/arrays/65383f707028e54f24fed72cb97e8e74835ca90c3ba2fa2fa713be0b2219a0a6.npz', 'results/export_batch_control/arrays/7049eea539c2d9355890b3ec6f688de0a481b66fa8cffe385829b3bd27ed1119.npz', 'results/export_batch_control/arrays/717ac772fcbc195eb90da1ee114c0acb1eacf4768b1487e5059833931dde3452.npz', 'results/export_batch_control/arrays/95e6371d779043d912679a973f90dbbaf6a0fefdf4468ee0bc0b30bb086fbd0b.npz', 'results/export_batch_control/arrays/a61e3933e7458220e0fb89252965a03586fa38a93679edb83016fa892e1de7ad.npz', 'results/export_batch_control/arrays/a73062fd59923f018f3d95aca43d2af6c0b00da5b72f570137d34eed98841dca.npz', 'results/export_batch_control/arrays/c9d51c5918d28e8213cc21953b1c837a0547195a47995b432119884a1a072f4b.npz', 'results/export_batch_control/arrays/d6d56c033882a78aeb2ff2d64e545e85ee699d4ade1ed8fcef6b0efea36cbb78.npz', 'results/export_batch_control/arrays/df84e63fd88d924fa9ce6c2381c4e57816a5aff0a2025293e8b715120aa6e7de.npz']};reports=[];proposed=[]
for name,targets in sets.items():
 folder=B/'artifacts'/name;relative=str(folder.relative_to(W));ret=json.loads((folder/'RETENTION.json').read_text());coldpath=folder/'GIT-COLD-ARCHIVE.json';cold=json.loads(coldpath.read_text()) if coldpath.exists() else None;source=cold['source_commit'] if cold else head;mirror=cold['mirror_commit'] if cold else tip
 for old,new in ((source,head),(mirror,tip)):subprocess.run(['git','-c','gc.auto=0','-C',str(W),'merge-base','--is-ancestor',old,new],check=True)
 ledgerpath=folder/'COLLECTION-SHA256SUMS';ledgerraw=ledgerpath.read_bytes();ledger={}
 for line in ledgerraw.decode().splitlines():
  digest,path=line.split('  ',1);path=path.removeprefix('./');assert len(digest)==64 and path not in ledger;ledger[path]=digest
 ledger['COLLECTION-SHA256SUMS']=hashlib.sha256(ledgerraw).hexdigest()
 historic_retention={}
 for commit in (source,mirror):
  assert git('show',commit+':'+relative+'/COLLECTION-SHA256SUMS')==ledgerraw
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
 assert set(seen)==set(ledger) and stream.index==len(chunks) and stream.bytes==ret['archive_bytes'] and stream.digest.hexdigest()==ret['archive_sha256']
 rows=[]
 if targets is None:
  for row in chunks:
   p=folder/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];st=p.stat();rows.append({'path':str(p),'kind':'working_archive_chunk_duplicate','sha256':row['sha256'],'bytes':row['bytes'],'git_blob':row['git_blob'],'repository_path':row['repository_path'],'allocated_bytes':st.st_blocks*512,'link_count':st.st_nlink,'device':st.st_dev,'inode':st.st_ino})
 else:
  for key in targets:
   p=folder/key;r=seen[key];assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];st=p.stat();rows.append({'path':str(p),'kind':'archived_numerical_npz_cache','sha256':r['sha256'],'bytes':r['bytes'],'archive_member':name+'/'+key,'allocated_bytes':st.st_blocks*512,'link_count':st.st_nlink,'device':st.st_dev,'inode':st.st_ino})
 report={'group':name,'source_commit':source,'mirror_commit':mirror,'archive_sha256':ret['archive_sha256'],'archive_bytes':ret['archive_bytes'],'retention_sha256':sha(folder/'RETENTION.json'),'historical_retention_sha256':historic_retention,'original_status_preserved':ret['status'],'original_ledger_sha256':sha(ledgerpath),'original_regular_files_and_ledger_verified':len(seen),'original_archive_links':links,'chunks':chunks,'proposed_paths':rows,'cold_manifest_sha256':sha(coldpath) if cold else None}
 if cold is None:
  # Proposal only: failed scientific status is retained, not relabeled as success.
  manifest={'schema':'ns-obstacle-git-cold-archive-v1','source_commit':source,'mirror_commit':mirror,'mirror_ref':ref,'verified_advertised_remote_tip':tip,'source_retention_path':relative+'/RETENTION.json','source_retention_sha256':historic_retention[source],'archive_bytes':ret['archive_bytes'],'archive_sha256':ret['archive_sha256'],'chunks':[{k:r[k] for k in ('repository_path','git_blob','bytes','sha256')} for r in chunks],'working_duplicates_verified':True,'unique_bytes_deleted':0,'recovery_needs_only_mirror_history':True,'policy':'proposed only; original failed-run status preserved; no prepare-status bypass or deletion performed'}
  mp=Path(__file__).with_name(name+'-RECOVERY-PROPOSED.json');mp.write_text(json.dumps(manifest,indent=2)+'\n');report['proposed_recovery_manifest']={'path':str(mp),'sha256':sha(mp)}
 reports.append(report);proposed.extend(rows)
out={'schema':'ns-obstacle-reference-exact-headroom-proposal-v1','proof_passed':True,'deletion_authorized':False,'deletions_performed':False,'source_head':head,'advertised_mirror_ref':ref,'advertised_mirror_tip':tip,'groups':reports,'exact_path_count':len(proposed),'logical_bytes':sum(r['bytes'] for r in proposed),'conservative_allocated_reclaim_bytes':sum(r['allocated_bytes'] for r in proposed if r['link_count']==1),'preserve':'all metadata/code/mesh/solution/config/reference admissions and original failure records; all Git objects/histories; every unlisted path','active_dependencies':'Pending owner/root exact dependency confirmation before any deletion; proposed paths exclude all native mesh/solution/config/runtime/lift metadata.','recovery':'Existing historical cold manifests plus explicit ladder archive descriptors preserve complete original bytes and statuses. Stream ordered historical chunks, verify each/full SHA, then original collection ledger before recovery into reserved fresh workspace. No shallow-tip-only checkout or history pruning.','large_files_written':False,'numerical_arrays_decoded':False,'validation_accessed':False,'elapsed_seconds':time.monotonic()-start};Path(__file__).with_name('PROPOSAL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('proof_passed','exact_path_count','logical_bytes','conservative_allocated_reclaim_bytes','elapsed_seconds')}))
