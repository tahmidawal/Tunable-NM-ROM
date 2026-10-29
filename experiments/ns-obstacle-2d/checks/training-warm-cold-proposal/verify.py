"""Read-only historical archive proof; no extraction, mutation or deletion."""
from pathlib import Path
import subprocess,hashlib,json,io,tarfile,time
started=time.monotonic();W=Path(__file__).resolve().parents[4];R=W.parent/'2026-10-27-ns-obstacle-rom';relative='experiments/ns-obstacle-rom/artifacts/training-materialize01';folder=R/relative
source='52a789a999713cb7d7e38676608dc4e4354c183a';mirror='7d8aaf078a2eff7fcf98d411a590d8a6d0a8e214';branch='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(R),*args])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
head=git('rev-parse','HEAD').decode().strip();advertised=git('ls-remote','origin',branch).decode().split()[0]
for a,b in ((source,head),(mirror,advertised)):subprocess.run(['git','-c','gc.auto=0','-C',str(R),'merge-base','--is-ancestor',a,b],check=True)
ret=json.loads((folder/'RETENTION.json').read_text());cold=json.loads((folder/'GIT-COLD-ARCHIVE.json').read_text());warm=json.loads((folder/'WARM-RECOVERY.json').read_text());inventory={x['path']:x for x in warm['all_original_regular_files']};assert len(inventory)==1113
assert ret['archive_sha256']==cold['archive_sha256']=='490ade93edb5c65b3fd7f3000987bec7588fc4e693fd049a5f7e9166e994e30f' and ret['archive_bytes']==cold['archive_bytes']==1330363646
metadata={}
for name in ('RETENTION.json','WARM-RECOVERY.json','COLLECTION-SHA256SUMS'):
 digest=sha(folder/name)
 for commit in (source,mirror):assert hashlib.sha256(git('show',commit+':'+relative+'/'+name)).hexdigest()==digest
 metadata[name]=digest
inventory['COLLECTION-SHA256SUMS']={'bytes':(folder/'COLLECTION-SHA256SUMS').stat().st_size,'sha256':metadata['COLLECTION-SHA256SUMS']}
chunks=[]
for row in ret['chunks']:
 path=relative+'/'+row['path'];objects={c:git('rev-parse',c+':'+path).decode().strip() for c in (source,mirror)};assert len(set(objects.values()))==1
 chunks.append(dict(row,repository_path=path,git_blobs=objects))
class HistoricalStream(io.RawIOBase):
 def __init__(self):self.index=0;self.buffer=b'';self.offset=0;self.digest=hashlib.sha256();self.bytes=0
 def readable(self):return True
 def read(self,n=-1):
  if n<0:raise ValueError('bounded reads only')
  parts=[];remaining=n
  while remaining:
   if self.offset==len(self.buffer):
    if self.index==len(chunks):break
    row=chunks[self.index];self.buffer=git('cat-file','blob',row['git_blobs'][mirror]);assert len(self.buffer)==row['bytes'] and hashlib.sha256(self.buffer).hexdigest()==row['sha256'];self.digest.update(self.buffer);self.bytes+=len(self.buffer);self.index+=1;self.offset=0
   take=min(remaining,len(self.buffer)-self.offset);parts.append(self.buffer[self.offset:self.offset+take]);self.offset+=take;remaining-=take
  return b''.join(parts)
stream=HistoricalStream();seen=set();links=[]
with tarfile.open(fileobj=stream,mode='r|gz') as archive:
 for member in archive:
  if member.isfile():
   name=str(Path(member.name).relative_to('training-materialize01'));assert name not in seen and name in inventory,(name,"unexpected or duplicate archive member");r=inventory[name];assert member.size==r['bytes'];assert hashlib.file_digest(archive.extractfile(member),'sha256').hexdigest()==r['sha256'];seen.add(name)
  elif member.issym() or member.islnk():links.append({'name':member.name,'target':member.linkname,'type':member.type.decode()})
assert seen==set(inventory) and stream.index==len(chunks) and stream.bytes==ret['archive_bytes'] and stream.digest.hexdigest()==ret['archive_sha256']
# Only named scientific container roles under exact original/result case namespaces.
roles={'full_rule.npz':'materialized full physical rule','mesh.vol.bin':'native immutable mesh','solution.sol':'native solution coefficients','stokes.sol':'native Stokes coefficients','primary_coefficients.npz':'lossless primary coefficient bundle'}
rows=[]
for name,r in sorted(inventory.items()):
 p=folder/name;parts=Path(name).parts
 if len(parts)<3 or parts[0] not in ('original','results') or not p.is_file():continue
 role=roles.get(p.name)
 if len(parts)==5 and parts[2]=='geometry-cache' and parts[3] in ('boundary','volume') and parts[4]=='arrays.npz':role='immutable geometry cache array bundle'
 if role is None:continue
 assert not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];st=p.stat()
 rows.append({'path':str(p),'original_relative_path':name,'role':role,'sha256':r['sha256'],'bytes':r['bytes'],'archive_member':r['archive_source']['member'],'archive_sha256':ret['archive_sha256'],'link_count':st.st_nlink,'allocated_bytes':st.st_blocks*512,'device':st.st_dev,'inode':st.st_ino})
assert len(rows)==78
out={'schema':'ns-obstacle-training-warm-payload-cold-proposal-v1','proof_passed':True,'deletion_authorized':False,'deletions_performed':False,'source_commit':source,'source_head_verified':head,'mirror_commit':mirror,'advertised_mirror_ref':branch,'advertised_mirror_tip':advertised,'archive_sha256':ret['archive_sha256'],'archive_bytes':ret['archive_bytes'],'cold_manifest_sha256':sha(folder/'GIT-COLD-ARCHIVE.json'),'metadata_sha256':metadata,'historical_archive_regular_files_verified':len(seen),'original_regular_files_verified':1113,'checksum_ledger_files_verified':1,'historical_archive_links':links,'historical_chunks':chunks,'exact_proposed_paths':rows,'logical_proposed_bytes':sum(r['bytes'] for r in rows),'conservative_single_link_allocated_reclaim_bytes':sum(r['allocated_bytes'] for r in rows if r['link_count']==1),'active_local_consumers':'ROM owner confirmed none: current recovery/FOM and future launches use independent immutable remote prepared/data trees; root says local reporting reads metadata only','eligible_only_after':'root exact-path authorization and owner fresh byte/hash/consumer checks; preserve any newly required local dependency','preserved':'all JSON/text/code/metadata, full archive and cold descriptors, Git objects/history, other artifacts and any paths not explicitly listed','recovery':'Stream ordered historical chunk blobs from either pinned history; verify every chunk and full archive, then original1113-file WARM-RECOVERY ledger. Recover only exact required member into reserved fresh workspace; no shallow-tip-only history assumption. Archive original source/case provenance remains unchanged.','numerical_operations':'none; byte hashing only; no validation fields, array decoding, native computation or local expansion','elapsed_seconds':time.monotonic()-started}
Path(__file__).with_name('PROPOSAL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('proof_passed','historical_archive_regular_files_verified','logical_proposed_bytes','conservative_single_link_allocated_reclaim_bytes','elapsed_seconds')}))
