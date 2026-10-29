"""Read-only proof for removal of exact working chunk duplicates; never deletes."""
from pathlib import Path
import subprocess,hashlib,json,io,tarfile,time
started=time.monotonic();W=Path(__file__).resolve().parents[4];R=W.parent/'2026-10-27-ns-obstacle-rom';relative='experiments/ns-obstacle-rom/artifacts/development-materialize02';folder=R/relative
source='83026ff45';mirror='bbed36fde';branch='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(R),*args])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=git('rev-parse',source).decode().strip();mirror=git('rev-parse',mirror).decode().strip();head=git('rev-parse','HEAD').decode().strip();advertised=git('ls-remote','origin',branch).decode().split()[0]
for a,b in ((source,head),(mirror,advertised)):subprocess.run(['git','-C',str(R),'merge-base','--is-ancestor',a,b],check=True)
r=json.loads((folder/'RETENTION.json').read_text());inventory=json.loads((folder/'ORIGINAL-FILE-INVENTORY.json').read_text());assert r['archive_sha256']=='93ab53b83b5abd699f6e0f6c1e0acbc312759abac135cd4ee3b7f5d3fb95982d' and r['archive_bytes']==504733195
for name in ('RETENTION.json','ORIGINAL-FILE-INVENTORY.json','COLLECTION-SHA256SUMS'):
 for commit in (source,mirror):assert hashlib.sha256(git('show',commit+':'+relative+'/'+name)).hexdigest()==sha(folder/name)
assert sha(folder/'ORIGINAL-FILE-INVENTORY.json')==r['original_file_inventory_sha256']
assert {p.name for p in (folder/'archive-chunks').iterdir()}=={Path(x['path']).name for x in r['chunks']}
rows=[];totals={c:hashlib.sha256() for c in (source,mirror)}
for row in r['chunks']:
 p=folder/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 rec=dict(row,absolute_path=str(p),link_count=p.stat().st_nlink,allocated_bytes=p.stat().st_blocks*512,git_blobs={})
 for c in (source,mirror):
  obj=git('rev-parse',c+':'+relative+'/'+row['path']).decode().strip();blob=git('cat-file','blob',obj)
  assert len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['sha256'];totals[c].update(blob);rec['git_blobs'][c]=obj
 rows.append(rec)
assert all(h.hexdigest()==r['archive_sha256'] for h in totals.values())
class HistoricalStream(io.RawIOBase):
 def __init__(self):self.index=0;self.buffer=b'';self.offset=0
 def readable(self):return True
 def read(self,n=-1):
  if n<0:raise ValueError('bounded streaming reads only')
  parts=[];remaining=n
  while remaining:
   if self.offset==len(self.buffer):
    if self.index==len(rows):break
    self.buffer=git('cat-file','blob',rows[self.index]['git_blobs'][mirror]);self.offset=0;self.index+=1
   take=min(remaining,len(self.buffer)-self.offset);parts.append(self.buffer[self.offset:self.offset+take]);self.offset+=take;remaining-=take
  return b''.join(parts)
seen=set();links=[]
with tarfile.open(fileobj=HistoricalStream(),mode='r|gz') as archive:
 for member in archive:
  if member.isfile():
   name=str(Path(member.name).relative_to('development-materialize02'));assert name not in seen and name in inventory
   stream=archive.extractfile(member);h=hashlib.file_digest(stream,'sha256').hexdigest();assert member.size==inventory[name]['bytes'] and h==inventory[name]['sha256'];seen.add(name)
  elif member.issym() or member.islnk():links.append({'name':member.name,'target':member.linkname,'type':member.type.decode()})
assert seen==set(inventory) and len(seen)==894
out={'schema':'ns-obstacle-working-chunk-cold-retirement-proposal-v1','proof_passed':True,'deletion_authorized':False,'deletions_performed':False,'source_commit':source,'source_head_verified':head,'mirror_commit':mirror,'advertised_mirror_ref':branch,'advertised_mirror_tip':advertised,'archive_sha256':r['archive_sha256'],'archive_bytes':r['archive_bytes'],'retention_sha256':sha(folder/'RETENTION.json'),'inventory_sha256':sha(folder/'ORIGINAL-FILE-INVENTORY.json'),'recovered_regular_files_verified_without_working_chunks':len(seen),'archive_links':links,'exact_proposed_paths':rows,'conservative_reclaimable_allocated_bytes':sum(x['allocated_bytes'] for x in rows if x['link_count']==1),'recovery':'Read ordered chunk blobs from pinned mirror (or source), verify each SHA/bytes and full archive SHA, then verify original-file inventory before materialization. Preserve all Git objects/history and metadata; no shallow-tip-only recovery.','scope':'Only these11 successful development-materialize02 working chunk paths; no metadata, raw results, failed attempt, other archive, or Git object deletion. Owner must fresh-check after root authorization.','elapsed_seconds':time.monotonic()-started}
Path(__file__).with_name('PROPOSAL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('proof_passed','archive_bytes','conservative_reclaimable_allocated_bytes','recovered_regular_files_verified_without_working_chunks','elapsed_seconds')}))
