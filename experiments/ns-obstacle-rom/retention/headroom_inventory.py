"""Read-only, narrow duplicate proposal. Never removes, extracts, or stages files."""
from pathlib import Path
import hashlib,json,subprocess,tarfile
ROOT=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle-2d')
BASE=ROOT/'experiments/ns-obstacle-2d'
HERE=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT),*args])
def blob(commit,p):return git('show',commit+':'+str(p.relative_to(ROOT)))
proofs={}
def archived_ledger(d,folder):
 class Reader:
  def __init__(self):self.index=0;self.buffer=b'';self.offset=0
  def read(self,n):
   if n<0:raise ValueError('bounded reads only')
   out=[];remaining=n
   while remaining:
    if self.offset==len(self.buffer):
     if self.index==len(d['chunks']):break
     item=d['chunks'][self.index];self.index+=1;self.buffer=git('cat-file','blob',item['git_blob']);self.offset=0
     assert len(self.buffer)==item['bytes'] and hashlib.sha256(self.buffer).hexdigest()==item['sha256']
    take=min(remaining,len(self.buffer)-self.offset);out.append(self.buffer[self.offset:self.offset+take]);self.offset+=take;remaining-=take
   return b''.join(out)
 with tarfile.open(fileobj=Reader(),mode='r|gz') as archive:
  for member in archive:
   if member.name==folder.name+'/COLLECTION-SHA256SUMS':
    assert member.isfile() and member.size<16*1024**2
    return archive.extractfile(member).read()
 raise ValueError('archive ledger not found')
def cold(folder):
 key=str(folder.relative_to(BASE))
 if key in proofs:return proofs[key]
 p=folder/'GIT-COLD-ARCHIVE.json'
 if not p.exists():return None
 d=json.loads(p.read_text());src=d['source_commit'];mirror=d['mirror_commit']
 for commit in (src,mirror):
  retention=git('show',commit+':'+d['source_retention_path'])
  assert hashlib.sha256(retention).hexdigest()==d['source_retention_sha256']
  retained=json.loads(retention)
  assert retained['archive_sha256']==d['archive_sha256'] and retained['archive_bytes']==d['archive_bytes']
  assert len(retained['chunks'])==len(d['chunks'])
  for original,c in zip(retained['chunks'],d['chunks']):
   assert Path(original['path']).name==Path(c['repository_path']).name
   assert (original['sha256'],original['bytes'])==(c['sha256'],c['bytes'])
  entries={}
  for row in git('ls-tree','-r','-l','-z',commit,'--',str(folder.relative_to(ROOT))).split(b'\0'):
   if row:
    meta,name=row.split(b'\t',1);mode,kind,digest,size=meta.split();entries[name.decode()]=(digest.decode(),int(size))
  for c in d['chunks']:assert entries[c['repository_path']]==(c['git_blob'],c['bytes'])
 assert archived_ledger(d,folder)==(folder/'COLLECTION-SHA256SUMS').read_bytes()
 prior=d['git_only_stream_verification'];assert prior['archive_sha256']==d['archive_sha256'] and prior['archive_bytes']==d['archive_bytes']
 proofs[key]={'certificate_sha256':sha(p),'source_commit':src,'mirror_commit':mirror,'archive_sha256':d['archive_sha256'],
  'archive_bytes':d['archive_bytes'],'chunks':d['chunks'],'both_pinned_history_bindings_rechecked':True,
  'original_complete_git_stream_proof':prior,'collection_ledger_sha256':sha(folder/'COLLECTION-SHA256SUMS'),'ledger_read_from_exact_archive_stream':True}
 return proofs[key]
def ledger(folder):
 return {line.split('  ',1)[1].removeprefix('./'):line.split('  ',1)[0] for line in (folder/'COLLECTION-SHA256SUMS').read_text().splitlines()}
def containing_cold(p):
 for folder in p.parents:
  if folder==BASE:break
  if (folder/'GIT-COLD-ARCHIVE.json').exists():return folder
 return None
def main():
 training=json.loads((HERE/'checks/training-materialization-input-precheck.json').read_text())
 protected={str((BASE/f['source']).resolve()) for row in training['rows'] for f in row['runtime_files']}
 assert len(protected)==156
 training_protected=sorted(protected)
 aggregate=BASE/'g2/population01/frozen/AGGREGATE-INDEX.json'
 shared={str((BASE/name).resolve()) for item in json.loads(aggregate.read_text())['shared_controls'].values() for name in item['evidence_sha256']}
 protected.update(shared)
 coarse=['g2-ladder02/results/dfg_h1','g2-ladder02/results/dfg_h2','g2-h3/results/dfg_h3','g2-p4trend/results/dfg_h1','g2-p4trend/results/dfg_h2','g2-p4trend/results/dfg_h3','g2-crossp02/results/immutable_lift']
 prefixes=[BASE/'artifacts'/x for x in coarse]
 groups=[]
 for name,subtrees in [('g2-population-re20-01',['retained','shared']),('g2-population-re20-02',['results/states']),('g2-domain-comparison01',['inputs'])]:
  folder=BASE/'artifacts'/name;expected=ledger(folder);manifest=json.loads((folder/'INPUT-MANIFEST.json').read_text());mapped={r['destination']:r for r in manifest['files']}
  own=cold(folder);rows=[]
  for subtree in subtrees:
   for p in sorted((folder/subtree).rglob('*')):
    if not p.is_file():continue
    rel=str(p.relative_to(folder));before=p.stat();actual=sha(p);after=p.stat()
    row={'path':str(p.relative_to(BASE)),'bytes':after.st_size,'allocated_bytes':after.st_blocks*512,'device':after.st_dev,'inode':after.st_ino,'hardlinks':after.st_nlink,'sha256':actual,'eligible_for_later_duplicate_retirement':False}
    if p.is_symlink() or str(p.resolve()) in protected or any(p.is_relative_to(x) for x in prefixes):row['reason']='protected or symbolic path'
    elif (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or expected.get(rel)!=actual:row['reason']='unstable or collection hash mismatch'
    elif own:row.update(eligible_for_later_duplicate_retirement=True,recovery_certificate=str((folder/'GIT-COLD-ARCHIVE.json').relative_to(BASE)),archive_member=name+'/'+rel)
    elif rel in mapped:
     source=BASE/mapped[rel]['source'];ancestor=containing_cold(source)
     if source.is_file() and not source.is_symlink() and source.stat().st_size==after.st_size and sha(source)==actual and ancestor and cold(ancestor) and ledger(ancestor).get(str(source.relative_to(ancestor)))==actual:
      row.update(eligible_for_later_duplicate_retirement=True,identical_preserved_source=str(source.relative_to(BASE)),recovery_certificate=str((ancestor/'GIT-COLD-ARCHIVE.json').relative_to(BASE)),archive_member=ancestor.name+'/'+str(source.relative_to(ancestor)))
     else:row['reason']='canonical duplicate or independent durable recovery not completely proved'
    else:row['reason']='no mapped canonical duplicate and no existing cold proof'
    rows.append(row)
  groups.append({'artifact':name,'files':rows,'eligible_bytes':sum(r['bytes'] for r in rows if r['eligible_for_later_duplicate_retirement']),
   'ineligible_bytes':sum(r['bytes'] for r in rows if not r['eligible_for_later_duplicate_retirement'])})
 staging=[]
 for name in ['g2-population-re20-01','g2-population-re20-02','g2-domain-comparison01']:
  p=BASE/'staging'/name/'payload';staging.append({'path':str(p.relative_to(BASE)),'exists':p.exists(),'eligible_bytes':0,'reason':'absent; no additional headroom credited' if not p.exists() else 'not proved; preserve'})
 result={'schema':'ns-obstacle-readonly-headroom-proposal-v1','source_commit_observed':git('rev-parse','HEAD').decode().strip(),
  'generator_sha256':sha(Path(__file__)),'deleted_bytes':0,'staging_or_disk_archive_extraction_performed':False,'bounded_archive_metadata_stream_read':True,'protected_training_paths':training_protected,'protected_shared_control_paths':sorted(shared),'aggregate_index_sha256':sha(aggregate),'protected_coarse_directories':[str(p) for p in prefixes],
  'groups':groups,'existing_cold_proofs':proofs,'obsolete_payload_checks':staging,
  'total_eligible_logical_bytes':sum(g['eligible_bytes'] for g in groups),'total_ineligible_logical_bytes':sum(g['ineligible_bytes'] for g in groups),
  'conditions_before_any_action':['Coordinator authorization and current owner/reviewer no-read-dependency confirmation','Recheck each exact file hash and recovery/history reachability immediately before action','Preserve canonical files, metadata, manifests, archives, and all protected inputs','Use per-file paths; never delete a whole parent directory from this proposal','Recheck allocated bytes/hardlinks and actual free space; logical bytes are not guaranteed reclaimed disk']}
 result['eligible_single_link_allocated_bytes']=sum(r['allocated_bytes'] for g in groups for r in g['files'] if r['eligible_for_later_duplicate_retirement'] and r['hardlinks']==1)
 result['eligible_multi_link_logical_bytes']=sum(r['bytes'] for g in groups for r in g['files'] if r['eligible_for_later_duplicate_retirement'] and r['hardlinks']!=1)
 (HERE/'checks/headroom-duplicate-proposal.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'groups':[{k:v for k,v in g.items() if k!='files'} for g in groups],'total_eligible_logical_bytes':result['total_eligible_logical_bytes'],'proofs':len(proofs)}))
if __name__=='__main__':main()
