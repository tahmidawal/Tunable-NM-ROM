from pathlib import Path
import json,hashlib,subprocess,shutil,time,os
B=Path(__file__).resolve().parents[1];repo=B.parents[1];a=B/'artifacts/training-materialize01'
root=repo.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/training-materialize01/WARM-PAYLOAD-COOLING-RELEASE.json'
proposal=repo.parent/'2026-10-27-ns-obstacle-2d/experiments/ns-obstacle-2d/checks/training-warm-cold-proposal/PROPOSAL.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*args],text=True).strip()
assert sha(root)=='935e09bb97e65ee473143053894059a9e975d053e0b333aaf9320c4171ce0ed6';r=json.loads(root.read_text());assert sha(proposal)==r['proposal_sha256'];p=json.loads(proposal.read_text())
assert p['proof_passed'] and len(p['exact_proposed_paths'])==r['authorized_files']==78 and p['archive_sha256']==r['archive_sha256']
assert sha(a/'GIT-COLD-ARCHIVE.json')==p['cold_manifest_sha256']
for name,digest in p['metadata_sha256'].items():assert sha(a/name)==digest
tip=git('ls-remote','--exit-code','origin',p['advertised_mirror_ref']).split()[0];git('merge-base','--is-ancestor',p['source_commit'],'HEAD');git('merge-base','--is-ancestor',p['mirror_commit'],tip)
for row in p['historical_chunks']:
 for commit,blob in row['git_blobs'].items():assert git('rev-parse',commit+':'+row['repository_path'])==blob
paths=[]
for row in p['exact_proposed_paths']:
 f=Path(row['path']);assert f==a/row['original_relative_path'] and f.resolve()==f and f.is_relative_to(a) and not f.is_symlink() and f.is_file();s=f.stat();assert s.st_nlink==row['link_count']==1 and s.st_size==row['bytes'] and s.st_ino==row['inode'] and s.st_dev==row['device'] and sha(f)==row['sha256'];paths.append(f)
assert len(set(paths))==78
# Check currently open local descriptors against this exact numerical payload set.
active=[];wanted={str(f) for f in paths}
for proc in Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if target in wanted:active.append({'pid':int(proc.name),'fd':fd.name,'path':target})
   except (FileNotFoundError,PermissionError,OSError):pass
 except (FileNotFoundError,PermissionError,OSError):pass
assert not active
life=a/'WARM-PAYLOAD-COOLING-LIFECYCLE.json';mapping=a/'WARM-PAYLOAD-COLD-RECOVERY.json';assert not life.exists() and not mapping.exists()
shutil.copyfile(root,a/'WARM-PAYLOAD-COOLING-RELEASE.json');shutil.copyfile(proposal,a/'INDEPENDENT-WARM-PAYLOAD-PROPOSAL.json')
recovery={'schema':'ns-obstacle-exact-warm-payload-cold-recovery-v1','source_commit':p['source_commit'],'mirror_commit':p['mirror_commit'],'archive_sha256':p['archive_sha256'],'archive_bytes':p['archive_bytes'],'cold_manifest_path':'GIT-COLD-ARCHIVE.json','cold_manifest_sha256':p['cold_manifest_sha256'],'files':p['exact_proposed_paths'],'recovery':p['recovery'],'unique_scientific_data_removed':False};mapping.write_text(json.dumps(recovery,indent=2)+'\n')
free=shutil.disk_usage(a).free;allocated=sum(f.stat().st_blocks*512 for f in paths);assert allocated==r['maximum_allocated_credit_bytes']
record={'schema':'ns-obstacle-exact-warm-payload-cooling-lifecycle-v1','status':'verified_before_retirement','release_sha256':sha(root),'proposal_sha256':sha(proposal),'recovery_map_sha256':sha(mapping),'immutable_metadata_sha256':p['metadata_sha256'],'advertised_tip':tip,'local_consumers':active,'remote_inputs_untouched':True,'local_free_before_bytes':free,'allocated_duplicate_bytes':allocated,'removed_paths':[str(f.relative_to(a)) for f in paths],'unique_data_removed':False};life.write_text(json.dumps(record,indent=2)+'\n')
for f in paths:f.unlink()
assert all(not f.exists() for f in paths)
for name,digest in p['metadata_sha256'].items():assert sha(a/name)==digest
now=shutil.disk_usage(a).free;record.update(status='complete_exact_duplicates_absent',local_free_after_bytes=now,observed_free_space_change_bytes=now-free,completed_epoch=time.time());life.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='removed_paths'},indent=2))
