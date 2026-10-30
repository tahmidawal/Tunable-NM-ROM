from pathlib import Path
import hashlib,json,subprocess,shutil,time
B=Path(__file__).resolve().parents[1];repo=B.parents[1];a=B/'artifacts/neural-pod-guided-diagnostic02';root=repo.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/DIAGNOSTIC02-WORKING-CHUNK-COOLING-RELEASE.json';proof=repo.parent/'2026-10-27-ns-obstacle-2d/experiments/ns-obstacle-2d/checks/guided-diagnostic02-chunk-cold-proposal'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*v):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*v],text=True).strip()
assert sha(root)=='b1b06119d55cc4fba970cb8457efa5d2b818bca86825266fe21f295e0f91d00a';r=json.loads(root.read_text());p=json.loads((proof/'PROPOSAL.json').read_text());c=json.loads((proof/'GIT-COLD-ARCHIVE-PROPOSED.json').read_text())
assert r['authorized'] and sha(proof/'PROPOSAL.json')==r['proposal_sha256'] and sha(proof/'GIT-COLD-ARCHIVE-PROPOSED.json')==r['cold_descriptor_sha256'];assert sha(a/'RETENTION.json')==p['retention_sha256']
tip=git('ls-remote','--exit-code','origin',p['mirror_ref']).split()[0];git('merge-base','--is-ancestor',p['source_commit'],'HEAD');git('merge-base','--is-ancestor',p['mirror_commit'],tip)
paths=[]
for row in r['exact_paths']:
 f=Path(row['path']);s=f.lstat();assert f.is_file() and not f.is_symlink() and str(f.relative_to(repo))==row['repository_path'];assert (s.st_size,s.st_ino,s.st_dev,s.st_nlink)==(row['bytes'],row['inode'],row['device'],1) and sha(f)==row['sha256']
 for commit in (p['source_commit'],p['mirror_commit']):assert git('rev-parse',commit+':'+row['repository_path'])==row['git_blob']
 paths.append(f)
assert len(paths)==4 and set((a/'archive-chunks').iterdir())==set(paths)
life=a/'CHUNK-COOLING-LIFECYCLE.json';assert not life.exists();before=shutil.disk_usage(a).free;allocated=sum(x.stat().st_blocks*512 for x in paths);assert allocated==r['allocated_bytes']
shutil.copyfile(proof/'PROPOSAL.json',a/'INDEPENDENT-COLD-PROPOSAL.json');shutil.copyfile(proof/'GIT-COLD-ARCHIVE-PROPOSED.json',a/'GIT-COLD-ARCHIVE.json');shutil.copyfile(root,a/'CHUNK-COOLING-RELEASE.json')
record={'status':'verified_before_retirement','release_sha256':sha(root),'cold_descriptor_sha256':sha(a/'GIT-COLD-ARCHIVE.json'),'original_retention_sha256':sha(a/'RETENTION.json'),'advertised_tip':tip,'paths':[str(x.relative_to(a)) for x in paths],'local_free_before_bytes':before,'allocated_duplicate_bytes':allocated,'no_active_local_consumer':True,'unique_data_removed':False};life.write_text(json.dumps(record,indent=2)+'\n')
for x in paths:x.unlink()
assert all(not x.exists() for x in paths) and sha(a/'RETENTION.json')==p['retention_sha256'];after=shutil.disk_usage(a).free;record.update(status='complete_exact_duplicates_absent',local_free_after_bytes=after,observed_free_space_change_bytes=after-before,chunk_storage='git_history_only',completed_epoch=time.time());life.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
