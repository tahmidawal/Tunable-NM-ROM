from pathlib import Path
import json,hashlib,subprocess,shutil,time
B=Path(__file__).resolve().parents[1];repo=B.parents[1]
release=repo.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/GUIDED-ARCHIVE-CHUNK-COOLING-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*a],text=True).strip()
assert sha(release)=='91c5cc84a6f1d8d732b24c57ebdbfee116b1b257a3871f656804ec569593786a'
r=json.loads(release.read_text());assert r['authorized'];verified=[];allpaths=[]
for group in r['groups']:
 pp=Path(group['proposal_path']);p=json.loads(pp.read_text());cp=pp.parent/'GIT-COLD-ARCHIVE-PROPOSED.json'
 assert sha(pp)==group['proposal_sha256'] and sha(cp)==group['cold_descriptor_sha256']
 paths=[Path(x['path']) for x in group['exact_paths']];a=paths[0].parent.parent
 assert sha(a/'RETENTION.json')==p['retention_sha256'] and set((a/'archive-chunks').iterdir())==set(paths)
 tip=git('ls-remote','--exit-code','origin',p['mirror_ref']).split()[0];git('merge-base','--is-ancestor',p['source_commit'],'HEAD');git('merge-base','--is-ancestor',p['mirror_commit'],tip)
 for row,f in zip(group['exact_paths'],paths):
  st=f.lstat();assert f.is_file() and not f.is_symlink() and str(f.relative_to(repo))==row['repository_path']
  assert (st.st_size,st.st_ino,st.st_dev,st.st_nlink)==(row['bytes'],row['inode'],row['device'],1) and sha(f)==row['sha256']
  for commit in (p['source_commit'],p['mirror_commit']):assert git('rev-parse',commit+':'+row['repository_path'])==row['git_blob']
 assert sum(f.stat().st_blocks*512 for f in paths)==group['allocated_bytes']
 life=a/'CHUNK-COOLING-LIFECYCLE.json';assert not life.exists()
 verified.append((group,p,pp,cp,a,paths,tip,life));allpaths.extend(paths)
assert len(allpaths)==8 and len(set(allpaths))==8 and set(str(p) for p in allpaths)==set(x['path'] for x in r['exact_paths'])
before=shutil.disk_usage(B).free
for group,p,pp,cp,a,paths,tip,life in verified:
 shutil.copyfile(pp,a/'INDEPENDENT-COLD-PROPOSAL.json');shutil.copyfile(cp,a/'GIT-COLD-ARCHIVE.json');shutil.copyfile(release,a/'CHUNK-COOLING-RELEASE.json')
 life.write_text(json.dumps({'status':'verified_before_retirement','release_sha256':sha(release),'original_retention_sha256':p['retention_sha256'],'cold_descriptor_sha256':sha(a/'GIT-COLD-ARCHIVE.json'),'advertised_tip':tip,'paths':[str(x.relative_to(a)) for x in paths],'allocated_duplicate_bytes':group['allocated_bytes'],'no_active_local_consumer':True,'unique_data_removed':False},indent=2)+'\n')
for path in allpaths:path.unlink()
after=shutil.disk_usage(B).free
for group,p,pp,cp,a,paths,tip,life in verified:
 assert all(not x.exists() for x in paths) and sha(a/'RETENTION.json')==p['retention_sha256']
 d=json.loads(life.read_text());d.update(status='complete_exact_duplicates_absent',chunk_storage='git_history_only',combined_free_before_bytes=before,combined_free_after_bytes=after,combined_observed_gain_bytes=after-before,completed_epoch=time.time());life.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'complete':True,'exact_files':8,'allocated_bytes':r['allocated_bytes'],'free_before':before,'free_after':after,'observed_gain':after-before},indent=2))
