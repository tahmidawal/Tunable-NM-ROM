from pathlib import Path
import hashlib,json,subprocess,shutil,os,stat,time
B=Path(__file__).resolve().parents[1];repo=B.parents[1];a=B/'artifacts/training-materialize01';scope=a/'deps'
release=repo.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/MATERIALIZATION-DEPS-COOLING-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*args],text=True).strip()
assert sha(release)=='b2b8c7899f9e1be5c2525de62ef35009be95d1a8652e866f184a4f843402664a';r=json.loads(release.read_text());proposal=Path(r['proposal_path']);assert sha(proposal)==r['proposal_sha256'];p=json.loads(proposal.read_text())
assert r['authorized'] and p['proof_passed'] and r['scope']==p['scope']==str(scope) and r['exact_paths']==p['exact_paths'] and len(p['exact_paths'])==745
assert sha(a/'GIT-COLD-ARCHIVE.json')==r['recovery_descriptor_sha256']==p['recovery']['descriptor_sha256'];c=json.loads((a/'GIT-COLD-ARCHIVE.json').read_text());assert c['archive_sha256']==p['archive_sha256'] and c['archive_bytes']==p['archive_bytes']
immutable={name:sha(a/name) for name in ('RETENTION.json','COLLECTION-SHA256SUMS','GIT-COLD-ARCHIVE.json')};assert immutable['RETENTION.json']==p['retention_sha256'] and immutable['COLLECTION-SHA256SUMS']==p['ledger_sha256']
tip=git('ls-remote','--exit-code','origin',c['mirror_ref']).split()[0]
for source,mirror in ((p['source_commit'],p['mirror_commit']),(c['source_commit'],c['mirror_commit'])):
 git('merge-base','--is-ancestor',source,'HEAD');git('merge-base','--is-ancestor',mirror,tip)
 for row in c['chunks']:
  assert git('rev-parse',source+':'+row['repository_path'])==row['git_blob']==git('rev-parse',mirror+':'+row['repository_path'])
paths=[]
for row in p['exact_paths']:
 f=Path(row['path']);s=f.lstat();assert f.is_relative_to(scope) and f.resolve()==f and stat.S_ISREG(s.st_mode) and not f.is_symlink();assert (s.st_size,s.st_ino,s.st_dev,s.st_nlink,stat.S_IMODE(s.st_mode))==(row['bytes'],row['inode'],row['device'],1,row['mode']) and sha(f)==row['sha256'];paths.append(f)
all_entries=list(scope.rglob('*'));assert not any(f.is_symlink() for f in all_entries);assert {f for f in all_entries if f.is_file()}==set(paths);assert all(f.is_file() or f.is_dir() for f in all_entries)
active=[];wanted=set(map(str,paths))
for proc in Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():
   try:
    if os.readlink(fd) in wanted:active.append(str(fd))
   except (OSError,PermissionError):pass
 except (OSError,PermissionError):pass
 try:
  for line in (proc/'maps').read_text().splitlines():
   if len(line.split(maxsplit=5))==6 and line.split(maxsplit=5)[5] in wanted:active.append(str(proc/'maps'))
 except (OSError,PermissionError):pass
assert not active
life=a/'DEPS-COOLING-LIFECYCLE.json';assert not life.exists();before=shutil.disk_usage(a).free;allocated=sum(f.stat().st_blocks*512 for f in paths);assert allocated==p['conservative_allocated_reclaim_bytes']
shutil.copyfile(proposal,a/'INDEPENDENT-DEPS-COLD-PROPOSAL.json');shutil.copyfile(release,a/'DEPS-COOLING-RELEASE.json')
record={'schema':'ns-obstacle-exact-isolated-deps-cooling-lifecycle-v1','status':'verified_before_retirement','release_sha256':sha(release),'proposal_sha256':sha(proposal),'recovery_descriptor_sha256':sha(a/'GIT-COLD-ARCHIVE.json'),'immutable_metadata':immutable,'advertised_tip':tip,'local_free_before_bytes':before,'allocated_duplicate_bytes':allocated,'removed_paths':[str(f.relative_to(a)) for f in paths],'active_readers':active,'unique_data_removed':False,'remote_inputs_untouched':True}
life.write_text(json.dumps(record,indent=2)+'\n')
for f in paths:f.unlink()
for d in sorted((f for f in all_entries if f.is_dir()),key=lambda f:len(f.parts),reverse=True):d.rmdir()
scope.rmdir();assert not scope.exists()
for name,h in immutable.items():assert sha(a/name)==h
now=shutil.disk_usage(a).free;record.update(status='complete_exact_duplicates_absent',local_free_after_bytes=now,observed_free_space_change_bytes=now-before,completed_epoch=time.time());life.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='removed_paths'},indent=2))
