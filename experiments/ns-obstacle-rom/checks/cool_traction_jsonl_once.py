from pathlib import Path
import hashlib,json,subprocess,shutil,stat,time,os
B=Path(__file__).resolve().parents[1];repo=B.parents[1]
release=repo.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/TRACTION-JSONL-WORKING-COOLING-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*args])
assert sha(release)=='2265fff9d8aefc050bd573ff98c67b87188bd036aa68776b4bdd58c41cb6da86'
r=json.loads(release.read_text());proof=Path(r['proposal_path']);assert r['authorized'] and sha(proof)==r['proposal_sha256'];p=json.loads(proof.read_text());assert r['exact_paths']==p['exact_paths'] and len(p['exact_paths'])==3
out=B/'checks/traction-jsonl-cooling';assert not out.exists()
tip=git('ls-remote','--exit-code','origin',p['mirror_ref']).decode().split()[0]
files=[];invariants={}
for group in p['groups']:
 git('merge-base','--is-ancestor',group['source_commit'],'HEAD');git('merge-base','--is-ancestor',group['mirror_commit'],tip)
 a=B/'artifacts'/group['group'];assert sha(a/'RETENTION.json')==group['retention_sha256'];invariants[str(a/'RETENTION.json')]=group['retention_sha256']
 assert sha(group['recovery_descriptor'])==group['recovery_descriptor_sha256']
for row in p['exact_paths']:
 f=Path(row['path']);s=f.lstat();assert str(f.relative_to(repo))==row['repository_path'] and stat.S_ISREG(s.st_mode) and not f.is_symlink()
 assert (s.st_size,s.st_dev,s.st_ino,s.st_nlink,stat.S_IMODE(s.st_mode),s.st_blocks*512)==(row['bytes'],row['device'],row['inode'],1,row['mode'],row['allocated_bytes']) and sha(f)==row['sha256']
 for commit,blob in row['direct_source_mirror_blobs'].items():
  assert git('rev-parse',commit+':'+row['repository_path']).decode().strip()==blob
  raw=git('cat-file','blob',blob);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 files.append(f)
paths={str(x) for x in files}
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  for fd in (proc/'fd').iterdir():
   try:assert os.readlink(fd) not in paths
   except (FileNotFoundError,PermissionError):pass
  maps=(proc/'maps').read_text()
  assert not any(x in maps for x in paths)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
before=shutil.disk_usage(B).free;out.mkdir();shutil.copyfile(release,out/'RELEASE.json');shutil.copyfile(proof,out/'PROPOSAL.json')
for group in p['groups']:shutil.copyfile(group['recovery_descriptor'],out/(group['group']+'-RECOVERY.json'))
life=out/'LIFECYCLE.json';record={'status':'verified_before_unlink','release_sha256':sha(release),'proposal_sha256':sha(proof),'advertised_tip':tip,'exact_paths':p['exact_paths'],'local_free_before_bytes':before,'allocated_duplicate_bytes':sum(x.stat().st_blocks*512 for x in files),'no_active_reader':True,'unique_data_removed':False};life.write_text(json.dumps(record,indent=2)+'\n')
for f in files:f.unlink()
assert all(not f.exists() for f in files) and all(sha(k)==v for k,v in invariants.items())
after=shutil.disk_usage(B).free;record.update(status='complete_exact_three_working_duplicates_absent',local_free_after_bytes=after,observed_gain_bytes=after-before,immutable_retention_unchanged=True,completed_epoch=time.time());life.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='exact_paths'},indent=2))
