from pathlib import Path
import hashlib,json,subprocess,shutil,stat,time,os,io,tarfile
B=Path(__file__).resolve().parents[1];repo=B.parents[1]
release=repo.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/FIXED-COORDINATE-JSONL-WORKING-COOLING-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*args])
assert sha(release)=='e109dd599bff6129f1aa0ce470cdccba0032469291ded7610af391641f73d216'
r=json.loads(release.read_text());proof=Path(r['proposal_path']);assert r['authorized'] and sha(proof)==r['proposal_sha256'];p=json.loads(proof.read_text());assert r['exact_paths']==p['exact_paths'] and len(p['exact_paths'])==3
out=B/'checks/coordinate-jsonl-cooling';assert not out.exists()
tip=git('ls-remote','--exit-code','origin',p['mirror_ref']).decode().split()[0]
files=[];invariants={}
for group in p['groups']:
 git('merge-base','--is-ancestor',group['source_commit'],'HEAD');git('merge-base','--is-ancestor',group['mirror_commit'],tip)
 a=B/'artifacts'/group['group'];assert sha(a/'RETENTION.json')==group['retention_sha256'];invariants[str(a/'RETENTION.json')]=group['retention_sha256']
 assert sha(group['recovery_descriptor'])==group['recovery_descriptor_sha256']
 descriptor=json.loads(Path(group['recovery_descriptor']).read_text());parts=[]
 for chunk in descriptor['chunks']:
  for commit in (group['source_commit'],group['mirror_commit']):assert git('rev-parse',commit+':'+chunk['repository_path']).decode().strip()==chunk['git_blob']
  data=git('cat-file','blob',chunk['git_blob']);assert len(data)==chunk['bytes'] and hashlib.sha256(data).hexdigest()==chunk['sha256'];parts.append(data)
 raw=b''.join(parts);assert len(raw)==group['archive_bytes'] and hashlib.sha256(raw).hexdigest()==group['archive_sha256']
 inventory=json.loads((a/'ORIGINAL-FILE-INVENTORY.json').read_text());observed={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r|gz') as tf:
  for member in tf:
   if member.isdir():continue
   assert member.isfile();rel=Path(member.name);assert rel.parts[0]==group['group'] and '..' not in rel.parts and not rel.is_absolute();name=str(Path(*rel.parts[1:]));assert name not in observed
   stream=tf.extractfile(member);h=hashlib.sha256();count=0
   for block in iter(lambda:stream.read(2**20),b''):h.update(block);count+=len(block)
   assert count==member.size;observed[name]={'sha256':h.hexdigest(),'bytes':count}
 assert observed==inventory and len(observed)==group['original_regular_files_verified']
 del raw,parts,data

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
