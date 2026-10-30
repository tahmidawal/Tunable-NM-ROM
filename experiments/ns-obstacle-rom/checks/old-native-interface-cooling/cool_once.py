from pathlib import Path
import json,hashlib,os,shutil,time,stat
out=Path(__file__).resolve().parent;B=out.parents[1];W=B.parents[1]
release=W.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/OLD-NATIVE-INTERFACE-COOLING-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(release)=='88c7042d1e37d50b98948eb4a7efcbabc8e1632f0679a94ae561c9a25f7a9f0c'
r=json.loads(release.read_text());original=Path(r['proposal_path']);assert sha(original)==r['proposal_sha256'];p=json.loads((out/'PROPOSAL.json').read_text());old=json.loads(original.read_text())
assert r['authorized'] and p['proof_passed'] and r['exact_paths']==p['groups'][0]['proposed_paths']+p['groups'][1]['proposed_paths']+p['groups'][2]['proposed_paths']==old['groups'][0]['proposed_paths']+old['groups'][1]['proposed_paths']+old['groups'][2]['proposed_paths']
assert len(r['exact_paths'])==10
immutable={}
for group in p['groups']:
 folder=B/'artifacts'/group['group']
 for name,key in [('RETENTION.json','retention_sha256'),('COLLECTION-SHA256SUMS','original_ledger_sha256'),('GIT-COLD-PROOF.json','cold_manifest_sha256')]:
  path=folder/name;assert sha(path)==group[key];immutable[str(path)]=sha(path)
 assert sha(out/(group['group']+'-RECOVERY.json'))==group['recovery_descriptor_sha256']
paths=[Path(x['path']) for x in r['exact_paths']];wanted=set(map(str,paths));active=[]
for proc in Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():
   try:
    if os.readlink(fd) in wanted:active.append(str(fd))
   except OSError:pass
 except OSError:pass
 try:
  for line in (proc/'maps').read_text().splitlines():
   if len(line.split(maxsplit=5))==6 and line.split(maxsplit=5)[5] in wanted:active.append(str(proc/'maps'))
 except OSError:pass
assert not active
for row,path in zip(r['exact_paths'],paths):
 st=path.lstat();assert path.resolve()==path and path.is_relative_to(B/'artifacts') and stat.S_ISREG(st.st_mode) and (st.st_size,st.st_ino,st.st_dev,st.st_nlink,st.st_blocks*512)==(row['bytes'],row['inode'],row['device'],1,row['allocated_bytes']) and sha(path)==row['sha256']
life=out/'LIFECYCLE.json';assert not life.exists();before=shutil.disk_usage(B).free;shutil.copyfile(release,out/'RELEASE.json')
record={'status':'verified_before_retirement','release_sha256':sha(release),'fresh_full_archive_proof_sha256':sha(out/'PROPOSAL.json'),'immutable_metadata':immutable,'exact_paths':r['exact_paths'],'active_readers':active,'local_free_before_bytes':before,'unique_data_removed':False,'remote_inputs_untouched':True}
life.write_text(json.dumps(record,indent=2)+'\n')
for path in paths:path.unlink()
assert not any(path.exists() for path in paths)
assert all(sha(Path(path))==h for path,h in immutable.items())
after=shutil.disk_usage(B).free;record.update(status='complete_exact_duplicates_absent',local_free_after_bytes=after,observed_free_space_change_bytes=after-before,completed_epoch=time.time());life.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in {'exact_paths','immutable_metadata'}},indent=2))
