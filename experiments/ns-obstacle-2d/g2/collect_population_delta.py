"""Terminal-only remote delta packing and one guarded local reconstruction.

Uses the frozen reviewed writer/reader without modifying them. No job submission,
PDE solve, retry, admission, Git-add, cooling, or remote deletion is performed.
"""
from pathlib import Path
import argparse,hashlib,json,shlex,shutil,subprocess,tarfile
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];PY='/home/tahmid/Dev/.venv/bin/python';CPY='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python';PREFIX='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login']
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def remote(code):return subprocess.run(SSH+[shlex.quote(CPY)+" - <<'PYREMOTE'\n"+code+"\nPYREMOTE"],capture_output=True,text=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--phase',choices=('pack','collect'),required=True);p.add_argument('--other-reservations-bytes',type=int,default=0);a=p.parse_args();
 if a.other_reservations_bytes<0:raise ValueError('reservations cannot be negative')
 schedule=read(B/'g2/population-rest01/SCHEDULE.json');row=next(x for x in schedule['groups']if x['group']==a.group);attempt=row['attempt'];stage=B/'staging'/attempt;submitted=read(stage/'STAGING.json');assert submitted['submitted']and submitted['candidate_sha256']==row['candidate_sha256'];job=submitted['job_id'];root=PREFIX+attempt;archive=root+'.delta.tar.gz';receipt=root+'.DELTA-WRITE.json';state=stage/'DELTA-COLLECTION';state.mkdir(exist_ok=True)
 if a.phase=='pack':
  code=f'''from pathlib import Path
import subprocess,hashlib,json
root=Path({root!r});archive=Path({archive!r});receipt=Path({receipt!r})
accounting=subprocess.check_output(['sacct','-j',{str(job)!r},'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw,ExitCode'],text=True)
rows=[x.split('|')for x in accounting.splitlines()if x.split('|')[0]=={str(job)!r}]
if len(rows)!=1 or rows[0][1]not in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','CANCELLED'):raise ValueError('authoritative terminal job required')
if archive.exists()or receipt.exists()or archive.with_suffix(archive.suffix+'.partial').exists():raise FileExistsError('prior packing state exists; inspect, do not duplicate')
if(root/'SACCT.txt').exists()or(root/'COLLECTION-SHA256SUMS').exists():raise FileExistsError('prior collection metadata exists; inspect before resuming')
(root/'SACCT.txt').write_text(accounting)
lines=[]
for p in sorted(root.rglob('*')):
 if p.is_file()and not p.is_symlink():
  with p.open('rb')as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  lines.append(digest+'  ./'+str(p.relative_to(root))+'\\n')
(root/'COLLECTION-SHA256SUMS').write_text(''.join(lines))
subprocess.run([{CPY!r},str(root/'g2/delta_retention.py'),'write','--root',str(root),'--index',str(root/'g2/population-rest01/SHARED-REUSE-INDEX.json'),'--output',str(archive),'--receipt',str(receipt)],check=True)
print(receipt.read_text())
'''
  result=remote(code);(state/'PACK.stdout').write_text(result.stdout);(state/'PACK.stderr').write_text(result.stderr);result.check_returncode();print(result.stdout,flush=True);return
 result=remote(f'from pathlib import Path\nprint(Path({receipt!r}).read_text())');(state/'REMOTE-RECEIPT.stdout').write_text(result.stdout);(state/'REMOTE-RECEIPT.stderr').write_text(result.stderr);result.check_returncode();written=json.loads(result.stdout);dump(state/'REMOTE-WRITE.json',written);bound=read(B/'g2/population-rest01/RESOURCE-BOUND.json');logical=written['logical_regular_bytes'];size=written['archive_bytes'];largest=bound['largest_cold_base_archive_bytes'];warm=bound['per_group_new_warm_Git_bound_bytes'];peak=max(logical+size+largest,logical+2*size+warm)+256*1024**2+a.other_reservations_bytes
 if logical>bound['per_group_expanded_regular_bound_bytes']or size>bound['per_group_delta_archive_bound_bytes']:raise MemoryError('measured output invalidates frozen envelope; retain remote originals')
 free=shutil.disk_usage(B).free
 if free<peak+10*1024**3:raise MemoryError('actual serial collection peak plus all reservations does not fit')
 dump(state/'DISK-GUARD.json',{'free_bytes':free,'actual_peak_bytes':peak,'untouched_reserve_bytes':10*1024**3,'other_reservations_bytes':a.other_reservations_bytes,'remote_original_preserved':True})
 local=B/'runs'/(attempt+'.delta.tar.gz');target=B/'artifacts'/attempt
 if local.exists()or target.exists():raise FileExistsError('prior transfer/reconstruction exists; inspect before resuming')
 subprocess.run(['scp','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login:'+archive,str(local)],check=True)
 if sha(local)!=written['archive_sha256']:raise ValueError('transport checksum mismatch')
 recovery=state/'RECOVERY.json';subprocess.run([PY,'-B',str(B/'g2/delta_retention.py'),'recover','--archive',str(local),'--sha256',written['archive_sha256'],'--index',str(B/'g2/population-rest01/SHARED-REUSE-INDEX.json'),'--target',str(target),'--repo',str(REPO),'--receipt',str(recovery)],check=True)
 with tarfile.open(local,'r|gz')as tar:
  for member in tar:
   if member.name=='RETENTION-MAP.json':raw=tar.extractfile(member).read();break
  else:raise ValueError('retention map absent')
 if hashlib.sha256(raw).hexdigest()!=written['retention_map_sha256']:raise ValueError('retention map mismatch')
 (target/'DELTA-RETENTION-MAP.json').write_bytes(raw)
 subprocess.run([PY,'-B',str(B/'g2/chunk_verified_delta.py'),'--archive',str(local),'--recovery',str(recovery),'--out',str(target/'archive-chunks')],check=True,capture_output=True)
 chunks=read(target/'archive-chunks/CHUNKS.json');retention={'status':'complete_all_regular_files_verified','archive_kind':'lossless_delta_with_exact_base_index','job_id':job,'archive_sha256':written['archive_sha256'],'archive_bytes':size,'verified_regular_files':read(recovery)['regular_files'],'chunks':[{**c,'path':'archive-chunks/'+c['path']}for c in chunks['chunks']],'retention_map_sha256':written['retention_map_sha256'],'reuse_index_sha256':sha(B/'g2/population-rest01/SHARED-REUSE-INDEX.json'),'recovery_reader_sha256':sha(B/'g2/delta_retention.py'),'recovery_requires_delta_reader_and_bound_base_archives':True,'source_commit':submitted['source_commit'],'reference_data_admitted':False,'remote_cleanup':'recorded separately after durable mirror; original retained','audit_scope':'Complete original regular-file inventory; all new bytes and every exact base/alias locator retained.'};dump(target/'RETENTION.json',retention);dump(state/'COMPLETE.json',{'canonical':str(target),'archive_sha256':written['archive_sha256'],'retention_sha256':sha(target/'RETENTION.json'),'original_files_verified':retention['verified_regular_files'],'remote_original_preserved':True});print(json.dumps(read(state/'COMPLETE.json')),flush=True)
if __name__=='__main__':main()
