"""Single fresh cluster fallback; exact coordinator release and guards required."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,shlex
p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
B=Path(__file__).resolve().parents[1];repo=B.parents[1];P=B/'staging/scanned-composite-fixture-cluster01'
JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/scanned-composite-fixture-cluster01'
PY='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(a.release)==a.release_sha256;r=json.loads(a.release.read_text());plan=json.loads((P/'PLAN.json').read_text());inv=json.loads((P/'RUNTIME-INVENTORY.json').read_text())
expected={'schema':'ns-obstacle-scanned-cluster-fixture-release-v1','authorized':True,'single_execution':True,'automatic_retry':False,'job_directory':JOB,'source_commit':a.source_commit,'stager_sha256':sha(__file__),'plan_sha256':sha(P/'PLAN.json'),'resource_check_sha256':sha(P/'RESOURCE-CHECK.json'),'runtime_inventory_sha256':sha(P/'RUNTIME-INVENTORY.json'),'allocation_seconds':120,'outer_timeout_seconds':105,'kill_grace_seconds':5,'memory_bytes':16*2**30,'raw_plus_archive_reservation_bytes':256*2**20,'model_sha256':plan['model_sha256'],'model_review_sha256':plan['model_review_sha256'],'points_sha256':plan['points_sha256'],'fixture_source_sha256':plan['fixture_source_sha256'],'prior_failed_execution_sha256':plan['prior_failed_execution_sha256']}
assert all(type(r.get(k)) is type(v) and r[k]==v for k,v in expected.items())
rows=inv['files'];seen=set()
for row in rows:
    f=Path(row['source']);d=Path(row['destination']);assert not d.is_absolute() and '..' not in d.parts and str(d) not in seen;seen.add(str(d))
    assert not f.is_symlink() and f.is_file() and f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
    raw=subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),'show',a.source_commit+':'+str(f.relative_to(repo))]);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
assert shutil.disk_usage(B).free>=10*2**30+256*2**20+64*2**20+sum(x['bytes'] for x in rows)
remote_pre=f'''from pathlib import Path
import json,subprocess,shutil
p=Path({JOB!r});base=p.parent.parent;names=['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'];u={{n:int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in names}}
total=sum(u.values())+15*2**30+256*2**20+{sum(x['bytes'] for x in rows)};assert total<50*2**30 and shutil.disk_usage(base).free>16*2**30
q=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True);assert sum('ctol_nsobs' in x for x in q.splitlines())<2
assert not p.exists();p.mkdir();print(json.dumps({{'queue_before':q,'namespace_bytes':u,'prospective_global_bytes':total}}))
'''
pre=json.loads(subprocess.check_output(['ssh','tufts-login',PY,'-'],input=remote_pre.encode()));(P/'STAGING-PREFLIGHT.json').write_text(json.dumps(pre,indent=2)+'\n')
dirs=sorted({str(Path(JOB)/Path(x['destination']).parent) for x in rows});subprocess.run(['ssh','tufts-login','mkdir -p -- '+' '.join(shlex.quote(x) for x in dirs)],check=True)
for row in rows:subprocess.run(['scp','-q',row['source'],'tufts-login:'+JOB+'/'+row['destination']],check=True)
for f,d in [(P/'RUNTIME-INVENTORY.json','RUNTIME-INVENTORY.json'),(a.release,'EXECUTION-RELEASE.json')]:subprocess.run(['scp','-q',str(f),'tufts-login:'+JOB+'/'+d],check=True)
final=f'''from pathlib import Path
import hashlib,json,subprocess,shutil
p=Path({JOB!r});sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(p/'RUNTIME-INVENTORY.json')=={sha(P/'RUNTIME-INVENTORY.json')!r} and sha(p/'EXECUTION-RELEASE.json')=={a.release_sha256!r}
inv=json.loads((p/'RUNTIME-INVENTORY.json').read_text())
for x in inv['files']:
 f=p/x['destination'];assert not f.is_symlink() and f.stat().st_size==x['bytes'] and sha(f)==x['sha256']
base=p.parent.parent;total=sum(int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in ['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'])+15*2**30+256*2**20;assert total<50*2**30 and shutil.disk_usage(base).free>16*2**30
q=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True);assert sum('ctol_nsobs' in x for x in q.splitlines())<2
assert not (p/'SUBMISSION.json').exists();(p/'SUBMISSION-CLAIM.json').open('x').write(json.dumps({{'release_sha256':{a.release_sha256!r},'automatic_retry':False}}))
lines=[x['sha256']+'  '+x['destination'] for x in inv['files']];lines += [{sha(P/'RUNTIME-INVENTORY.json')!r}+'  RUNTIME-INVENTORY.json',{a.release_sha256!r}+'  EXECUTION-RELEASE.json'];(p/'SOURCE-SHA256SUMS').write_text('\\n'.join(lines)+'\\n')
submission=subprocess.run(['sbatch','--parsable','native.sh'],cwd=p,text=True,capture_output=True)
(p/'SBATCH-RESULT.json').write_text(json.dumps({{'returncode':submission.returncode,'stdout':submission.stdout,'stderr':submission.stderr,'automatic_retry':False}},indent=2)+'\\n')
assert submission.returncode==0;job=submission.stdout.strip();assert job.isdigit()
after=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True)
receipt={{'job_id':int(job),'source_commit':{a.source_commit!r},'release_sha256':{a.release_sha256!r},'queue_before':q,'queue_after':after,'prospective_global_bytes':total,'all_runtime_files_verified':True,'automatic_retry':False}}
(p/'SUBMISSION.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''
receipt=json.loads(subprocess.check_output(['ssh','tufts-login',PY,'-'],input=final.encode()));out=B/'checks/scanned-composite-fixture-cluster01-submission.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
