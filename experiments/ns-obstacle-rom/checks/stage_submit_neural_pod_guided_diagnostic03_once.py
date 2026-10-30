from pathlib import Path
import subprocess,json,hashlib,shutil,shlex,argparse
a=argparse.ArgumentParser();a.add_argument('--release',type=Path,required=True);a.add_argument('--release-sha256',required=True);a.add_argument('--source-commit',required=True);a=a.parse_args()
B=Path(__file__).resolve().parents[1];P=B/'staging/neural-pod-guided-diagnostic03';R=a.release.resolve().parent;job='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-diagnostic03';py='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
release=a.release.resolve();assert sha(release)==a.release_sha256
r=json.loads(release.read_text());inv=json.loads((P/'RUNTIME-INVENTORY.json').read_text());assert r['source_commit']==a.source_commit;assert sha(P/'RUNTIME-INVENTORY.json')==r['runtime_inventory_sha256'] and sha(P/'PLAN.json')==r['plan_sha256'];assert shutil.disk_usage(B).free>=10*2**30+256*2**20+64*2**20+sum(x['bytes'] for x in inv['files'])
assert sha(P/'RESOURCE-CHECK.json')==r['resource_check_sha256'];assert sha(Path(__file__))==r['stager_sha256'];assert shutil.disk_usage(B).free>=json.loads((P/'RESOURCE-CHECK.json').read_text())['local_required_bytes']
rows=inv['files'];seen=set()
for row in rows:
 p=Path(row['source']).resolve();d=Path(row['destination']);assert d not in seen and not d.is_absolute() and '..' not in d.parts;seen.add(d);assert not p.is_symlink() and p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
for row in rows:
 raw=subprocess.check_output(['git','-c','gc.auto=0','show',a.source_commit+':'+str(Path(row['source']).resolve().relative_to(B.parents[1]))]);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
initial=f'''from pathlib import Path
import subprocess,json,shutil
p=Path({job!r});base=p.parent.parent;names=['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'];u={{n:int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in names}};total=sum(u.values())+11*2**30+256*2**20+{sum(x['bytes'] for x in rows)};assert total<50*2**30 and shutil.disk_usage(base).free>12*2**30
q=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True);assert sum('ctol_nsobs' in x for x in q.splitlines())<2;p.mkdir(exist_ok=False);print(json.dumps({{'namespace_bytes':u,'maximum_global_bytes':total,'queue':q}}))
'''
pre=json.loads(subprocess.check_output(['ssh','tufts-login',py,'-'],input=initial.encode()));(P/'STAGING-PREFLIGHT.json').write_text(json.dumps(pre,indent=2)+'\n')
dirs=sorted({str(Path(job)/Path(row['destination']).parent) for row in rows});subprocess.run(['ssh','tufts-login','mkdir -p -- '+' '.join(shlex.quote(x) for x in dirs)],check=True)
for row in rows:subprocess.run(['scp','-q',str(Path(row['source']).resolve()),'tufts-login:'+job+'/'+row['destination']],check=True)
for name in ('PLAN.json','RUNTIME-INVENTORY.json','RESOURCE-CHECK.json'):
 subprocess.run(['scp','-q',str(P/name),'tufts-login:'+job+'/'+name],check=True)
shutil.copyfile(release,P/'DIAGNOSTIC-RELEASE.json');subprocess.run(['scp','-q',str(release),'tufts-login:'+job+'/DIAGNOSTIC-RELEASE.json'],check=True)
source=subprocess.check_output(['git','-c','gc.auto=0','rev-parse',a.source_commit],text=True).strip()
final=f'''from pathlib import Path
import sys,os,subprocess,json,hashlib,shutil,time
p=Path({job!r});os.chdir(p);sys.path.insert(0,str(p))
def sha(x):
 with Path(x).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha('DIAGNOSTIC-RELEASE.json')=={a.release_sha256!r}
from accuracy_v2.diagnostic03_launch import verify
plan,r,inputs=verify('PLAN.json','DIAGNOSTIC-RELEASE.json','RUNTIME-INVENTORY.json');assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
base=p.parent.parent;names=['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'];u={{n:int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in names}};total=sum(u.values())+11*2**30+256*2**20;assert total<50*2**30 and shutil.disk_usage(base).free>12*2**30
q=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True);assert sum('ctol_nsobs' in x for x in q.splitlines())<2
assert not Path('DIAGNOSTIC-CLAIM.json').exists() and not Path('SUBMISSION-CLAIM.json').exists()
Path('SOURCE_COMMIT.txt').write_text({source!r}+'\\n');files=sorted(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts);Path('SOURCE-SHA256SUMS').write_text(''.join(sha(x)+'  '+str(x.relative_to(p))+'\\n' for x in files));subprocess.run(['sha256sum','--status','-c','SOURCE-SHA256SUMS'],check=True)
record={{'source_commit':{source!r},'release_sha256':sha('DIAGNOSTIC-RELEASE.json'),'source_manifest_sha256':sha('SOURCE-SHA256SUMS'),'actual_full_preflight_passed':True,'actual_accepted_model_and_development_verified':True,'no_JAX_or_native_import':True,'namespace_bytes':u,'maximum_global_bytes':total,'queue_before':q,'time_unix':time.time()}}
Path('PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\\n')
with Path('SUBMISSION-CLAIM.json').open('x') as f:json.dump(record,f,indent=2)
run=subprocess.run(['sbatch','--parsable','native.sh'],capture_output=True,text=True);Path('SBATCH.stdout').write_text(run.stdout);Path('SBATCH.stderr').write_text(run.stderr);run.check_returncode();jid=run.stdout.strip().split(';')[0];assert jid.isdigit();record.update(job_id=int(jid),submitted_once=True,queue_after=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True));Path('SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
receipt=json.loads(subprocess.check_output(['ssh','tufts-login',py,'-'],input=final.encode()));(B/'checks/neural-pod-guided-diagnostic03-submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
