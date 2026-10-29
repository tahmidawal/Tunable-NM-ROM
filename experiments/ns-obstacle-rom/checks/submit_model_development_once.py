from pathlib import Path
import subprocess,json,hashlib,shutil
B=Path(__file__).resolve().parents[1];P=B/'staging/model-development01';root=B.parents[1].parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/model-development01';job='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/model-development01';py='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=root/'STAGE-RELEASE.json';assert sha(r)=='f69c8b42ef5dd0c2ddbfead1d9e7ae97dd5f63ef731345502f850c07097f66eb';assert shutil.disk_usage(B).free>=10*2**30+256*2**20
for p in sorted(root.iterdir()):
 if p.is_file():shutil.copyfile(p,P/p.name);subprocess.run(['scp','-q',str(p),'tufts-login:'+job+'/'+p.name],check=True)
source=subprocess.check_output(['git','-c','gc.auto=0','rev-parse','7616a27b5'],text=True).strip()
code=f'''from pathlib import Path
import os,sys,json,hashlib,subprocess,shutil,time
root=Path({job!r});os.chdir(root);sys.path.insert(0,str(root))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha('STAGE-RELEASE.json')=='f69c8b42ef5dd0c2ddbfead1d9e7ae97dd5f63ef731345502f850c07097f66eb'
from development_launch2d import verify
c=verify('STAGE-CANDIDATE.json','STAGE-RELEASE.json','RUNTIME-INVENTORY.json');assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
pop=Path(c['inputs']['development']['path']);m=json.loads(pop.read_text());assert [x['Re'] for x in m['cases']]==[16.875,23.125];fields=[]
for row in m['cases']:
 p=pop.parent/row['array_file'];assert not p.is_symlink() and p.resolve().is_relative_to(pop.parent) and sha(p)==row['array_sha256'];fields.append({{'Re':row['Re'],'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}})
assert not Path('STAGE-CLAIM.json').exists() and not Path('SUBMISSION-CLAIM.json').exists()
base=root.parent.parent;names=['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'];usage={{n:int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in names}};maximum=sum(usage.values())+12*2**30+256*2**20;assert maximum<50*2**30 and shutil.disk_usage(base).free>22*2**30
queue=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True);assert sum('ctol_nsobs' in x for x in queue.splitlines())<2
Path('SOURCE_COMMIT.txt').write_text({source!r}+'\\n');files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts];assert not Path('SOURCE-SHA256SUMS').exists();Path('SOURCE-SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\\n' for p in sorted(files)))
subprocess.run(['sha256sum','--status','-c','SOURCE-SHA256SUMS'],check=True)
record={{'release_sha256':sha('STAGE-RELEASE.json'),'candidate_sha256':sha('STAGE-CANDIDATE.json'),'source_manifest_sha256':sha('SOURCE-SHA256SUMS'),'source_commit':{source!r},'actual_full_release_preflight_passed':True,'both_manifest_bound_development_fields':fields,'native_or_JAX_imported_before_submission':False,'queue_before':queue,'namespace_bytes':usage,'maximum_global_bytes':maximum,'output_archive_envelope_bytes':12*2**30,'time_unix':time.time()}}
Path('PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\\n')
with Path('SUBMISSION-CLAIM.json').open('x') as f:json.dump(record,f,indent=2)
p=subprocess.run(['sbatch','--parsable','native.sh'],capture_output=True,text=True);Path('SBATCH.stdout').write_text(p.stdout);Path('SBATCH.stderr').write_text(p.stderr);p.check_returncode();job=p.stdout.strip().split(';')[0];assert job.isdigit();record.update(job_id=int(job),submitted_once=True,queue_after=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True));Path('SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
receipt=json.loads(subprocess.check_output(['ssh','tufts-login',py,'-'],input=code.encode()));(B/'checks/model-development-submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
