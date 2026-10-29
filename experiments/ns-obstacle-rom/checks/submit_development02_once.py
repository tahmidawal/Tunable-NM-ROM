from pathlib import Path
import subprocess,json,shutil
b=Path(__file__).resolve().parents[1];packet=b/'staging/development-materialize02';r=json.loads((packet/'HANDOFF-RELEASE.json').read_text());s=json.loads((packet/'STAGING.json').read_text());assert shutil.disk_usage(b).free>=r['local_floor_bytes']+r['other_local_reserved_bytes']
code='''from pathlib import Path
import json,hashlib,subprocess,os,time,sys,shutil
root=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/development-materialize02');os.chdir(root);sys.path.insert(0,str(root))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha('HANDOFF-RELEASE.json')=='e64ebe7b17b194d1e00df6185d19c359510e41fc2b7a60bf838dd033b423aebe';r=json.loads(Path('HANDOFF-RELEASE.json').read_text());assert sha('SOURCE-SHA256SUMS')=='655b758e4847dd60c459504d435ed365aa6a69925f3225c4137a634a539903dd';subprocess.run(['sha256sum','--status','-c','SOURCE-SHA256SUMS'],check=True)
from development_handoff_launch2d import verify
verify('HANDOFF-PLAN.json','HANDOFF-RELEASE.json','INPUT-INVENTORY.json');assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
assert not Path('SUBMISSION-CLAIM.json').exists() and not Path('DEVELOPMENT-HANDOFF-CLAIM.json').exists()
base=root.parent.parent;names=['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'];usage={n:int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in names};projected=sum(usage.values())+r['remote_workspace_archive_envelope_bytes']+r['other_active_disk_reservation_bytes'];assert projected<r['global_remote_limit_bytes'] and shutil.disk_usage(base).free>16*2**30
queue=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True);assert sum('ctol_nsobs' in line for line in queue.splitlines())<2
record={'release_sha256':sha('HANDOFF-RELEASE.json'),'source_manifest_sha256':sha('SOURCE-SHA256SUMS'),'queue_before':queue,'namespace_bytes':usage,'maximum_global_bytes':projected,'time_unix':time.time(),'actual_full_release_verification_before_submit':True,'native_or_jax_imported_by_preflight':False}
with Path('SUBMISSION-CLAIM.json').open('x') as f:json.dump(record,f,indent=2)
p=subprocess.run(['sbatch','--parsable','native.sh'],capture_output=True,text=True);Path('SBATCH.stdout').write_text(p.stdout);Path('SBATCH.stderr').write_text(p.stderr);p.check_returncode();job=p.stdout.strip().split(';')[0];assert job.isdigit();record.update(job_id=int(job),queue_after=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%i|%j|%T'],text=True),submitted_once=True)
Path('SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
r=json.loads(subprocess.check_output(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python','-'],input=code.encode()));(b/'checks/development-materialization02-submission.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
