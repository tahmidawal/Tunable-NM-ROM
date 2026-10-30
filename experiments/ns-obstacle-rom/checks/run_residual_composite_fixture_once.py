"""Future single bounded local composite evaluator fixture; requires exact release."""
from pathlib import Path
import argparse,hashlib,json,subprocess,os,time,shlex
p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);a=p.parse_args()
B=Path(__file__).resolve().parents[1];repo=B.parents[1];out=B/'checks/residual-composite-fixture01';py='/home/tahmid/Dev/.venv/bin/python';points=B/'checks/fast-traction-bank-kernel/arrays.npz'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(a.release)==a.release_sha256;r=json.loads(a.release.read_text())
expected={'schema':'ns-obstacle-residual-composite-fixture-release-v1','authorized':True,'single_execution':True,'timeout_seconds':50,'kill_grace_seconds':3,'outer_timeout_seconds':58,'outer_kill_grace_seconds':1,'python_executable':py,'memory_max':'36G','memory_high':'30G','GPU_required':True,'x64_required':True,'matmul_precision':'highest','output_directory':str(out),'wrapper_sha256':sha(__file__),'fixture_source_sha256':sha(B/'accuracy_v2/residual_composite_fixture.py'),'points_fixture_sha256':sha(points),'automatic_retry':False}
assert all(type(r.get(k)) is type(v) and r[k]==v for k,v in expected.items())
model=Path(r['model_path']);review=Path(r['model_review_path']);assert model.is_file() and not model.is_symlink() and sha(model)==r['model_sha256'] and sha(review)==r['model_review_sha256'];m=json.loads(review.read_text());assert m['passed'] is True and m['model_sha256']==r['model_sha256']
for name,identity in r['source_files'].items():
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts;f=B/rel;assert sha(f)==identity
 raw=subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),'show',r['source_commit']+':'+str(f.relative_to(repo))]);assert hashlib.sha256(raw).hexdigest()==identity
required={'accuracy_v2/residual_composite_fixture.py','accuracy_v2/residual_head_bank.py','accuracy_v2/independent_residual_heads.py','accuracy_v2/spatial_bank.py','accuracy_v2/fast_spatial_bank.py','accuracy_v2/train.py','geometry.py','qualification.py','MODEL-CONTRACT.json','FD1-SCOPE-CONTRACT.json'}
assert set(r['source_files'])==required
# Check all active default36G jaxrun scopes conservatively before this one.
units=subprocess.check_output(['systemctl','--user','list-units','--type=scope','--state=running','--no-legend','--plain'],text=True);active=[]
for row in units.splitlines():
 unit=row.split()[0];limit=subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryMax','--value'],text=True).strip()
 if limit==str(36*2**30):active.append(unit)
assert len(active)<3
assert not out.exists();claim=out.with_name(out.name+'-CLAIM.json');stdout=out.with_suffix('.stdout.txt');stderr=out.with_suffix('.stderr.txt');execution=out.with_name(out.name+'-EXECUTION.json');assert not any(x.exists() for x in [claim,stdout,stderr,execution])
record={'release_sha256':sha(a.release),'source_commit':r['source_commit'],'model_sha256':r['model_sha256'],'existing_default_36GiB_scopes':active,'start_epoch':time.time(),'automatic_retry':False}
with claim.open('x') as f:json.dump(record,f,indent=2)
env=dict(os.environ,PYTHONPATH=str(B),JAX_ENABLE_X64='True',JAX_DEFAULT_MATMUL_PRECISION='highest',JAXRUN_MAX='36G',JAXRUN_HIGH='30G')
command=['/usr/bin/timeout','--signal=TERM','--kill-after=1s','58s','/home/tahmid/.local/bin/jaxrun','/usr/bin/timeout','--signal=TERM','--kill-after=3s','50s',py,'-m','accuracy_v2.residual_composite_fixture','--model',str(model),'--model-sha256',r['model_sha256'],'--points-fixture',str(points),'--points-sha256',sha(points),'--output',str(out)]
shell='source /etc/profile.d/jax-mem.sh\nexec '+shlex.join(command)
with stdout.open('x') as so,stderr.open('x') as se:run=subprocess.run(['/bin/bash','-c',shell],env=env,cwd=repo,stdout=so,stderr=se)
record.update(returncode=run.returncode,finish_epoch=time.time(),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),scope='one actual-model evaluator fixture; no TRAIN/DEV/VAL data, solve or fitting; cgroup bounds CPU RSS and XLA profile controls GPU allocation')
execution.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));run.check_returncode()
