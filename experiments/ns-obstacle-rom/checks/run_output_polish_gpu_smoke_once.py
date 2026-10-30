"""One root-authorized synthetic-only GPU QR fixture; no campaign inputs."""
from pathlib import Path
import argparse,hashlib,json,subprocess,os,time,shlex,shutil
p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);p.add_argument('--wrapper-sha256',required=True);a=p.parse_args()
B=Path(__file__).resolve().parents[1];repo=B.parents[1];out=B/'checks/output-polish-gpu-synthetic01';py='/home/tahmid/Dev/.venv/bin/python'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(__file__)==a.wrapper_sha256
assert len(a.source_commit)==40
sources={}
for name in ['accuracy_v2/output_polish_gpu_smoke.py','accuracy_v2/fixed_hidden_output_polish.py','accuracy_v2/spatial_bank.py','geometry.py']:
 f=B/name;assert not f.is_symlink();identity=sha(f);raw=subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),'show',a.source_commit+':'+str(f.relative_to(repo))]);assert hashlib.sha256(raw).hexdigest()==identity;sources[name]=identity
required=10*2**30+256*2**20+64*2**20;free=shutil.disk_usage(B).free;assert free>=required
units=subprocess.check_output(['systemctl','--user','list-units','--type=scope','--state=running','--no-legend','--plain'],text=True);active=[]
for row in units.splitlines():
 unit=row.split()[0];limit=subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryMax','--value'],text=True).strip()
 if limit==str(36*2**30):active.append(unit)
assert len(active)<3
claim=out.with_name(out.name+'-CLAIM.json');stdout=out.with_suffix('.stdout.txt');stderr=out.with_suffix('.stderr.txt');execution=out.with_name(out.name+'-EXECUTION.json');assert not any(x.exists() for x in [out,claim,stdout,stderr,execution])
record={'scope':'synthetic random matrices only; no TRAIN/DEV/VAL/model/solve','source_commit':a.source_commit,'source_files':sources,'wrapper_sha256':sha(__file__),'python_sha256':sha(py),'jaxrun_sha256':sha('/home/tahmid/.local/bin/jaxrun'),'existing_default_36GiB_scopes':active,'local_free_before_bytes':free,'required_local_free_bytes':required,'start_epoch':time.time(),'automatic_retry':False,'single_execution':True}
with claim.open('x') as f:json.dump(record,f,indent=2)
env=dict(os.environ,PYTHONPATH=str(B),JAX_ENABLE_X64='True',JAX_DEFAULT_MATMUL_PRECISION='highest',JAXRUN_MAX='36G',JAXRUN_HIGH='30G',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=['/usr/bin/timeout','--signal=TERM','--kill-after=1s','58s','/home/tahmid/.local/bin/jaxrun','/usr/bin/timeout','--signal=TERM','--kill-after=3s','50s',py,'-m','accuracy_v2.output_polish_gpu_smoke','--out',str(out)]
record['command']=command
shell='source /etc/profile.d/jax-mem.sh\nexec '+shlex.join(command)
with stdout.open('x') as so,stderr.open('x') as se:run=subprocess.run(['/bin/bash','-c',shell],env=env,cwd=repo,stdout=so,stderr=se)
record.update(returncode=run.returncode,finish_epoch=time.time(),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
execution.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));run.check_returncode()
