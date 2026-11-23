"""Collect only explicit owned terminal jobs, verify every scientific file, retain logs."""
from pathlib import Path
import subprocess,json,hashlib,re,sys
L=Path(__file__).resolve().parents[1];NS='/cluster/tufts/paralab/tawal01/jcp-width-deploy-20261119'
a,j=sys.argv[1:];assert re.fullmatch('[a-z][a-z0-9]+',a) and j.isdigit()
folder=L/'runs'/a;remote=f'{NS}/{a}'
raw=subprocess.check_output(['ssh','tufts-login',f'sacct -j {j} --noheader --parsable2 --format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES'],text=True)
row=next(x.split('|') for x in raw.splitlines() if x.split('|')[0]==j)
assert row[1]=='ctol_jwd_'+a and row[2] in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED']
(folder/'sacct.txt').write_text(raw)
cmd=f"cd {remote} && find code output -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > RESULTS.sha256 && sha256sum job.log run.sbatch MANIFEST.sha256 >> RESULTS.sha256"
subprocess.check_call(['ssh','tufts-login',cmd])
subprocess.check_call(['rsync','-a','--exclude=cache/','--exclude=tmp/','--exclude=__pycache__/','tufts-login:'+remote+'/',str(folder)+'/'])
verified=[]
for line in (folder/'RESULTS.sha256').read_text().splitlines():
 expected,rel=line.split('  ',1);p=folder/rel;assert p.resolve().is_relative_to(folder.resolve())
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 assert h.hexdigest()==expected,p
 verified.append(dict(path=rel,sha256=expected,bytes=p.stat().st_size))
log=(folder/'job.log').read_text();envpath=folder/'output/environment.json';env=json.loads(envpath.read_text()) if envpath.exists() else {}
warning=re.findall(r'(?im)^.*(?:captured.*constant|large.*constant|out.of.memory|no space left|traceback|cuInit|killed|oom-kill).*$',log)
valid=bool('jax_backend=gpu' in log and env.get('backend')=='gpu' and env.get('x64') and env.get('matmul_precision')=='highest')
c=dict(job_id=j,scheduler_state=row[2],exit_code=row[3],allocated_gpu_seconds=int(row[4]),allocation=row[5],checksum_verified=True,file_count=len(verified),files=verified,gpu_log_valid=valid,no_environmental_warnings=not warning,environmental_warning_lines=warning,terminal_marker_present='completed=True' in log,remote_directory_deleted=False,remote_directory=remote,scientifically_qualified=bool(valid and not warning and row[2]=='COMPLETED' and row[3]=='0:0' and 'completed=True' in log))
(folder/'collection.json').write_text(json.dumps(c,indent=2)+'\n')
lp=L/'runs/LEDGER.json';ledger=json.loads(lp.read_text());existing=next((x for x in ledger['attempts'] if x['attempt']==a),None)
if existing is None:
 existing={'attempt':a,'job_id':j};ledger['attempts'].append(existing)
existing.update(state=row[2],allocated_gpu_seconds=int(row[4]),source_commit=env.get('commit'),remote=remote)
ledger['total_allocated_gpu_seconds']=sum(x.get('allocated_gpu_seconds',0) for x in ledger['attempts']);assert ledger['total_allocated_gpu_seconds']<=ledger['ceiling_allocated_gpu_seconds']
lp.write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({k:v for k,v in c.items() if k!='files'}))
