"""Single bounded read-only execution; preserve failure output without retry."""
from pathlib import Path
import hashlib,json,subprocess,time
p=Path(__file__).resolve().parent;s=p/'audit_saved_arrays.py';digest=hashlib.sha256(s.read_bytes()).hexdigest();assert digest=='0e8a3786f1191207ec50fa27b86eff756a35278265d894e7469f666bfd93c14f'
binding=json.loads((p/'BINDING.json').read_text());input_source=('BINDING='+repr(binding)+'\n').encode()+s.read_bytes()
for n in ('RESULT.json','STDERR.txt','EXECUTION.json'):assert not (p/n).exists(),n
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login','OPENBLAS_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -B -'];t=time.monotonic();timedout=False
try:
 x=subprocess.run(cmd,input=input_source,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120);out,err,code=x.stdout,x.stderr,x.returncode
except subprocess.TimeoutExpired as e:out=e.stdout or b'';err=e.stderr or b'';code=None;timedout=True
(p/'RESULT.json').write_bytes(out);(p/'STDERR.txt').write_bytes(err);r={'schema':'ns-obstacle-bounded-saved-array-audit-execution-v1','source_sha256':digest,'binding_sha256':hashlib.sha256((p/'BINDING.json').read_bytes()).hexdigest(),'executed_input_sha256':hashlib.sha256(input_source).hexdigest(),'command':cmd,'timeout_seconds':120,'timed_out':timedout,'exit_code':code,'stdout_sha256':hashlib.sha256(out).hexdigest(),'stderr_sha256':hashlib.sha256(err).hexdigest(),'elapsed_seconds':time.monotonic()-t,'no_retry':True};(p/'EXECUTION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
assert code==0 and not err and not timedout
j=json.loads(out);assert j['passed'];print(json.dumps({'cases':len(j['cases']),'elapsed_seconds':j['elapsed_seconds'],'peak_rss_bytes':j['peak_rss_bytes']}))
