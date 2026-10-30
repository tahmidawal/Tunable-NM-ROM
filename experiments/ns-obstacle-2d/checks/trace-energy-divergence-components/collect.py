"""Single bounded read-only execution; preserve failure output without retry."""
from pathlib import Path
import hashlib,json,subprocess,time
p=Path(__file__).resolve().parent;s=p/'audit_saved_arrays.py';digest=hashlib.sha256(s.read_bytes()).hexdigest();assert digest=='c06278715664601e77754710b64eeda2be9984d4665b5aa5c520b2dab82245a9'
binding={'diagnostics':[]}
for variant,folder,expected in [('trace','neural-trace-diagnostic-review','1d94ff32ae18d4ec5552363993ab8f15e58b842d3e87d36067a5946523ac8122'),('energy','neural-energy-diagnostic-review','92a0fb031343338790b1e5f6ca895572f5035a129ea7fba6221d43642b6379fb')]:
 audit=p.parent/folder/'RESULT.json';assert hashlib.sha256(audit.read_bytes()).hexdigest()==expected;j=json.loads(audit.read_text());assert j['passed'] and j['physical_admission'] is False
 row=json.loads((audit.parent/'BINDING.json').read_text());row.update(variant=variant,job_namespace='neural-pod-guided-diagnostic'+('03' if variant=='trace' else '04'),audit_sha256=expected,accepted_provenance=j['provenance']);binding['diagnostics'].append(row)
assert not (p/'BINDING.json').exists();(p/'BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
input_source=('BINDING='+repr(binding)+'\n').encode()+s.read_bytes()
for n in ('RESULT.json','STDERR.txt','EXECUTION.json'):assert not (p/n).exists(),n
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login','OPENBLAS_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -B -'];t=time.monotonic();timedout=False
try:
 x=subprocess.run(cmd,input=input_source,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120);out,err,code=x.stdout,x.stderr,x.returncode
except subprocess.TimeoutExpired as e:out=e.stdout or b'';err=e.stderr or b'';code=None;timedout=True
(p/'RESULT.json').write_bytes(out);(p/'STDERR.txt').write_bytes(err);r={'schema':'ns-obstacle-bounded-saved-array-audit-execution-v1','source_sha256':digest,'binding_sha256':hashlib.sha256((p/'BINDING.json').read_bytes()).hexdigest(),'executed_input_sha256':hashlib.sha256(input_source).hexdigest(),'command':cmd,'timeout_seconds':120,'timed_out':timedout,'exit_code':code,'stdout_sha256':hashlib.sha256(out).hexdigest(),'stderr_sha256':hashlib.sha256(err).hexdigest(),'elapsed_seconds':time.monotonic()-t,'no_retry':True};(p/'EXECUTION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
assert code==0 and not err and not timedout
j=json.loads(out);assert j['passed'];print(json.dumps({'cases':len(j['cases']),'elapsed_seconds':j['elapsed_seconds'],'peak_rss_bytes':j['peak_rss_bytes']}))
