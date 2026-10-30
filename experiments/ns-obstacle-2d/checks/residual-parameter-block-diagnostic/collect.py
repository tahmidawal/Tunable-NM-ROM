"""One post-terminal read-only execution; preserve stdout/stderr even on timeout."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
p=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--audit',type=Path,required=True);a.add_argument('--audit-sha256',required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(args.audit)==args.audit_sha256
j=json.loads(args.audit.read_text());assert j['passed'] and j['schema']=='ns-obstacle-independent-residual-refinement-saved-audit-v1'
s=p/'audit_saved_blocks.py';expected=json.loads((p/'PROSPECTIVE.json').read_text())['source_sha256'];assert sha(s)==expected
assert not args.output.exists();args.output.mkdir(parents=True)
binding={'audit_sha256':args.audit_sha256,'model_sha256':j['selected_model_sha256'],'parent_model_sha256':j['parent_model_sha256']}
(args.output/'BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
source=('BINDING='+repr(binding)+'\nAUDIT='+repr(j)+'\n').encode()+s.read_bytes();cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','tufts-login','OPENBLAS_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -B -'];t=time.monotonic();timed_out=False
try:
 r=subprocess.run(cmd,input=source,capture_output=True,timeout=60);stdout,stderr,code=r.stdout,r.stderr,r.returncode
except subprocess.TimeoutExpired as e:stdout,stderr,code=e.stdout or b'',e.stderr or b'',None;timed_out=True
(args.output/'RESULT.json').write_bytes(stdout);(args.output/'STDERR.txt').write_bytes(stderr)
execution={'source_sha256':expected,'audit_sha256':args.audit_sha256,'binding_sha256':sha(args.output/'BINDING.json'),'input_sha256':hashlib.sha256(source).hexdigest(),'command':cmd,'exit_code':code,'timed_out':timed_out,'timeout_seconds':60,'no_retry':True,'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest(),'elapsed_seconds':time.monotonic()-t}
(args.output/'EXECUTION.json').write_text(json.dumps(execution,indent=2)+'\n');print(json.dumps(execution))
assert code==0 and not timed_out and not stderr and json.loads(stdout)['passed']
