"""Single bounded read-only execution; preserve failure output without retry."""
from pathlib import Path
import hashlib,json,subprocess,time,argparse
p=Path(__file__).resolve().parent;s=p/'audit_saved_arrays.py';digest=hashlib.sha256(s.read_bytes()).hexdigest();assert digest=='1bba066f24683aec0b0ce2d096c25a21a8b276c03e3a7b01805b016ef592c364'
parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True);args=parser.parse_args()
assert hashlib.sha256(args.release.read_bytes()).hexdigest()==args.release_sha256
release=json.loads(args.release.read_text());expected={'schema':'ns-obstacle-coordinate-polish-velocity-decomposition-release-v1','authorized':True,'single_execution':True,'audit_source_sha256':digest,'timeout_seconds':120,'remote_alarm_seconds':115,'address_space_bytes':4*2**30,'BLAS_threads':1,'diagnostic07_audit_sha256':'3c9f4db61954e09de1d3357a5f146e0d9140b1356509035d7d0c65f0a1ecee19','diagnostic08_audit_sha256':'73de91920c80b51b15a8c714204aec2e9288604cb6582c24ede6c4f48f690e48','prepared_manifest_sha256':'7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6','projection':'weighted_L2_first_four_teacher_only','new_neural_FE_PDE_evaluation_allowed':False,'validation_allowed':False}
for key,value in expected.items():assert release[key]==value,(key,release.get(key),value)
if 'collector_sha256' in release:assert release['collector_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
binding={'diagnostics':[]}
for variant,folder,expected in [('coordinate','neural-coordinate-diagnostic-review','3c9f4db61954e09de1d3357a5f146e0d9140b1356509035d7d0c65f0a1ecee19'),('output_polish','neural-output-polish-diagnostic-review','73de91920c80b51b15a8c714204aec2e9288604cb6582c24ede6c4f48f690e48')]:
 audit=p.parent/folder/'RESULT.json';assert hashlib.sha256(audit.read_bytes()).hexdigest()==expected;j=json.loads(audit.read_text());assert j['passed'] and j['physical_admission'] is False
 row=json.loads((audit.parent/'BINDING.json').read_text());row.update(variant=variant,job_namespace='neural-pod-guided-diagnostic'+('07' if variant=='coordinate' else '08'),audit_sha256=expected,accepted_provenance=j['provenance']);binding['diagnostics'].append(row)
assert not (p/'BINDING.json').exists();(p/'BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
input_source=('BINDING='+repr(binding)+'\n').encode()+s.read_bytes()
for n in ('RESULT.json','STDERR.txt','EXECUTION.json'):assert not (p/n).exists(),n
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login','OPENBLAS_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -B -'];t=time.monotonic();timedout=False
try:
 x=subprocess.run(cmd,input=input_source,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120);out,err,code=x.stdout,x.stderr,x.returncode
except subprocess.TimeoutExpired as e:out=e.stdout or b'';err=e.stderr or b'';code=None;timedout=True
(p/'RESULT.json').write_bytes(out);(p/'STDERR.txt').write_bytes(err);r={'release_path':str(args.release),'release_sha256':args.release_sha256,'schema':'ns-obstacle-bounded-saved-array-audit-execution-v1','source_sha256':digest,'binding_sha256':hashlib.sha256((p/'BINDING.json').read_bytes()).hexdigest(),'executed_input_sha256':hashlib.sha256(input_source).hexdigest(),'command':cmd,'timeout_seconds':120,'timed_out':timedout,'exit_code':code,'stdout_sha256':hashlib.sha256(out).hexdigest(),'stderr_sha256':hashlib.sha256(err).hexdigest(),'elapsed_seconds':time.monotonic()-t,'no_retry':True};(p/'EXECUTION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
assert code==0 and not err and not timedout
j=json.loads(out);assert j['passed'];print(json.dumps({'cases':len(j['cases']),'elapsed_seconds':j['elapsed_seconds'],'peak_rss_bytes':j['peak_rss_bytes']}))
