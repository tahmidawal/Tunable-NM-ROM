from pathlib import Path
import hashlib,json,subprocess,time,numpy as np
p=Path(__file__).resolve().parent;b=p.parents[1];a=b/'checks/neural-trace-diagnostic-review';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(a/'RESULT.json')=='1d94ff32ae18d4ec5552363993ab8f15e58b842d3e87d36067a5946523ac8122'
ad=b/'g2/population01/complete15/REFERENCE-ADMISSION.json';assert sha(ad)=='5c9402af9ddb2af0ad8925ef224a43ae72cb36209df99865908617c63b5b506f';admission=json.loads(ad.read_text())
audit=json.loads((a/'RESULT.json').read_text());assert audit['passed'];binding=json.loads((a/'BINDING.json').read_text());truth=[]
for re,key in [(16.875,'re16p875'),(23.125,'re23p125')]:
 d=b/'artifacts'/('g2-population-'+key+'-01')/'results/states/p4_g4_h3_baseline';names=('config.json','result.json','boundary_rule.json');export=d/'EXPORT-MANIFEST.json';member=[r for r in admission['admitted_references'] if r['Re']==re and r['split']=='development'];assert len(member)==1 and sha(export)==member[0]['export_manifest_sha256'];manifest=json.loads(export.read_text());assert all(sha(d/n)==manifest['files'][n] for n in names);raw={n:json.loads((d/n).read_text()) for n in names};cfg=raw['config.json'];m=raw['result.json']['metrics']['forces'];rows=[r for r in raw['boundary_rule.json'] if r['label']=='obstacle'];assert cfg['diagnostic_order']==12;w=np.array([r['weight'] for r in rows]);n=np.array([r['normal'] for r in rows]);v=np.array([r['pressure'] for r in rows]);g=np.array([r['velocity_gradient'] for r in rows]);pressure=np.einsum('n,n,ni->i',w,v,n)/m['normalization'];visc=-cfg['nu']*np.einsum('n,nij,nj->i',w,g+g.transpose(0,2,1),n)/m['normalization'];t12=np.array(m['physical_symmetric_order12']['coefficient']);t16=np.array(m['physical_symmetric_order16']['coefficient']);assert np.allclose(pressure+visc,t12,rtol=2e-12,atol=1e-13)
 truth.append(dict(Re=re,pressure_coefficient=pressure.tolist(),viscous_coefficient=visc.tolist(),order12_total=t12.tolist(),order16_total=t16.tolist(),order12_minus_order16=(t12-t16).tolist(),native_order=12,obstacle_points=len(w),normalization=m['normalization'],provenance={str(d/n):sha(d/n) for n in names+('EXPORT-MANIFEST.json',)},reference_admission_sha256=sha(ad)))
s=p/'audit_remote.py';source='BINDING='+repr(binding)+'\nTRUTH='+repr(truth)+'\nACCEPTED_PROVENANCE='+repr(audit['provenance'])+'\n'+s.read_text()
for n in ('RESULT.json','STDERR.txt','EXECUTION.json'):assert not (p/n).exists()
t=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','tufts-login','OPENBLAS_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -B -'],input=source.encode(),capture_output=True,timeout=120)
except subprocess.TimeoutExpired as e:
 from types import SimpleNamespace
 r=SimpleNamespace(stdout=e.stdout or b'',stderr=e.stderr or b'',returncode=None)
(p/'RESULT.json').write_bytes(r.stdout);(p/'STDERR.txt').write_bytes(r.stderr);(p/'EXECUTION.json').write_text(json.dumps(dict(source_sha256=sha(s),collector_sha256=sha(Path(__file__)),audit_sha256=sha(a/'RESULT.json'),executed_input_sha256=hashlib.sha256(source.encode()).hexdigest(),exit_code=r.returncode,timed_out=r.returncode is None,elapsed_seconds=time.monotonic()-t,stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest(),no_retry=True),indent=2)+'\n');assert r.returncode==0 and not r.stderr;print(sha(p/'RESULT.json'))
