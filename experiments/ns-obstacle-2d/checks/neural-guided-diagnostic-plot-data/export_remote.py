"""Read only completed actual invocations; never reconstruct, fit or solve."""
from pathlib import Path
import hashlib,json,io,sys,time,signal
import numpy as np
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('saved-output export60s')));signal.alarm(60)
b=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-diagnostic01');out=b/'results/diagnostic';provenance={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def decode(x,z):
 if isinstance(x,dict):
  if set(x)=={'array','shape','dtype','content_sha256'}:
   a=z[x['array']];assert list(a.shape)==x['shape'] and str(a.dtype)==x['dtype'];h=hashlib.sha256(str((a.dtype.str,a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());assert h.hexdigest()==x['content_sha256'];return a
  return {k:decode(v,z) for k,v in x.items()}
 if isinstance(x,list):return [decode(v,z) for v in x]
 return x
def load(receipt):
 name=receipt['record_file'];assert Path(name).name==name;p=out/name;assert sha(p)==receipt['record_sha256'];provenance[str(p)]=sha(p);d=json.loads(p.read_text());assert d['array_file']==receipt['array_file'] and d['array_sha256']==receipt['array_sha256'];a=out/d['array_file'];assert Path(d['array_file']).name==d['array_file'] and sha(a)==d['array_sha256'];provenance[str(a)]=sha(a)
 with np.load(a,allow_pickle=False) as z:return decode(d['record'],z)
p=out/'DIAGNOSTIC.json';assert sha(p)==STATUS_SHA;status=json.loads(p.read_text());assert status['status']=='completed_diagnostic_pending_independent_review' and status['completed_case_records']==2;assert status['physical_admission'] is False;provenance[str(p)]=sha(p)
record=load(status['latest_evidence']);assert record['model_sha256']=='faab6651618ae9d6afc79dbd91ad6e9c7a1a3101907d93d9e33ab37dfba2e4c6' and record['original_bank_gate_passed'] is False
assert record['validation_seen'] is False and record['reference_fit_initialization_used'] is False and record['physical_admission'] is False;assert [r['Re'] for r in record['cases']]==[16.875,23.125]
ids=np.asarray(INDICES,dtype=np.int64);assert len(ids)==18000 and hashlib.sha256(ids.tobytes()).hexdigest()==INDICES_SHA;fields={'velocity':[],'pressure':[]};rows=[]
for row in record['cases']:
 invocation=load(row['raw_invocation']);outputs=invocation['outputs'];assert outputs['velocity'].shape==(1036497,2) and outputs['pressure'].shape==(1036497,)
 for key in fields:
  a=outputs[key];assert a.dtype==np.float64 and np.isfinite(a).all();fields[key].append(a[ids])
 rows.append({'Re':row['Re'],'recorded_full_rule_scores':clean(row.get('scores')),'solver_status':row.get('solver_status'),'work':clean(invocation['work']),'diagnostic_numerical_targets_passed':row.get('diagnostic_numerical_targets_passed'),'physical_admission':False,'eligible_accuracy':False,'raw_invocation_receipt':row['raw_invocation'],'representation_diagnostic':clean(row.get('representation_diagnostic'))})
metadata={'schema':'ns-obstacle-actual-guided-neural-diagnostic-plot-extension-v1','audit_result_sha256':AUDIT_SHA,'base_plot_archive_sha256':BASE_SHA,'indices_sha256':INDICES_SHA,'same_original_indices':True,'kind':'actual user-authorized exact weak-LS neural-ROM diagnostic invocation; NOT best-fit projection','units':{'velocity':'u/U','pressure':'p/U^2','U':.2,'D':.1},'cases':rows,'source_sha256':{name:sha(b/name) for name in ('accuracy_v2/diagnostic.py','accuracy_v2/spatial_bank.py','pilot2d.py','pilot2d_blocked.py','weak.py')},'release_inputs_sha256':{name:sha(b/name) for name in ('PLAN.json','DIAGNOSTIC-RELEASE.json','RUNTIME-INVENTORY.json')},'retained_output_sha256':provenance,'original_representation_gate_passed':False,'physical_admission':False,'new_solves_or_reconstruction':False,'validation_accessed':False,'elapsed_seconds':time.monotonic()-start}
arrays={'pod_guided_neural_r4_solve_'+key:np.stack(value) for key,value in fields.items()};arrays['metadata_json']=np.frombuffer(json.dumps(metadata,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8);buf=io.BytesIO();np.savez_compressed(buf,**arrays);payload=buf.getvalue();assert len(payload)<=1200000;signal.alarm(0);sys.stdout.buffer.write(payload)
