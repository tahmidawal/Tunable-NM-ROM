"""Read only completed actual invocations; never reconstruct, fit or solve."""
from pathlib import Path
import hashlib,json,io,sys,time,signal
import numpy as np
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('saved-output export60s')));signal.alarm(60)
b=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/pod-recovery01');out=b/'results/recovery';provenance={}
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
p=out/'RECOVERY.json';assert sha(p)==STATUS_SHA;status=json.loads(p.read_text());assert status['status']=='completed_unsampled_controls_pending_independent_review' and status['validation_seen'] is False;provenance[str(p)]=sha(p)
ids=np.asarray(INDICES,dtype=np.int64);assert len(ids)==18000 and hashlib.sha256(ids.tobytes()).hexdigest()==INDICES_SHA
arrays={};rows=[]
expected={f'POD_r{r}_{re}_{method}_raw_invocation' for r in (4,8) for re in (16.875,23.125) for method in ('exact_tensor_weak_LS','coefficient_PCHIP','mixed_Galerkin')}
receipts={q['key']:q for q in status['query_receipts']};assert set(receipts)==expected
for rank in (4,8):
 for method in ('exact_tensor_weak_LS','coefficient_PCHIP','mixed_Galerkin'):
  fields={'velocity':[],'pressure':[]}
  for re in (16.875,23.125):
   key=f'POD_r{rank}_{re}_{method}_raw_invocation';q=receipts[key];assert not q['failed'];invocation=load(q['receipt']);case=load(status['cases'][f'POD_r{rank}_development_{re}']['receipt']);row=case['methods'][method]
   assert np.array_equal(invocation['evidence']['state'],row['invocation']['evidence']['state'])
   for field,shape in [('velocity',(1036497,2)),('pressure',(1036497,))]:
    a=invocation['outputs'][field];assert a.shape==shape and a.dtype==np.float64 and np.isfinite(a).all() and np.array_equal(a,row['invocation']['outputs'][field]);fields[field].append(a[ids])
   rows.append({'Re':re,'rank':rank,'method':method,'kind':'coefficient interpolation; NOT PDE solve' if method=='coefficient_PCHIP' else 'actual returned reduced PDE invocation','recorded_full_rule_scores':clean(row['scores']),'work':clean(invocation['work']),'raw_invocation_receipt':q['receipt'],'eligible_accuracy_recorded':row['eligible_accuracy'],'physical_admission':False})
  for field,value in fields.items():arrays[f'POD_r{rank}_{method}_{field}']=np.stack(value)
metadata={'schema':'ns-obstacle-actual-pod-recovery-plot-extension-v1','base_plot_archive_sha256':BASE_SHA,'audit_result_sha256':AUDIT_SHA,'indices_sha256':INDICES_SHA,'same_original_indices':True,'units':{'velocity':'u/U','pressure':'p/U^2','U':.2,'D':.1},'cases':rows,'retained_output_sha256':provenance,'original_representation_gate_passed':False,'physical_admission':False,'new_solves_or_reconstruction':False,'validation_accessed':False,'global_metrics_from_full_rule_not_display_subset':True,'elapsed_seconds':time.monotonic()-start}
arrays['metadata_json']=np.frombuffer(json.dumps(metadata,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8);buf=io.BytesIO();np.savez_compressed(buf,**arrays);payload=buf.getvalue();assert len(payload)<=8*2**20;signal.alarm(0);sys.stdout.buffer.write(payload)
