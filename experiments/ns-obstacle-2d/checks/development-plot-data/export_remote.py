"""Read-only deterministic plot subset; bank projection is NOT an online solve."""
from pathlib import Path
import sys,json,hashlib,io,time,signal,resource
import numpy as np
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('plot export90s')));signal.alarm(90)
ns=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027');b=ns/'model-development01';out=b/'results/development-stage/development';vol=ns/'fine-preparation01/results/preparation/prepared/volume';dev=ns/'development-materialize02/results/population'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
provenance={};expected={r['path']:r for r in EXPECTED['terminal']['outputs']}
def verified(path):
 rel=str(path.relative_to(b));assert sha(path)==expected[rel]['sha256'];provenance[str(path)]=sha(path)
def decode(x,z):
 if isinstance(x,dict):
  if 'array' in x:
   a=z[x['array']];assert list(a.shape)==x['shape'] and str(a.dtype)==x['dtype'];h=hashlib.sha256(str((a.dtype.str,a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());assert h.hexdigest()==x['content_sha256'];return a
  return {k:decode(v,z) for k,v in x.items()}
 if isinstance(x,list):return [decode(v,z) for v in x]
 return x
def load(name):
 p=out/(name+'.json');verified(p);d=read(p);a=out/d['array_file'];verified(a)
 with np.load(a,allow_pickle=False) as z:return decode(d['record'],z)
geom=vol/'geometry.npz';assert sha(geom)==EXPECTED['geometry_sha256'];provenance[str(geom)]=sha(geom)
with np.load(geom,allow_pickle=False) as z:
 n=len(z['weights_dimensionless']);ids=(np.arange(18000,dtype=np.int64)*(n-1))//17999;points=z['points_physical'][ids];weights=z['weights_dimensionless'][ids]
assert len(np.unique(ids))==18000
arrays={'points_physical':points,'indices':ids,'Re':np.array([16.875,23.125]),'weights_dimensionless_subset_not_for_global_metrics':weights};meta={'schema':'ns-obstacle-development-plot-subset-v1','original_point_count':n,'plot_point_count':len(ids),'selection':'indices[i]=floor(i*(N-1)/(18000-1)),i=0..17999; same original point enumeration for every case/method','units':{'velocity':'u/U','pressure':'p/U^2','points':'physical coordinates','U':.2,'D':.1},'geometry':{'xmin':0.,'xmax':2.2,'H':.41,'center':[.2,.2],'radius':.05},'recorded_full_rule_metrics':{},'limitations':['plotting subset is not quadrature or metric certification','bank_r8_projection is saved best-fit enriched representation, NOT online PDE solve','PODr2 methods are actual saved invocations; all fail registered force accuracy','PODr4/r8 online results incomplete at original deadline; not shown'],'new_solves':False,'validation_accessed':False}
manifest=read(dev/'development.json');assert sha(dev/'development.json')=='7b206ba1ed7ce10a4a76ab17a035d13fe3da2fc4f5f5334966b4faa7e7934070';provenance[str(dev/'development.json')]=sha(dev/'development.json')
ru=[];rp=[]
for row in manifest['cases']:
 f=dev/row['array_file'];assert sha(f)==row['array_sha256'];provenance[str(f)]=sha(f)
 with np.load(f,allow_pickle=False) as z:ru.append(z['velocity'][ids]);rp.append(z['pressure'][ids])
 meta['recorded_full_rule_metrics'][str(row['Re'])]={'reference_force_coefficient':row['force_coefficient']}
arrays['reference_velocity']=np.stack(ru);arrays['reference_pressure']=np.stack(rp)
method_arrays={k:{'velocity':[],'pressure':[]} for k in ('mixed_Galerkin','exact_tensor_weak_LS','coefficient_PCHIP')}
for re in (16.875,23.125):
 record=load('POD_r2_development_'+str(re));assert record['Re']==re
 for method,fields in method_arrays.items():
  r=record['methods'][method]
  for k in fields:fields[k].append(r['invocation']['outputs'][k][ids])
  meta['recorded_full_rule_metrics'][str(re)]['POD_r2_'+method]={'scores':clean(r['scores']),'status':r['invocation']['work']['status'],'eligible_accuracy':r['eligible_accuracy']}
for method,fields in method_arrays.items():
 for k,values in fields.items():arrays['POD_r2_'+method+'_'+k]=np.stack(values)
# Read only the four existing prepared field values needed for saved projection;
# never instantiate native primary replay or assemble an operator.
vm=read(vol/'manifest.json');assert sha(vol/'manifest.json')==VOLUME_MANIFEST_SHA;provenance[str(vol/'manifest.json')]=sha(vol/'manifest.json');subset={};block_provenance={}
for name in ('lift_u','lift_p_per_nu','supremizers','pressure'):
 pieces=[]
 for row in vm['fields'][name]['blocks']:
  chosen=ids[(ids>=row['start'])&(ids<row['end'])]-row['start']
  if not len(chosen):continue
  f=vol/row['file'];assert sha(f)==row['sha256'];block_provenance[row['file']]=row['sha256']
  with np.load(f,allow_pickle=False) as z:pieces.append(z['values'][chosen])
 subset[name]=np.concatenate(pieces);assert len(subset[name])==len(ids)
model=load('bank_r8_model');sys.path.insert(0,str(b));from bank import FrozenBank
bankfile=ns/'bank-fit01/results/fit/bank.npz';assert sha(bankfile)=='da32607494d7ddf8c1d7c231be7289488ff8cfb3a071a2ec58b369a9d936eacd';provenance[str(bankfile)]=sha(bankfile);bank=FrozenBank.load(bankfile);base,_=bank.evaluate(points,8);basis=np.concatenate((base,subset['supremizers'][...,:8]),axis=-1);bu=[];bp=[]
for row in model['development_projection']:
 state=np.asarray(row['achieved_state'],dtype=np.float64);assert state.shape==(24,)
 bu.append(subset['lift_u'][...,0]+np.einsum('niv,v->ni',basis,state[:16]));bp.append(subset['lift_p_per_nu'][...,0]/row['Re']+np.einsum('np,p->n',subset['pressure'][...,:8],state[16:]))
 meta['recorded_full_rule_metrics'][str(row['Re'])]['bank_r8_best_fit_projection']=clean(row)
arrays['bank_r8_projection_velocity']=np.stack(bu);arrays['bank_r8_projection_pressure']=np.stack(bp)
for a in arrays.values():assert np.isfinite(a).all()
meta.update(source_sha256={n:sha(b/n) for n in ('bank.py','geometry.py','pilot2d.py','pilot2d_blocked.py')},input_sha256=provenance,prepared_value_block_sha256=block_provenance,bank_projection_reconstruction='lift_u+concat(bank.evaluate(points,8),supremizers[:8])@saved_state[:16]; p=lift_p_per_nu/Re+pressure[:8]@saved_state[16:]',indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
arrays['metadata_json']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);payload=buffer.getvalue();assert len(payload)<=5000000,len(payload);signal.alarm(0);sys.stdout.buffer.write(payload)
