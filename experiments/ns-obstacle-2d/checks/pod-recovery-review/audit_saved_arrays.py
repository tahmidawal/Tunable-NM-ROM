"""Read-only saved-array audit. No native imports, solve, or model evaluation."""
import hashlib,json,pathlib,time,resource,numpy as np
start=time.monotonic(); resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
B=pathlib.Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027');P=B/'pod-recovery01/results/recovery';Q=B/'fine-preparation01/results/preparation/prepared';D=B/'development-materialize02/results/population'
provenance={}
def checked(p,h=None):
 with p.open('rb') as f:a=hashlib.file_digest(f,'sha256').hexdigest()
 if h is not None:assert a==h,(str(p),a,h)
 provenance[str(p)]=a;return a
def load(n,receipt=None):
 p=P/(n+'.json');checked(p,None if receipt is None else receipt['record_sha256']);j=json.loads(p.read_text())
 if receipt:assert receipt['record_file']==p.name and receipt['array_file']==j['array_file'] and receipt['array_sha256']==j['array_sha256']
 checked(P/j['array_file'],j['array_sha256'])
 with np.load(P/j['array_file'],allow_pickle=False) as z:
  def dec(x):
   if isinstance(x,dict) and 'array' in x:
    a=z[x['array']];assert list(a.shape)==x['shape'] and str(a.dtype)==x['dtype'] and np.isfinite(a).all();h=hashlib.sha256(str((a.dtype.str,a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());assert h.hexdigest()==x['content_sha256'];return a
   if isinstance(x,dict):return {k:dec(v) for k,v in x.items()}
   if isinstance(x,list):return [dec(v) for v in x]
   return x
  return dec(j['record'])
def same(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
def npz(p,h):
 checked(p,h)
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
checked(P/'RECOVERY.json','fae1405eaeebe284d9e6bcc716fa1266f2a9718cfd20b1819786b773e331d785');s=json.loads((P/'RECOVERY.json').read_text())
planpath=B/'pod-recovery01/RECOVERY-PLAN.json';checked(planpath,'6cd3fec19d668a7aec5adfeeba2a2e8abe5bb466601c3fec17ad4f8e4b81dd7d');plan=json.loads(planpath.read_text())
assert len(plan['original_outputs'])==20 and not s['validation_seen'] and not s['quadrature_refitted'] and not s['bank_gate']['passed']
for x in plan['original_outputs']:
 p=P/x['name'];assert p.stat().st_size==x['bytes'];checked(p,x['sha256'])
frozen=load('frozen_models',s['frozen_models'])
assert set(frozen['models'])=={'POD_r2','POD_r4','POD_r8'} and not frozen['bank_gate']['passed']
assert all(x['selected_support'] is None and not x['validation_seen'] for x in frozen['choices'].values())
ops={}
for rank in [2,4,8]:
 name=f'POD_r{rank}';a=load(name+'_recovery_operator',s['models'][name]['operator_evidence']);old=load(name+'_model');src=dict(a['source']);prior=dict(old['source']);src.pop('offline_build_seconds');prior.pop('offline_build_seconds');assert same(src,prior) and same(a['stability'],old['stability']);assert np.array_equal(a['fingerprint_arrays']['training_coefficients'],old['training_coefficients']);ops[rank]=a['fingerprint_arrays']
 h=hashlib.sha256()
 for k in sorted(ops[rank]):
  x=np.ascontiguousarray(ops[rank][k]);h.update(k.encode());h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes())
 assert h.hexdigest()==a['fresh_fingerprint']==s['models'][name]['fresh_fingerprint']==frozen['models'][name]['fingerprint']
checked(Q/'manifest.json','7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6');m=json.loads((Q/'manifest.json').read_text());checked(Q/'volume/manifest.json',m['coefficient_files']['volume/manifest.json']);vm=json.loads((Q/'volume/manifest.json').read_text())
geo=npz(Q/'volume/geometry.npz',vm['files']['geometry.npz']);w=geo['weights_dimensionless'];bd=npz(Q/'boundary.npz',m['coefficient_files']['boundary.npz'])
checked(D/'development.json','7b206ba1ed7ce10a4a76ab17a035d13fe3da2fc4f5f5334966b4faa7e7934070');pop=json.loads((D/'development.json').read_text());refs={}
for c in pop['cases']:refs[c['Re']]=(c,npz(D/c['array_file'],c['array_sha256']))
assert set(refs)=={16.875,23.125};cases={k:load(k,x['receipt']) for k,x in s['cases'].items() if not x['reused']};rows=[];work=[]
norm=lambda a:float(np.sqrt(w@np.sum(a.reshape(len(w),-1)**2,axis=1)))
expected={f'POD_r{r}_{re}_{method}_raw_invocation' for r in (4,8) for re in (16.875,23.125) for method in ('exact_tensor_weak_LS','coefficient_PCHIP','mixed_Galerkin')}
assert len(s['query_receipts'])==12 and {q['key'] for q in s['query_receipts']}==expected
for q in s['query_receipts']:
 assert not q['failed'];key=q['key'];_,r,re,method=key.split('_',3);rank=int(r[1:]);re=float(re);method=method.removesuffix('_raw_invocation');v=load(key,q['receipt']);c=cases[f'POD_r{rank}_development_{re}']['methods'][method];assert same(v,c['invocation']);z=v['evidence']['state'];assert z.shape==(3*rank,);u=v['outputs']['velocity'];p=v['outputs']['pressure'];ref,rf=refs[re];op=ops[rank];force=(op['force_lifting_per_nu']+op['force_velocity_per_nu']@z[:2*rank])/re+op['force_pressure']@z[2*rank:]
 x=np.r_[1,z[:2*rank]];o={k.removeprefix('operator_'):a for k,a in op.items() if k.startswith('operator_')};nu=1/re
 rr=np.r_[np.linalg.solve(o['lv'],np.einsum('mab,a,b->m',o['convection'],x,x)+nu*o['viscous']@x+o['pressure']@z[2*rank:]+nu*o['pressure_lift']),np.linalg.solve(o['lq'],o['continuity']@x)]
 J=np.r_[np.linalg.solve(o['lv'],np.c_[np.einsum('mab,b->ma',o['convection']+o['convection'].transpose(0,2,1),x)[:,1:]+nu*o['viscous'][:,1:],o['pressure']]),np.linalg.solve(o['lq'],np.c_[o['continuity'][:,1:],np.zeros((o['continuity'].shape[0],rank))])]
 grad=float(np.linalg.norm((J@o['unknown_map']).T@rr));objective=float(.5*rr@rr)
 if method=='exact_tensor_weak_LS':
  hist=v['evidence']['history'];assert np.isclose(objective,hist[-1]['objective'],rtol=1e-10,atol=1e-20);assert np.isclose(grad,hist[-1]['gradient_norm'],rtol=1e-4,atol=1e-14);assert grad<=1e-9*max(hist[0]['gradient_norm'],1)
 vals={'velocity_relative':norm(u-rf['velocity'])/norm(rf['velocity']),'pressure_scaled':norm(p-rf['pressure'])/np.sqrt(w.sum()),'drag_relative':abs(force[0]-ref['force_coefficient'][0])/abs(ref['force_coefficient'][0]),'transverse_absolute':abs(force[1]-ref['force_coefficient'][1])}
 bu=bd['lift_u'][...,0]+np.concatenate([bd['pod_velocity'][...,:rank],bd['supremizers'][...,:rank]],axis=-1)@z[:2*rank];labels=bd['labels'];flux=bd['weights_dimensionless']*np.einsum('ni,ni->n',bu,bd['normals_fluid']);qin=-flux[labels=='inlet'].sum();qout=flux[labels=='outlet'].sum();vals['mass']=abs(qout-qin)/abs(qin);target=np.zeros_like(bu);y=bd['points_physical'][:,1];sel=labels=='inlet';target[sel,0]=6*y[sel]*(.41-y[sel])/.41**2;vals['boundary_trace']=np.linalg.norm((bu-target)[labels!='outlet'],axis=1).max()
 row={'key':key,'Re':re,'rank':rank,'method':method,'metrics':vals,'force':force.tolist(),'eligible_accuracy_recorded':c['eligible_accuracy'],'physical_admission':False,'full_weak_LS_objective':objective,'full_weak_LS_gradient_norm':grad};rows.append(row);work.append((z,u,rf,c,row));row['_div2']=0.;row['_grad2']=0.;row['_u_gap']=0.
assert len(rows)==12
for i,b in enumerate(vm['fields']['lift_u']['blocks']):
 lo,hi=b['start'],b['end'];arrays={}
 for name in ('lift_u','pod_velocity','supremizers'):
  block=vm['fields'][name]['blocks'][i];assert (block['start'],block['end'])==(lo,hi);filename='volume/'+block['file'];assert m['coefficient_files'][filename]==block['sha256']==vm['files'][block['file']];arrays[name]=npz(Q/filename,block['sha256'])
 for z,u,rf,c,row in work:
  r=row['rank'];uv=arrays['lift_u']['values'][...,0]+np.concatenate([arrays['pod_velocity']['values'][...,:r],arrays['supremizers']['values'][...,:r]],axis=-1)@z[:2*r];g=arrays['lift_u']['gradients'][...,0]+np.concatenate([arrays['pod_velocity']['gradients'][...,:r],arrays['supremizers']['gradients'][...,:r]],axis=-1)@z[:2*r];row['_u_gap']=max(row['_u_gap'],float(abs(uv-u[lo:hi]).max()));row['_div2']+=float(w[lo:hi]@np.trace(g,axis1=1,axis2=2)**2);row['_grad2']+=float(w[lo:hi]@np.sum((g-rf['velocity_gradient'][lo:hi])**2,axis=(1,2)))
 assert time.monotonic()-start<115,'bounded audit exceeded115s; no automatic retry'
for z,u,rf,c,row in work:
 row['metrics']['divergence']=np.sqrt(row.pop('_div2')/w.sum());row['metrics']['velocity_gradient_scaled']=np.sqrt(row.pop('_grad2')/w.sum());row['saved_velocity_reconstruction_maxgap']=row.pop('_u_gap');assert row['saved_velocity_reconstruction_maxgap']<1e-12
 checks={name:bool(row['metrics'][metric]<=limit) for name,metric,limit in [('velocity','velocity_relative',.01),('pressure','pressure_scaled',.02),('drag','drag_relative',.02),('transverse','transverse_absolute',.002),('divergence','divergence',.005),('mass','mass',.001),('boundary_trace','boundary_trace',1e-6)]}
 assert checks==c['scores']['target_checks'] and all(checks.values())==c['eligible_accuracy'];row['replayed_target_checks']=checks
 for k,value in row['metrics'].items():assert np.isclose(value,c['scores'][k],rtol=2e-10,atol=2e-13),(row['key'],k,value,c['scores'][k])
print(json.dumps({'schema':'ns-obstacle-pod-recovery-saved-array-audit-v1','passed':True,'physical_admission':False,'cases':rows,'provenance':provenance,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'limitations':['No native FE assembly, model evaluation, solve, optimization or validation access.','Mixed Galerkin operator itself not separately retained; its returned fields independently scored.','Original full-operator fingerprints unavailable; only available original source/stability/training coefficient bindings compared.','Original r2 failures and failed quadrature remain unchanged.','Force replay uses saved contractions, not fresh traction assembly.','Direct/tensor/Jacobian diagnostic arrays were not retained; recorded gap scalars are not independently replayed.']},indent=2,allow_nan=False))
