import hashlib,json,pathlib,time,resource,numpy as np
start=time.monotonic()
B=pathlib.Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027'); P=B/'neural-diagnostic01/results/diagnostic'; Q=B/'fine-preparation01/results/preparation/prepared'; D=B/'development-materialize02/results/population'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance={}
def checked(p,h=None):
 a=sha(p)
 if h is not None: assert a==h,(str(p),a,h)
 provenance[str(p)]=a
 return a
def load(n,receipt=None):
 p=P/(n+'.json')
 if receipt is not None: assert receipt['record_file']==p.name
 checked(p,None if receipt is None else receipt['record_sha256']); j=json.loads(p.read_text())
 if receipt is not None: assert j['array_file']==receipt['array_file'] and j['array_sha256']==receipt['array_sha256']
 checked(P/j['array_file'],j['array_sha256']); z=np.load(P/j['array_file'],allow_pickle=False)
 def dec(x):
  if isinstance(x,dict) and 'array' in x:
   a=z[x['array']]; assert list(a.shape)==x['shape'] and str(a.dtype)==x['dtype'] and np.isfinite(a).all()
   h=hashlib.sha256(str((a.dtype.str,a.shape)).encode()); h.update(np.ascontiguousarray(a).tobytes()); assert h.hexdigest()==x['content_sha256']; return a
  if isinstance(x,dict): return {k:dec(v) for k,v in x.items()}
  if isinstance(x,list): return [dec(v) for v in x]
  return x
 return dec(j['record'])
checked(P/'DIAGNOSTIC.json','94cd246c362cce71c41a8427d4fb5595014d067d087ee90027bbe420c6f1262c'); top=json.loads((P/'DIAGNOSTIC.json').read_text()); checked(P/'diagnostic_status_005.json','b5ca43bf0929f2fe876615d354e284f52540711d1b69cdb4b5b1d64d722fb8b0')
s=load('diagnostic_status_005',top['latest_evidence']); m=load('bank_r8_diagnostic_model',s['model']); o=m['operator']; f=m['force']
checked(Q/'manifest.json','7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6'); ma=json.loads((Q/'manifest.json').read_text())
for n in ['training.npz','boundary.npz']: checked(Q/n,ma['coefficient_files'][n])
w=np.load(Q/'training.npz')['weights_dimensionless']; bd=np.load(Q/'boundary.npz'); checked(D/'development.json','7b206ba1ed7ce10a4a76ab17a035d13fe3da2fc4f5f5334966b4faa7e7934070'); pop=json.loads((D/'development.json').read_text()); rows=[]
assert s['bank_gate']['passed'] is False and s['physical_admission'] is False
assert len(s['cases'])==len(pop['cases'])==len(s['bank_gate']['rows'])==2
for c,ref,floor in zip(s['cases'],pop['cases'],s['bank_gate']['rows']):
 re=c['Re']; assert re==ref['Re']==floor['Re']; checked(D/ref['array_file'],ref['array_sha256']); r=np.load(D/ref['array_file']); v=load('bank_r8_'+str(re)+'_raw_invocation',c['raw_invocation']); a=load('bank_r8_'+str(re)+'_diagnostic_arrays',c['diagnostic_arrays']); z=v['evidence']['state']; nu=1/re; x=np.r_[1,z[:16]]; u=a['full']['velocity']; p=a['full']['pressure']; g=a['full']['velocity_gradient']; assert np.array_equal(u,v['outputs']['velocity']) and np.array_equal(p,v['outputs']['pressure'])
 norm=lambda t:np.sqrt(w@np.sum(t.reshape(len(w),-1)**2,axis=1))
 vals={'velocity_relative':norm(u-r['velocity'])/norm(r['velocity']),'pressure_scaled':norm(p-r['pressure'])/np.sqrt(w.sum()),'divergence':norm(np.trace(g,axis1=1,axis2=2))/np.sqrt(w.sum()),'velocity_gradient_scaled':norm(g-r['velocity_gradient'])/np.sqrt(w.sum())}
 labels=bd['labels']; bu=a['boundary']['velocity']; flux=bd['weights_dimensionless']*np.einsum('ni,ni->n',bu,bd['normals_fluid']); qin=-flux[labels=='inlet'].sum(); qout=flux[labels=='outlet'].sum(); vals['mass']=abs(qout-qin)/abs(qin)
 target=np.zeros_like(bu); y=bd['points_physical'][:,1]; sel=labels=='inlet'; target[sel,0]=6*y[sel]*(.41-y[sel])/.41**2; vals['boundary_trace']=np.linalg.norm((bu-target)[labels!='outlet'],axis=1).max()
 force=nu*(f['lifting_per_nu']+f['velocity_per_nu']@z[:16])+f['pressure']@z[16:]; fgap=float(abs(force-a['force_coefficient']).max()); assert fgap<1e-13
 vals['drag_relative']=abs(force[0]-ref['force_coefficient'][0])/abs(ref['force_coefficient'][0]); vals['transverse_absolute']=abs(force[1]-ref['force_coefficient'][1])
 for k,value in vals.items(): assert np.isclose(value,c['scores'][k],rtol=2e-12,atol=2e-14),(k,value,c['scores'][k])
 def res(t):
  x=np.r_[1,t[:16]]; return np.r_[np.linalg.solve(o['lv'],np.einsum('mab,a,b->m',o['convection'],x,x)+nu*o['viscous']@x+o['pressure']@t[16:]+nu*o['pressure_lift']),np.linalg.solve(o['lq'],o['continuity']@x)]
 rr=res(z); assert np.allclose(rr,a['tensor'],rtol=1e-13,atol=1e-14)
 J=np.r_[np.linalg.solve(o['lv'],np.c_[np.einsum('mab,b->ma',o['convection']+o['convection'].transpose(0,2,1),x)[:,1:]+nu*o['viscous'][:,1:],o['pressure']]),np.linalg.solve(o['lq'],np.c_[o['continuity'][:,1:],np.zeros((16,8))])]
 jd=J@np.ones(24); fd=[float(np.linalg.norm((res(z+h)-res(z-h))/(2*h)-jd)/np.linalg.norm(jd)) for h in [1e-4,1e-5,1e-6]]; grad=float(np.linalg.norm((J@o['unknown_map']).T@rr)); assert grad<1e-9 and max(fd)<1e-8
 assert np.isclose(.5*rr@rr,v['evidence']['history'][-1]['objective'],rtol=1e-12); assert np.isclose(grad,v['evidence']['history'][-1]['gradient_norm'],rtol=1e-5)
 gap=float(np.linalg.norm(a['direct']-a['tensor'])/sum(np.linalg.norm(t) for t in a['terms'].values())); assert np.isclose(gap,c['full_direct_tensor_relative_gap'],rtol=1e-12)
 assert vals['velocity_relative']>=floor['velocity_relative']; assert c['eligible_accuracy'] is False and c['physical_admission'] is False
 rows.append(dict(Re=re,metrics=vals,force=force.tolist(),force_replay_maxgap=fgap,direct_tensor_gap=gap,finite_difference_errors=fd,stationarity_gradient=grad,residual_norm=float(np.linalg.norm(rr)),projection_floor=floor['velocity_relative'],floor_consistent=True))
print(json.dumps(dict(schema='ns-obstacle-neural-diagnostic-saved-array-audit-v1',passed=True,physical_admission=False,pressure_metric='scaled RMS error divided by U^2; not pressure-relative error',cases=rows,provenance=provenance,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=['No native FE assembly or new solve.','Saved direct arrays compared with independently replayed saved tensor.','Force contraction replay, not fresh boundary traction assembly.','No original full-operator fingerprint equality claim.']),indent=2))
