import hashlib,json,pathlib,time,resource,numpy as np
start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
# BINDING is injected by a separately pinned collector only after actual completion.
assert set(BINDING)=={'status_sha256','latest_record_sha256','release_sha256','release_file','model_sha256','runtime_relative','runtime_sha256'}
B=pathlib.Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027'); P=B/'neural-pod-guided-diagnostic03/results/diagnostic'; Q=B/'fine-preparation01/results/preparation/prepared'; D=B/'development-materialize02/results/population'
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
checked(P/'DIAGNOSTIC.json',BINDING['status_sha256']);top=json.loads((P/'DIAGNOSTIC.json').read_text());assert top['status']=='completed_diagnostic_pending_independent_review' and top['completed_case_records']==2
assert top['latest_evidence']['record_sha256']==BINDING['latest_record_sha256']
s=load(pathlib.Path(top['latest_evidence']['record_file']).stem,top['latest_evidence']);m=load('pod_guided_neural_r4_model',s['model']);o=m['operator'];f=m['force'];nv=o['viscous'].shape[1]-1;npres=o['pressure'].shape[1];assert (nv,npres)==(8,4)
assert s['model_sha256']==BINDING['model_sha256'] and s['original_bank_gate_passed'] is False and s['new_model_gate_not_yet_admitted'] is True
assert s['validation_seen'] is False and s['reference_fit_initialization_used'] is False and s['training_or_quadrature_fit_performed'] is False
job=P.parents[1];assert pathlib.Path(BINDING['release_file']).name==BINDING['release_file'];checked(job/BINDING['release_file'],BINDING['release_sha256'])
assert s['bank_evaluator_class']=='accuracy_v2.fast_spatial_bank.FastSpatialBank'
runtime_path=pathlib.Path(BINDING['runtime_relative']);assert not runtime_path.is_absolute() and '..' not in runtime_path.parts;checked(job/runtime_path,BINDING['runtime_sha256']);runtime=json.loads((job/runtime_path).read_text());assert runtime['jax_backend']=='gpu' and runtime['jax_x64'] is True and runtime['matmul_precision']=='highest'
release=json.loads((job/BINDING['release_file']).read_text());assert release['authorized'] and release['single_execution'] and release['validation_allowed'] is False
assert release['numpy_derivative_contraction']=='optimized_einsum';checked(job/'accuracy_v2/fast_spatial_bank.py',release['fast_evaluator_sha256']);checked(job/'accuracy_v2/diagnostic02_launch.py',release['launcher_sha256'])
checked(job/'PLAN.json',release['plan_sha256']);plan=json.loads((job/'PLAN.json').read_text());checked(job/'RUNTIME-INVENTORY.json',release['runtime_inventory_sha256']);inventory=json.loads((job/'RUNTIME-INVENTORY.json').read_text())
for item in inventory['files']:
 rel=pathlib.Path(item['destination']);assert not rel.is_absolute() and '..' not in rel.parts;path=job/rel;assert path.stat().st_size==item['bytes'];checked(path,item['sha256'])
assert plan['inputs']['bank']['sha256']==s['model_sha256'];checked(pathlib.Path(plan['inputs']['bank']['path']),s['model_sha256'])
for name in ('model_receipt','model_review'):
 desc=plan['inputs'][name];checked(pathlib.Path(desc['path']),desc['sha256'])
fingerprint={'training_coefficients':m['training_coefficients']};fingerprint.update({'operator_'+k:v for k,v in o.items() if isinstance(v,np.ndarray)});fingerprint.update({'force_'+k:v for k,v in f.items() if isinstance(v,np.ndarray)});h=hashlib.sha256()
for name in sorted(fingerprint):
 value=np.ascontiguousarray(fingerprint[name]);h.update(name.encode());h.update(str((value.shape,str(value.dtype))).encode());h.update(value.tobytes())
assert h.hexdigest()==m['fresh_fingerprint'] and m['stability']['passed']
checked(Q/'manifest.json','7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6'); ma=json.loads((Q/'manifest.json').read_text())
for n in ['training.npz','boundary.npz']: checked(Q/n,ma['coefficient_files'][n])
w=np.load(Q/'training.npz')['weights_dimensionless']; bd=np.load(Q/'boundary.npz'); checked(D/'development.json','7b206ba1ed7ce10a4a76ab17a035d13fe3da2fc4f5f5334966b4faa7e7934070'); pop=json.loads((D/'development.json').read_text()); rows=[]
assert s['physical_admission'] is False and len(s['cases'])==len(pop['cases'])==2
assert [c['Re'] for c in s['cases']]==[16.875,23.125]
for c,ref in zip(s['cases'],pop['cases']):
 floor=c['representation_diagnostic']
 re=c['Re']; assert re==ref['Re']==floor['Re']; checked(D/ref['array_file'],ref['array_sha256']); r=np.load(D/ref['array_file']); v=load('pod_guided_neural_r4_'+str(re)+'_raw_invocation',c['raw_invocation']); a=load('pod_guided_neural_r4_'+str(re)+'_diagnostic_arrays',c['diagnostic_arrays']); z=v['evidence']['state']; nu=1/re; x=np.r_[1,z[:nv]]; u=a['full']['velocity']; p=a['full']['pressure']; g=a['full']['velocity_gradient']; assert np.array_equal(u,v['outputs']['velocity']) and np.array_equal(p,v['outputs']['pressure'])
 norm=lambda t:np.sqrt(w@np.sum(t.reshape(len(w),-1)**2,axis=1))
 vals={'velocity_relative':norm(u-r['velocity'])/norm(r['velocity']),'pressure_scaled':norm(p-r['pressure'])/np.sqrt(w.sum()),'divergence':norm(np.trace(g,axis1=1,axis2=2))/np.sqrt(w.sum()),'velocity_gradient_scaled':norm(g-r['velocity_gradient'])/np.sqrt(w.sum())}
 labels=bd['labels']; bu=a['boundary']['velocity']; flux=bd['weights_dimensionless']*np.einsum('ni,ni->n',bu,bd['normals_fluid']); qin=-flux[labels=='inlet'].sum(); qout=flux[labels=='outlet'].sum(); vals['mass']=abs(qout-qin)/abs(qin)
 target=np.zeros_like(bu); y=bd['points_physical'][:,1]; sel=labels=='inlet'; target[sel,0]=6*y[sel]*(.41-y[sel])/.41**2; vals['boundary_trace']=np.linalg.norm((bu-target)[labels!='outlet'],axis=1).max()
 force=nu*(f['lifting_per_nu']+f['velocity_per_nu']@z[:nv])+f['pressure']@z[nv:]; fgap=float(abs(force-a['force_coefficient']).max()); assert fgap<1e-13
 vals['drag_relative']=abs(force[0]-ref['force_coefficient'][0])/abs(ref['force_coefficient'][0]); vals['transverse_absolute']=abs(force[1]-ref['force_coefficient'][1])
 for k,value in vals.items(): assert np.isclose(value,c['scores'][k],rtol=2e-12,atol=2e-14),(k,value,c['scores'][k])
 def res(t):
  x=np.r_[1,t[:nv]]; return np.r_[np.linalg.solve(o['lv'],np.einsum('mab,a,b->m',o['convection'],x,x)+nu*o['viscous']@x+o['pressure']@t[nv:]+nu*o['pressure_lift']),np.linalg.solve(o['lq'],o['continuity']@x)]
 raw_terms={'advection':np.einsum('mab,a,b->m',o['convection'],x,x),'viscous':nu*o['viscous']@x,'pressure':o['pressure']@z[nv:],'pressure_lifting':nu*o['pressure_lift'],'continuity':o['continuity']@x}
 for key,value in raw_terms.items():assert np.allclose(value,a['terms'][key],rtol=1e-12,atol=1e-14)
 raw=np.r_[sum(raw_terms[k] for k in ('advection','viscous','pressure','pressure_lifting')),raw_terms['continuity']];assert np.allclose(raw,a['tensor'],rtol=1e-12,atol=1e-14);rr=res(z)
 J=np.r_[np.linalg.solve(o['lv'],np.c_[np.einsum('mab,b->ma',o['convection']+o['convection'].transpose(0,2,1),x)[:,1:]+nu*o['viscous'][:,1:],o['pressure']]),np.linalg.solve(o['lq'],np.c_[o['continuity'][:,1:],np.zeros((o['continuity'].shape[0],npres))])]
 jd=J@np.ones(nv+npres); fd=[float(np.linalg.norm((res(z+h)-res(z-h))/(2*h)-jd)/np.linalg.norm(jd)) for h in [1e-4,1e-5,1e-6]]; grad=float(np.linalg.norm((J@o['unknown_map']).T@rr)); denom=max(v['evidence']['history'][0]['gradient_norm'],1.);stationarity_passed=bool(grad<=1e-9*denom)
 if v['work']['status']=='stationary':assert stationarity_passed
 assert max(fd)<1e-8
 assert np.isclose(denom,v['evidence']['history'][-1]['gradient_denominator'],rtol=1e-12)
 assert np.isclose(.5*rr@rr,v['evidence']['history'][-1]['objective'],rtol=1e-12); assert np.isclose(grad,v['evidence']['history'][-1]['gradient_norm'],rtol=1e-5)
 gap_denominator=max(sum(np.linalg.norm(t) for t in a['terms'].values()),1e-12);assert np.isclose(gap_denominator,c['full_direct_tensor_denominator'],rtol=1e-12)
 gap=float(np.linalg.norm(a['direct']-a['tensor'])/gap_denominator); assert np.isclose(gap,c['full_direct_tensor_relative_gap'],rtol=1e-12)
 assert vals['velocity_relative']+1e-12>=floor['velocity_relative']; assert c['eligible_accuracy'] is False and c['physical_admission'] is False
 checks={name:bool(vals[metric]<=limit) for name,metric,limit in [('velocity','velocity_relative',.01),('pressure','pressure_scaled',.02),('drag','drag_relative',.02),('transverse','transverse_absolute',.002),('divergence','divergence',.005),('mass','mass',.001),('boundary_trace','boundary_trace',1e-6)]}
 assert checks==c['scores']['target_checks'] and all(checks.values())==c['scores']['accuracy_targets_passed']
 assert c['diagnostic_numerical_targets_passed']==bool(v['work']['status'] in ('stationary','root_converged') and all(checks.values()) and c['jacobian_check']['passed'] and gap<=1e-10)
 rows.append(dict(Re=re,metrics=vals,target_checks=checks,diagnostic_targets_passed=c['diagnostic_numerical_targets_passed'],force=force.tolist(),force_replay_maxgap=fgap,direct_tensor_gap=gap,finite_difference_errors=fd,stationarity_gradient=grad,stationarity_denominator=denom,stationarity_passed=stationarity_passed,solver_status=v['work']['status'],residual_norm=float(np.linalg.norm(rr)),projection_floor=floor['velocity_relative'],floor_consistent=True))
print(json.dumps(dict(schema='ns-obstacle-neural-refined-diagnostic-saved-array-audit-v1',passed=True,physical_admission=False,model_sha256=s['model_sha256'],bank_evaluator_class=s['bank_evaluator_class'],runtime_record=runtime,pressure_metric='scaled RMS error divided by U^2; not pressure-relative error',cases=rows,provenance=provenance,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=['No native FE assembly or new solve.','Saved direct arrays compared with independently replayed saved tensor.','Force contraction replay, not fresh boundary traction assembly.','Fresh saved fingerprint checked; no equality to a prior neural model operator is claimed.','Representation floor is a recorded achieved same-space projection consistency check, not continuum truth accuracy.','Saved neural gradients compared to references; no fresh network derivative or FE assembly.']),indent=2))
