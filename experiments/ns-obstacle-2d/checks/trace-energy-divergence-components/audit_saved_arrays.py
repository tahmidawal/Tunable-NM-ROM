import hashlib,json,pathlib,time,resource,numpy as np
start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
# BINDING is injected by a separately pinned collector only after actual completion.
assert set(BINDING)=={'diagnostics'} and [d['variant'] for d in BINDING['diagnostics']]==['trace','energy']
B=pathlib.Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027'); Q=B/'fine-preparation01/results/preparation/prepared'; D=B/'development-materialize02/results/population'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance={}
def checked(p,h=None):
 a=sha(p)
 if h is not None: assert a==h,(str(p),a,h)
 if '/neural-pod-guided-diagnostic' in str(p):assert str(p) in current['accepted_provenance'] and a==current['accepted_provenance'][str(p)]
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
checked(Q/'manifest.json','7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6');ma=json.loads((Q/'manifest.json').read_text())
def arrays(path,digest):
 checked(path,digest)
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
checked(Q/'volume/manifest.json',ma['coefficient_files']['volume/manifest.json']);vm=json.loads((Q/'volume/manifest.json').read_text())
assert ma['scales']['D']==.1 and ma['scales']['U']==.2 and ma['split']=='training'
assert vm['units']=={'values':'dimensionless','gradients':'d/d(x/D)','weights':'dx/D**2'}
assert vm['files']['geometry.npz']==ma['coefficient_files']['volume/geometry.npz']
geo=arrays(Q/'volume/geometry.npz',vm['files']['geometry.npz']);w=geo['weights_dimensionless'];assert np.isfinite(w).all() and (w>0).all()
rule_hash=hashlib.sha256()
for name in ('points_dimensionless','weights_dimensionless'):
 value=np.ascontiguousarray(geo[name]);rule_hash.update(name.encode());rule_hash.update(str((value.shape,str(value.dtype))).encode());rule_hash.update(value.tobytes())
assert rule_hash.hexdigest()=='1b8cb44f65c64b3e2cbfeefc5848bc3f3dc06c2b30b8123797bb6fe5c74b4590'
bd=arrays(Q/'boundary.npz',ma['coefficient_files']['boundary.npz']);rows=[];states=[]
for current in BINDING['diagnostics']:
 P=B/current['job_namespace']/'results/diagnostic'
 assert current['job_namespace']=={'trace':'neural-pod-guided-diagnostic03','energy':'neural-pod-guided-diagnostic04'}[current['variant']]
 checked(P/'DIAGNOSTIC.json',current['status_sha256']);top=json.loads((P/'DIAGNOSTIC.json').read_text());assert top['status']=='completed_diagnostic_pending_independent_review' and top['completed_case_records']==2
 assert top['latest_evidence']['record_sha256']==current['latest_record_sha256']
 s=load(pathlib.Path(top['latest_evidence']['record_file']).stem,top['latest_evidence']);m=load('pod_guided_neural_r4_model',s['model']);o=m['operator'];f=m['force'];nv=o['viscous'].shape[1]-1;npres=o['pressure'].shape[1];assert (nv,npres)==(8,4)
 assert s['model_sha256']==current['model_sha256'] and s['original_bank_gate_passed'] is False and s['new_model_gate_not_yet_admitted'] is True
 assert s['validation_seen'] is False and s['reference_fit_initialization_used'] is False and s['training_or_quadrature_fit_performed'] is False
 
 checked(P.parents[1]/current['release_file'],current['release_sha256'])
 assert [c['Re'] for c in s['cases']]==[16.875,23.125]
 for c in s['cases']:
  re=c['Re'];v=load('pod_guided_neural_r4_'+str(re)+'_raw_invocation',c['raw_invocation']);a=load('pod_guided_neural_r4_'+str(re)+'_diagnostic_arrays',c['diagnostic_arrays']);z=v['evidence']['state'];assert z.shape==(12,)
  assert np.array_equal(a['full']['velocity'],v['outputs']['velocity']) and np.array_equal(a['full']['pressure'],v['outputs']['pressure'])
  total=a['full']['velocity_gradient'];assert total.shape==(len(w),2,2)
  lift=bd['lift_u'][...,0];sup=bd['supremizers'][...,:4]@z[4:8];neural=a['boundary']['velocity']-lift-sup
  parts=np.stack([lift,neural,sup]);flux=bd['weights_dimensionless'][None,:]*np.einsum('kni,ni->kn',parts,bd['normals_fluid'])
  labels=sorted(set(bd['labels'].tolist()));bylabel={lab:flux[:,bd['labels']==lab].sum(axis=1).tolist() for lab in labels}
  totalflux=flux.sum(axis=0);qin=-totalflux[bd['labels']=='inlet'].sum();qout=totalflux[bd['labels']=='outlet'].sum()
  assert np.isclose(abs(qout-qin)/abs(qin),c['scores']['mass'],rtol=2e-12,atol=2e-14)
  row=dict(variant=current['variant'],model_sha256=s['model_sha256'],accepted_audit_sha256=current['audit_sha256'],Re=re,state=z.tolist(),component_order=['FE_lift','inferred_neural','FE_supremizers'],boundary_outward_flux_by_label=bylabel,boundary_net_outward_flux=flux.sum(axis=1).tolist(),inlet_positive_inflow=(-np.array(bylabel['inlet'])).tolist(),outlet_positive_outflow=bylabel['outlet'],total_inlet_flux=float(qin),total_outlet_flux=float(qout),total_mass_metric=float(abs(qout-qin)/abs(qin)),recorded_divergence=c['scores']['divergence'],divergence_gram=np.zeros((3,3)),gradient_gram=np.zeros((3,3)),signed_divergence_integrals=np.zeros(3),block_components=[])
  rows.append(row);states.append((z,total))
nextpoint=0
for i,b in enumerate(vm['fields']['lift_u']['blocks']):
 lo,hi=b['start'],b['end'];assert lo==nextpoint;nextpoint=hi;data={}
 for name in ('lift_u','supremizers'):
  block=vm['fields'][name]['blocks'][i];assert (block['start'],block['end'])==(lo,hi);fn='volume/'+block['file'];assert ma['coefficient_files'][fn]==block['sha256']==vm['files'][block['file']];data[name]=arrays(Q/fn,block['sha256'])
 lift=data['lift_u']['gradients'][...,0]
 for row,(z,total) in zip(rows,states):
  sup=data['supremizers']['gradients'][...,:4]@z[4:8];neural=total[lo:hi]-lift-sup;g=np.stack([lift,neural,sup]);d=np.trace(g,axis1=2,axis2=3);wb=w[lo:hi]
  gram=np.einsum('n,an,bn->ab',wb,d,d);integrals=d@wb
  row['divergence_gram']+=gram;row['gradient_gram']+=np.einsum('n,anij,bnij->ab',wb,g,g);row['signed_divergence_integrals']+=integrals
  row['block_components'].append(dict(start=lo,end=hi,divergence_gram=gram.tolist(),signed_divergence_integrals=integrals.tolist()))
assert nextpoint==len(w) and len(rows)==4
for row in rows:
 gram=row['divergence_gram'];volume=float(w.sum());total=float(np.sqrt(max(gram.sum(),0)/volume));assert np.isclose(total,row['recorded_divergence'],rtol=2e-12,atol=2e-14)
 row['component_divergence_rms']=np.sqrt(np.maximum(np.diag(gram),0)/volume).tolist();row['signed_cross_contributions_to_squared_rms']=(2*np.triu(gram,1)/volume).tolist();row['reconstructed_total_divergence_rms']=total;row['component_gradient_rms']=np.sqrt(np.maximum(np.diag(row['gradient_gram']),0)/volume).tolist();row['volume']=volume
 row['volume_divergence_minus_boundary_flux']=(row['signed_divergence_integrals']-np.array(row['boundary_net_outward_flux'])).tolist()
 for k in ('divergence_gram','gradient_gram','signed_divergence_integrals'):row[k]=row[k].tolist()
print(json.dumps(dict(schema='ns-obstacle-neural-divergence-component-audit-v1',passed=True,physical_admission=False,cases=rows,provenance=provenance,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=['Neural component inferred by subtraction of saved FE lift and supremizers; no new neural evaluation.','Dimensionless gradients d(u/U)/d(x/D), dimensionless volume and boundary weights.','Signed cross terms retained; component norms are not additive.','Volume-boundary mismatch is a quadrature diagnostic, not assumed exact for neural fields.','No new PDE solve, fitting, native assembly, validation, or counterfactual solver claim.']),indent=2))
