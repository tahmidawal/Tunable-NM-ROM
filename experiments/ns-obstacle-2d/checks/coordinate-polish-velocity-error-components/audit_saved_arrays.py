import hashlib,json,pathlib,time,resource,signal,numpy as np
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("115s remote audit bound")));signal.alarm(115)
start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
# BINDING is injected by a separately pinned collector only after actual completion.
assert set(BINDING)=={'diagnostics'} and [d['variant'] for d in BINDING['diagnostics']]==['coordinate','output_polish']
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
checked(D/'development.json','7b206ba1ed7ce10a4a76ab17a035d13fe3da2fc4f5f5334966b4faa7e7934070');pop=json.loads((D/'development.json').read_text());assert [r['Re'] for r in pop['cases']]==[16.875,23.125]
truth={}
for r in pop['cases']:
 data=arrays(D/r['array_file'],r['array_sha256']);truth[r['Re']]=data['velocity'];assert truth[r['Re']].shape==(len(w),2) and np.isfinite(truth[r['Re']]).all()
rows=[];states=[]
for current in BINDING['diagnostics']:
 P=B/current['job_namespace']/'results/diagnostic'
 assert current['job_namespace']=={'coordinate':'neural-pod-guided-diagnostic07','output_polish':'neural-pod-guided-diagnostic08'}[current['variant']]
 checked(P/'DIAGNOSTIC.json',current['status_sha256']);top=json.loads((P/'DIAGNOSTIC.json').read_text());assert top['completed_case_records']==2 and top['latest_evidence']['record_sha256']==current['latest_record_sha256']
 s=load(pathlib.Path(top['latest_evidence']['record_file']).stem,top['latest_evidence']);m=load('pod_guided_neural_r4_model',s['model']);assert m['operator']['viscous'].shape[1]==9 and m['operator']['pressure'].shape[1]==4
 assert s['model_sha256']==current['model_sha256'] and s['validation_seen'] is False and s['reference_fit_initialization_used'] is False and s['training_or_quadrature_fit_performed'] is False
 checked(P.parents[1]/current['release_file'],current['release_sha256']);assert [c['Re'] for c in s['cases']]==[16.875,23.125]
 for case in s['cases']:
  v=load('pod_guided_neural_r4_'+str(case['Re'])+'_raw_invocation',case['raw_invocation']);z=v['evidence']['state'];u=v['outputs']['velocity'];assert z.shape==(12,) and u.shape==(len(w),2) and z.dtype==u.dtype==np.float64
  rows.append(dict(variant=current['variant'],Re=case['Re'],model_sha256=s['model_sha256'],accepted_audit_sha256=current['audit_sha256'],state=z.tolist(),recorded_velocity_relative=case['scores']['velocity_relative'],primitive_gram=np.zeros((8,8)),truth_norm_squared=0.,direct_error_norm_squared=0.,pointwise_closure_maxgap=0.))
  states.append((z,u,truth[case['Re']]))
nextpoint=0
for i,block0 in enumerate(vm['fields']['lift_u']['blocks']):
 lo,hi=block0['start'],block0['end'];assert lo==nextpoint;nextpoint=hi;data={}
 for name in ('lift_u','pod_velocity','supremizers'):
  block=vm['fields'][name]['blocks'][i];assert (block['start'],block['end'])==(lo,hi);fn='volume/'+block['file'];assert ma['coefficient_files'][fn]==block['sha256']==vm['files'][block['file']];data[name]=arrays(Q/fn,block['sha256'])['values'];assert data[name].dtype==np.float64 and np.isfinite(data[name]).all()
 lift=data['lift_u'][...,0];T=data['pod_velocity'][...,:4];S=data['supremizers'][...,:4];assert T.shape==S.shape==(hi-lo,2,4) and lift.shape==(hi-lo,2)
 wb=w[lo:hi]
 for row,(z,u,ref) in zip(rows,states):
  h=ref[lo:hi]-lift;ta=T@z[:4];sb=S@z[4:8];ea=u[lo:hi]-lift-sb-ta
  primitive=np.concatenate([np.stack([ea,ta,sb,-h]),T.transpose(2,0,1)],axis=0)
  row['primitive_gram']+=np.einsum('n,ani,bni->ab',wb,primitive,primitive)
  direct=u[lo:hi]-ref[lo:hi];gap=float(np.max(np.abs(primitive[:4].sum(axis=0)-direct)));row['pointwise_closure_maxgap']=max(row['pointwise_closure_maxgap'],gap);assert gap<2e-13
  row['truth_norm_squared']+=float(np.einsum('n,ni,ni',wb,ref[lo:hi],ref[lo:hi]));row['direct_error_norm_squared']+=float(np.einsum('n,ni,ni',wb,direct,direct))
assert nextpoint==len(w) and len(rows)==4
def matrix_checks(M, inherited_bound=0.):
 assert np.isfinite(M).all()
 scale=float(np.linalg.norm(M,ord=2));bound=max(inherited_bound,64*np.finfo(np.float64).eps*len(vm['fields']['lift_u']['blocks'])*max(scale,np.finfo(np.float64).tiny))
 symmetry=float(np.max(np.abs(M-M.T)));minimum=float(np.linalg.eigvalsh((M+M.T)/2)[0]);diagonal=float(np.diag(M).min());assert symmetry<=bound and minimum>=-bound and diagonal>=-bound,(symmetry,minimum,diagonal,bound)
 return dict(finite=True,spectral_scale=scale,symmetry_maxgap=symmetry,minimum_eigenvalue=minimum,minimum_diagonal=diagonal,roundoff_bound=bound,clipped_diagonal_indices=np.where(np.diag(M)<0)[0].tolist(),clipped_diagonal_values=np.diag(M)[np.diag(M)<0].tolist())
for row in rows:
 G=row.pop('primitive_gram');gcheck=matrix_checks(G);gram=G[4:,4:];rhs=-G[4:,3];eig=np.linalg.eigvalsh(gram);assert eig[0]>0 and np.linalg.matrix_rank(gram)==4
 c=np.linalg.solve(gram,rhs);orth=gram@c-rhs;assert np.linalg.norm(orth)<=1e-12*max(np.linalg.norm(rhs),1.)
 A=np.zeros((4,8));A[:,:4]=np.eye(4);A[1,4:]=-c;A[3,4:]=c;E=A@G@A.T;echeck=matrix_checks(E,gcheck['roundoff_bound']*float(np.linalg.norm(A,ord=2))**2);row['matrix_roundoff_checks']={'primitive':gcheck,'components':echeck,'clipping_policy':'Only roundoff-bounded negative component diagonals clipped to zero for displayed norms; raw Gram retained unchanged.'};total=float(E.sum());direct=row['direct_error_norm_squared'];assert np.isclose(total,direct,rtol=2e-8,atol=2e-14)
 relative=float(np.sqrt(direct/row['truth_norm_squared']));assert np.isclose(relative,row['recorded_velocity_relative'],rtol=2e-12,atol=2e-14)
 row.update(teacher_only_projection_coefficients=c.tolist(),teacher_gram=gram.tolist(),teacher_rhs=rhs.tolist(),teacher_gram_eigenvalues=eig.tolist(),teacher_gram_condition=float(eig[-1]/eig[0]),teacher_rank=4,teacher_projection_orthogonality=orth.tolist(),primitive_gram=G.tolist(),component_order=['eA_actual_neural_minus_teacher_same_coefficients','eC_teacher_coefficient_difference','eS_supremizer','eT_teacher_only_projection_error'],weighted_error_gram=E.tolist(),signed_cross_terms=(2*np.triu(E,1)).tolist(),component_l2_norms=np.sqrt(np.maximum(np.diag(E),0)).tolist(),relative_squared_gram=(E/row['truth_norm_squared']).tolist(),total_squared_from_gram=total,replayed_velocity_relative=relative)
print(json.dumps(dict(schema='ns-obstacle-convention-dependent-velocity-error-components-v1',passed=True,physical_admission=False,convention='h=truth-lift; c solves T-only weighted L2 projection using first four retained prepared POD velocity modes. eA=u-lift-Sb-Ta; eC=T(a-c); eS=Sb; eT=Tc-h.',cases=rows,provenance=provenance,units={'velocity':'u/U','weights':'dx/D**2'},elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=['Convention-dependent aggregate identity, not causal attribution or per-mode error allocation.','No enriched projection coefficients used; c is a new small saved-array T-only projection diagnostic, not a PDE solve or model fit.','Actual neural contribution inferred from saved solved velocity minus retained FE lift and supremizers; no network evaluation.','Component norms are not additive; all signed cross terms retained.','Existing TRAIN-derived teacher/lift/sup arrays only; no new TRAIN population/evaluation or final validation access.','No native assembly, neural differentiation, running fit reads, or parameter changes.']),indent=2,allow_nan=False))
