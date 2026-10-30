import hashlib,json,pathlib,time,resource,numpy as np
start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
# BINDING is injected by a separately pinned collector only after actual completion.
# Exact parent audit binding supplied by collector.
B=pathlib.Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027'); P=B/'neural-pod-guided-diagnostic03/results/diagnostic'; Q=B/'fine-preparation01/results/preparation/prepared'; D=B/'development-materialize02/results/population'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance={}
def checked(p,h=None):
 a=sha(p)
 if h is not None: assert a==h,(str(p),a,h)
 assert str(p) in ACCEPTED_PROVENANCE and a==ACCEPTED_PROVENANCE[str(p)]
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
checked(P/'DIAGNOSTIC.json',BINDING['status_sha256']);top=json.loads((P/'DIAGNOSTIC.json').read_text());s=load(pathlib.Path(top['latest_evidence']['record_file']).stem,top['latest_evidence']);assert provenance[str(P/top['latest_evidence']['record_file'])]==BINDING['latest_record_sha256']
m=load(pathlib.Path(s['model']['record_file']).stem,s['model']);f=m['force'];assert f['velocity_per_nu'].shape==(2,8) and f['pressure'].shape==(2,4)
checked(Q/'manifest.json','7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6');checked(Q/'boundary.npz','7965a23c84dcd515322efb779aec7c4da3462206a211781a3d1f7d9ae51e619f');bd=np.load(Q/'boundary.npz');sel=bd['labels']=='obstacle';w=bd['weights_dimensionless'][sel];n=-bd['normals_fluid'][sel];pl=bd['lift_p_per_nu'][sel,...,0];lp=-2*np.einsum('n,n,ni->i',w,pl,n)
assert [c['Re'] for c in s['cases']]==[t['Re'] for t in TRUTH]==[16.875,23.125]
rows=[]
for c,t in zip(s['cases'],TRUTH):
 assert c['Re']==t['Re'];re=c['Re'];raw=load(pathlib.Path(c['raw_invocation']['record_file']).stem,c['raw_invocation']);out={}
 for role,z in [('actual_PDE',np.asarray(raw['evidence']['state'])),('L2_projection_not_PDE',np.asarray(c['representation_diagnostic']['achieved_state']))]:
  assert z.shape==(12,) and np.isfinite(z).all();pressure=lp/re+f['pressure']@z[8:];vl=(f['lifting_per_nu']-lp)/re;vn=f['velocity_per_nu'][:,:4]@z[:4]/re;vs=f['velocity_per_nu'][:,4:]@z[4:8]/re;viscous=vl+vn+vs;total=pressure+viscous;original=(f['lifting_per_nu']+f['velocity_per_nu']@z[:8])/re+f['pressure']@z[8:];assert np.allclose(total,original,rtol=1e-13,atol=1e-13)
  saved=c['scores']['force_coefficient'] if role=='actual_PDE' else c['representation_diagnostic']['force_coefficient'];assert np.allclose(total,saved,rtol=1e-13,atol=1e-13)
  pe=pressure-np.asarray(t['pressure_coefficient']);ve=viscous-np.asarray(t['viscous_coefficient']);assert np.allclose(pe+ve,total-np.asarray(t['order12_total']),rtol=1e-12,atol=1e-13)
  assert np.allclose(pe+ve+np.asarray(t['order12_minus_order16']),total-np.asarray(t['order16_total']),rtol=1e-12,atol=1e-13)
  out[role]=dict(state=z.tolist(),pressure=pressure.tolist(),viscous=viscous.tolist(),viscous_lift=vl.tolist(),viscous_neural=vn.tolist(),viscous_supremizer=vs.tolist(),total=total.tolist(),signed_pressure_error_vs_truth_order12=pe.tolist(),signed_viscous_error_vs_truth_order12=ve.tolist(),signed_total_error_vs_truth_order12=(pe+ve).tolist(),signed_total_error_vs_truth_order16=(total-np.asarray(t['order16_total'])).tolist(),force_matrix_replay_gap=float(abs(total-original).max()))
 difference={k:(np.asarray(out['actual_PDE'][k])-np.asarray(out['L2_projection_not_PDE'][k])).tolist() for k in ('pressure','viscous','viscous_lift','viscous_neural','viscous_supremizer','total')}
 rows.append(dict(Re=re,truth=t,states=out,solved_minus_projection=difference))
print(json.dumps(dict(schema='ns-obstacle-signed-force-components-v1',passed=True,physical_admission=False,cases=rows,provenance=provenance,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=['Truth pressure/viscous split is native order12, not canonical order16; only total order12-to16 gap retained.','ROM components use saved force matrices and prepared pressure lift; no fresh neural derivatives or traction assembly.','L2 projection is representation-only, not a PDE solve.','Signed component errors cancel; absolute component magnitudes are not additive error attribution.']),indent=2))
