"""Read-only retained partial development audit; no model/operator reconstruction."""
from pathlib import Path
import json,hashlib,sys,time,resource,signal
import numpy as np
started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('audit60s')));signal.alarm(60)
b=Path(sys.argv[1]);p=b/'results/development-stage/development';ns=b.parent

def sha(f):
 with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def read(f):return json.loads(f.read_text())
def close(a,c):assert abs(a-c)<=1e-11*max(1.,abs(a),abs(c)),(a,c)
def decode(d,z):
 if isinstance(d,dict):
  if 'array' in d:
   a=z[d['array']];assert list(a.shape)==d['shape'] and str(a.dtype)==d['dtype'];h=hashlib.sha256(str((a.dtype.str,a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());assert h.hexdigest()==d['content_sha256'];assert np.isfinite(a).all();return a
  return {k:decode(v,z) for k,v in d.items()}
 if isinstance(d,list):return [decode(v,z) for v in d]
 return d
def load(name):
 doc=read(p/(name+'.json'));assert sha(p/doc['array_file'])==doc['array_sha256']
 with np.load(p/doc['array_file'],allow_pickle=False) as z:return decode(doc['record'],z)
for row in EXPECTED['terminal']['outputs']:
 q=b/row['path'];assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
summary=b/'results/batch_summary.json';assert sha(summary)==EXPECTED['terminal']['summary_sha256']
floors=load('development_floors');assert floors['validation_seen'] is False and floors['bank_online_gate']['passed'] is False
floor_rows=[]
for name,rows in floors['rows'].items():
 for row in rows:
  close(row['velocity_relative'],row['velocity_numerator']/row['velocity_denominator']);close(row['pressure_scaled'],row['pressure_numerator']/row['pressure_denominator'])
  close(row['velocity_numerator']**2,row['projection_velocity']['squared_errors'][0]);close(row['pressure_numerator']**2,row['projection_pressure']['squared_errors'][0])
  floor_rows.append({'model':name,'Re':row['Re'],'velocity_relative':row['velocity_relative'],'pressure_scaled':row['pressure_scaled'],'representation_pass':row['velocity_relative']<=.002 and row['pressure_scaled']<=.004,'verification_scope':'saved scalar constituents only; projected fields not retained'})
geom=ns/'fine-preparation01/results/preparation/prepared/volume/geometry.npz';assert sha(geom)==EXPECTED['geometry_sha256']
with np.load(geom,allow_pickle=False) as z:w=z['weights_dimensionless']
dev=ns/'development-materialize02/results/population';manifest=read(dev/'development.json');assert sha(dev/'development.json')=='7b206ba1ed7ce10a4a76ab17a035d13fe3da2fc4f5f5334966b4faa7e7934070'
references={}
for row in manifest['cases']:
 f=dev/row['array_file'];assert sha(f)==row['array_sha256']
 with np.load(f,allow_pickle=False) as z:references[row['Re']]={'velocity':z['velocity'],'pressure':z['pressure'],'force':row['force_coefficient']}
scored=[]
def scores(record,label):
 ref=references[record['Re']]
 for method,row in record['methods'].items():
  if 'invocation' not in row:continue
  u=row['invocation']['outputs']['velocity'];pr=row['invocation']['outputs']['pressure'];s=row['scores']
  du=u-ref['velocity'];dp=pr-ref['pressure'];un=float(np.sqrt(np.einsum('n,ni,ni->',w,du,du)));ud=float(np.sqrt(np.einsum('n,ni,ni->',w,ref['velocity'],ref['velocity'])));pn=float(np.sqrt(w@(dp*dp)));pd=float(np.sqrt(w.sum()))
  for k,v in [('velocity_numerator',un),('velocity_denominator',ud),('velocity_relative',un/ud),('pressure_numerator',pn),('pressure_denominator',pd),('pressure_scaled',pn/pd)]:close(v,s[k])
  force=np.asarray(s['force_coefficient']);truth=np.asarray(ref['force']);close(abs(force[0]-truth[0])/abs(truth[0]),s['drag_relative']);close(abs(force[1]-truth[1]),s['transverse_absolute'])
  assert row['eligible_accuracy'] is False
  scored.append({'case':label,'Re':record['Re'],'method':method,'status':row['invocation']['work']['status'],'velocity_relative':un/ud,'pressure_scaled':pn/pd,'drag_relative':s['drag_relative'],'transverse_absolute':s['transverse_absolute'],'force_target_pass':s['drag_relative']<=.02 and s['transverse_absolute']<=.002,'eligible_accuracy':False,'force_verification_scope':'retained predicted force scalar error arithmetic, not new traction evaluation'})
for re in (16.875,23.125):scores(load('POD_r2_development_'+str(re)),'unsampled_final')
quad=load('POD_r2_quadrature_candidates');fits=[];gaps=[]
for row in quad:
 f=row['fit'];a=np.asarray(f['selected_normalized_columns']);target=np.asarray(f['normalized_target']);weights=np.asarray(f['selected_weights']);res=target-a@weights;relative=float(np.linalg.norm(res)/np.linalg.norm(target));close(relative,f['relative_moment_residual'])
 gradient=-a.T@res;positive=weights>0;st=float(np.max(abs(gradient[positive]))) if np.any(positive) else float('inf');dual=float(max(0.,-np.min(gradient[~positive]))) if np.any(~positive) else 0.;comp=float(np.max(abs(weights*gradient)));kkt=max(st,dual,comp)/max(float(np.linalg.norm(a,2))*float(np.linalg.norm(target)),1e-12);close(kkt,f['kkt'])
 assert f['passed']==(f['reason']=='fit_tolerance' and kkt<=1e-8)
 fits.append({'support_cap':row['support_cap'],'selected_columns':len(f['selected']),'positive_support':len(f['support']),'relative_moment_residual':relative,'restricted_KKT':kkt,'fit_passed':f['passed'],'reason':f['reason']})
 for check in row['development_checks']:
  scores(check,'support_'+str(row['support_cap']));q=check['methods']['fitted_convection_weak_LS']['quadrature_reached_state_check'];assert not q['passed']
  for at in q['operator_at_both_reached_states']:
   close(at['operator_numerator']/at['operator_denominator'],at['operator_relative']);close(at['jacobian_numerator']/at['jacobian_denominator'],at['jacobian_relative']);assert at['jacobian_relative']>1e-4
   gaps.append({'support_cap':row['support_cap'],'Re':check['Re'],**at,'verification_scope':'saved numerator/denominator consistency; original Jacobian arrays/operators not retained'})
assert not list(p.glob('POD_r4_development*')) and not list(p.glob('POD_r8_development*')) and not list(p.glob('bank_r*_development*'))
result={'schema':'ns-obstacle-independent-partial-development-audit-v1','audit_passed':True,'job_id':'6636731','model_admitted':False,'development_completed':False,'timeout_summary':read(summary),'expected_packet_sha256':EXPECTED_SHA,'source_sha256':{n:sha(b/n) for n in ('pilot2d.py','metrics.py','quadrature_blocked.py','weak.py','development_launch2d.py')},'verified_output_files':len(EXPECTED['terminal']['outputs']),'representation_rows':floor_rows,'actual_saved_field_scores':scored,'quadrature_fit_replays':fits,'reached_operator_ratio_checks':gaps,'limitations':['Representation floor projected fields are absent; scalar constituents checked, no fresh projection.','Force error arithmetic verified from retained predicted coefficients; no fresh boundary traction.','Original full/fitted Jacobian arrays and operator fingerprints absent; only retained ratio constituents and frozen source criterion checked.','PODr4/r8 model/projection records exist but online and quadrature outcomes are incomplete at deadline; do not label them failed or eligible.','No frozen model choice, model validation, or speed comparison follows.'],'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
signal.alarm(0);print(json.dumps(result,indent=2))
