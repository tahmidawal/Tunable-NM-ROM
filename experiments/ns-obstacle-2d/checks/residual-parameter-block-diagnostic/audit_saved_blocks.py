"""Post-terminal saved parameter algebra only; no neural evaluation or fitting."""
import hashlib,json,resource,signal,time
from pathlib import Path
import numpy as np

SHAPES=((21,64),(64,64),(64,2))
def digest(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()
def blocks():
 rows=[];offset=0
 for head in range(4):
  for layer,shape in enumerate(SHAPES):
   for kind,s in (('weight',shape),('bias',(shape[1],))):
    size=int(np.prod(s));rows.append({'head':head,'layer':layer,'kind':kind,'shape':list(s),'start':offset,'end':offset+size,'role':'output' if layer==2 else 'hidden'});offset+=size
 assert offset==22792
 return rows
def stats(a):
 a=np.asarray(a);assert a.dtype==np.float64 and a.ndim==1 and np.isfinite(a).all()
 return {'count':len(a),'l2':float(np.linalg.norm(a)),'linf':float(np.max(abs(a))) if len(a) else 0.,'exact_zero_count':int(np.count_nonzero(a==0)),'nonzero_count':int(np.count_nonzero(a)),'sum_squares':float(a@a)}
def summarize(theta,gradient,initial=None,previous=None):
 assert theta.shape==gradient.shape==(22792,) and theta.dtype==gradient.dtype==np.float64
 rows=[]
 for b in blocks():
  s=slice(b['start'],b['end']);t,g=theta[s],gradient[s];row=dict(b,parameter=stats(t),gradient=stats(g))
  row['gradient_l2_over_parameter_l2']=row['gradient']['l2']/row['parameter']['l2'] if row['parameter']['l2'] else None
  if initial is not None:row['parameter_delta_from_initial']=stats(t-initial[0][s]);row['gradient_delta_from_initial']=stats(g-initial[1][s])
  if previous is not None:row['parameter_delta_from_previous_retained']=stats(t-previous[0][s]);row['gradient_delta_from_previous_retained']=stats(g-previous[1][s])
  rows.append(row)
 groups=[]
 for head in (None,0,1,2,3):
  for role in ('all','hidden','output'):
   ids=np.concatenate([np.arange(b['start'],b['end']) for b in blocks() if (head is None or b['head']==head) and (role=='all' or b['role']==role)])
   groups.append({'head':head,'role':role,'parameter':stats(theta[ids]),'gradient':stats(gradient[ids]),'parameter_delta_from_initial':stats(theta[ids]-initial[0][ids]) if initial else None,'gradient_delta_from_initial':stats(gradient[ids]-initial[1][ids]) if initial else None})
 assert np.isclose(sum(row['parameter']['sum_squares'] for row in rows),float(theta@theta),rtol=1e-13,atol=1e-14)
 assert np.isclose(sum(row['gradient']['sum_squares'] for row in rows),float(gradient@gradient),rtol=1e-13,atol=1e-14)
 return {'blocks':rows,'groups':groups}
def run(binding,audit):
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('saved block diagnostic55s')));signal.alarm(55)
 assert audit['schema']=='ns-obstacle-independent-residual-refinement-saved-audit-v1' and audit['passed'] is True and audit['physical_admission'] is False
 assert audit['status'] in ('job_deadline','gradient_stationary','line_search_stopped','iteration_limit') and audit['selected_theta_scope']=='child parameters only' and audit['trainable_parameters']==22792
 assert audit['selected_model_sha256']==binding['model_sha256'] and audit['parent_model_sha256']==binding['parent_model_sha256']=='cd052629569aa1b8ea243a65a9ec22bd4a9ce9a0713999047d8ca1fb98a748d5'
 root=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-residual01/results/fit');provenance={}
 def check(p):
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  assert str(p) in audit['provenance'] and audit['provenance'][str(p)]==h;provenance[str(p)]=h;return h
 final_path=root/'FINAL.json';check(final_path);final=json.loads(final_path.read_text());assert final['status']==audit['status'] and final['accepted_steps']==audit['accepted_steps']
 checkpoints=final['checkpoints'];assert checkpoints[0]['label']=='initial' and checkpoints[-1]['label']=='final' and len(checkpoints)==audit['retained_checkpoints']
 records=[];initial=None;previous=None;previous_iteration=None
 for row in checkpoints:
  stem=f"state-{row['iteration']:05d}-{row['label']}";rp=root/(stem+'.json');check(rp);assert json.loads(rp.read_text())==row and row['array_file']==stem+'.npz'
  p=root/row['array_file'];assert check(p)==row['array_sha256']
  with np.load(p,allow_pickle=False) as z:theta=z['theta'].copy();gradient=z['gradient'].copy()
  assert digest(theta)==row['theta_sha256']==audit['checkpoint_semantic_arrays'][stem]['theta'] and digest(gradient)==row['gradient_sha256']==audit['checkpoint_semantic_arrays'][stem]['gradient']
  assert np.isfinite(theta).all() and np.isfinite(gradient).all()
  if initial is None:initial=(theta.copy(),gradient.copy())
  record={'iteration':row['iteration'],'label':row['label'],'theta_sha256':digest(theta),'gradient_sha256':digest(gradient),'objective':row['objective'],'previous_retained_iteration':previous_iteration,'accepted_steps_since_previous_retained':None if previous_iteration is None else row['iteration']-previous_iteration,**summarize(theta,gradient,initial,previous)}
  records.append(record);previous=(theta,gradient);previous_iteration=row['iteration']
 assert records[-1]['theta_sha256']==audit['selected_theta_sha256']
 parent=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-traction01/results/fit/spatial-bank-raw.npz');assert check(parent)==binding['parent_model_sha256'];parent_rows=[]
 with np.load(parent,allow_pickle=False) as z:
  md=json.loads(str(z['metadata_json']))
  for layer in range(md['layers']):
   for kind,key in (('weight','W'),('bias','b')):
    a=z[key+str(layer)];assert a.dtype==np.float64 and np.isfinite(a).all();parent_rows.append({'layer':layer,'kind':kind,'shape':list(a.shape),'parameter':stats(a.ravel()),'gradient':None,'gradient_scope':'Frozen parent excluded from child optimizer; no parent-gradient data inferred.'})
 initial_hidden=next(g for g in records[0]['groups'] if g['head'] is None and g['role']=='hidden')
 signal.alarm(0)
 return {'schema':'ns-obstacle-residual-saved-parameter-block-diagnostic-v1','passed':True,'physical_admission':False,'independent_model_audit_sha256':binding['audit_sha256'],'model_sha256':binding['model_sha256'],'parent_model_sha256':binding['parent_model_sha256'],'status':audit['status'],'child_parameters':22792,'output_parameters':520,'hidden_parameters':22272,'initial_hidden_gradient_exact_zero_count':initial_hidden['gradient']['exact_zero_count'],'initial_hidden_gradient_linf':initial_hidden['gradient']['linf'],'parent_parameter_blocks':parent_rows,'retained_states':records,'provenance':provenance,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'limitations':['No neural evaluation, objective/autodiff replay, training or PDE solve.','No running outputs: only terminal independently audited checkpoints.','Gradient support counts use exact floating zero, not a chosen smallness threshold.','Parameter norms/ratios are coordinate-dependent diagnostics, not field correction amplitudes or Hessian condition estimates.','Retained-state differences span many accepted steps; missing intermediate optimizer directions/gradients are not reconstructed.','Frozen parent has no child-optimizer gradient; null is not an inferred zero parent loss gradient.','No scale selection, convergence/plateau inference, validation or new run authorization.']}

if __name__=='__main__' and 'BINDING' in globals():
 print(json.dumps(run(BINDING,AUDIT),indent=2,allow_nan=False))
