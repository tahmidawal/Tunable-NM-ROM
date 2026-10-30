"""Audit completed compact warm refinement; BINDING supplied only after terminal retention."""
from pathlib import Path
import hashlib,json,time,resource,signal,io
import numpy as np
start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('audit115s')));signal.alarm(115)
B=Path(BINDING['job_root']);P=B/BINDING['result_relative'];provenance={};ledger={}
def check(p,h=None):
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if h is not None:assert actual==h,(str(p),actual,h)
 if p.is_relative_to(B) and p!=B/'COLLECTION-SHA256SUMS':assert str(p.relative_to(B)) in ledger and actual==ledger[str(p.relative_to(B))]
 provenance[str(p)]=actual;return actual
def finite_tree(v):
 if isinstance(v,float):assert np.isfinite(v)
 elif isinstance(v,dict):
  for x in v.values():finite_tree(x)
 elif isinstance(v,list):
  for x in v:finite_tree(x)
def read(p,h=None):
 check(p,h);v=json.loads(p.read_text());finite_tree(v);return v
def digest(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()
def arrays(p,h):
 check(p,h)
 with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in a.values());return a
def lines(n):
 p=P/n
 if not p.exists():return []
 check(p);v=[json.loads(x) for x in p.read_text().splitlines()];finite_tree(v);return v
check(B/'COLLECTION-SHA256SUMS',BINDING['ledger_sha256'])
for line in (B/'COLLECTION-SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);n=n.removeprefix('./');assert n not in ledger;ledger[n]=h
release=read(B/BINDING['release_file'],BINDING['release_sha256']);assert release['authorized'] and release['validation_allowed'] is False
plan=read(B/'PLAN.json',release['plan_sha256']);inventory=read(B/'RUNTIME-INVENTORY.json',release['runtime_inventory_sha256'])
for item in inventory['files']:
 rel=Path(item['destination']);assert not rel.is_absolute() and '..' not in rel.parts;assert (B/rel).stat().st_size==item['bytes'];check(B/rel,item['sha256'])
c=read(P/'CONFIG.json',BINDING['config_sha256']);final=read(P/'FINAL.json',BINDING['final_sha256']);receipt=read(P/'MODEL-RECEIPT.json',BINDING['receipt_sha256'])
assert B==Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-coordinate01') and P==B/'results/fit'
assert release['schema']=='ns-obstacle-fixed-coordinate-residual-refinement-release-v1' and plan['schema']=='ns-obstacle-fixed-coordinate-residual-refinement-plan-v1'
assert release['job_directory']==plan['job_directory']==str(B) and release['output_directory']==str(P) and release['single_execution'] is True and release['parent_frozen'] is True
assert set(plan['inputs'])=={'prepared','preparation_release','parent_bank','parent_receipt','parent_review','training_coefficients_source','warm_bank','warm_receipt','warm_review'}
assert c['parent_bank_sha256']=='cd052629569aa1b8ea243a65a9ec22bd4a9ce9a0713999047d8ca1fb98a748d5'
for name in ('parent_bank','parent_receipt','parent_review'):
 desc=plan['inputs'][name];assert BINDING[name]==desc['path'] and c[name+'_sha256']==desc['sha256'] and desc['kind']=='file' and Path(desc['path']).stat().st_size==desc['bytes']
assert final['warm_restart'] is True
assert c['schema']=='ns-obstacle-fixed-coordinate-residual-head-training-v1' and c['protocol']=={'max_iterations':10000,'checkpoint_every':1000,'history_size':10,'line_search_steps':20,'gradient_tolerance':1e-10,'warm_restart':True,'new_adam_steps':0,'supervised_divergence_weight':1.0,'traction_weight':1.0,'head_seed':30102026,'hidden_coordinate_scale':100.0,'output_coordinate_scale':1.0}
assert c['protocol']==plan['refinement_protocol']==release['refinement_protocol']
assert c['warm_restart'] is True and c['curvature_reset'] and not c['exact_optimizer_continuation_claim'] and not c['all_intermediate_vectors_retained'];check(B/'accuracy_v2/fixed_coordinate_residual_refine.py',c['source_sha256']);assert c['source_sha256']==BINDING['trainer_sha256']
parent=Path(BINDING['parent_bank']);check(parent,c['parent_bank_sha256']);pr=read(Path(BINDING['parent_receipt']),c['parent_receipt_sha256']);review=read(Path(BINDING['parent_review']),c['parent_review_sha256']);assert review['passed'] and review['model_sha256']==c['parent_bank_sha256'] and review['model_receipt_sha256']==c['parent_receipt_sha256'];assert pr['selected_theta_sha256']==c['parent_selected_theta_sha256']

def bank(p,h):
 check(p,h)
 with np.load(p,allow_pickle=False) as z:
  md=json.loads(str(z['metadata_json']));finite_tree(md);assert np.array_equal(z['transform'],np.eye(4));n=md['layers'];assert set(z.files)=={'metadata_json','transform'}|{k+str(i) for i in range(n) for k in ('W','b')};a=[z[k+str(i)] for i in range(n) for k in ('W','b')];assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in a);theta=np.concatenate([v.ravel() for v in a])
 return md,theta
pm,pt=bank(parent,c['parent_bank_sha256']);assert digest(pt)==c['parent_selected_theta_sha256'];t=c['teacher_identity'];assert t==pm['teacher_identity'] and c['config']==pm['config'] and t['split']=='training' and t['Re']==[15+1.25*i for i in range(9)]
assert t['reference_admission_sha256']=='5c9402af9ddb2af0ad8925ef224a43ae72cb36209df99865908617c63b5b506f';assert t['prepared_manifest_sha256']=='7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6';assert t['volume_manifest_sha256']=='21563bc7a114147716d300e8f1981e664da5ae728db6b1aa21ee3aa8bd947866';assert t['training_array_sha256']=='6a65cbcd387c32b9ab70fc3956335a975e40b62646370fdf3413c7df4bfd8f30'
assert pm.get('objective_variant')=='supervised_trace_and_boundary_traction_lambda_1' and c['objective_changed_from_parent'] is False and c['absolute_incompressibility_enforced'] is False and c['causal_comparison_to_equal_additional_base_training'] is False
assert c['architecture']=={'head_count':4,'layer_widths':[21,64,64,2],'trainable_parameters':22792,'parent_frozen':True}
assert c['selected_theta_scope']=='child parameters only; embedded parent hash is separate'
assert c['optimizer_counts_scope']=='child warm restart plus historical lineage; curvature reset, no exact optimizer continuation'
# Exact released warm child; no random reinitialization and no neural evaluation.
for name,key in (('warm_bank','warm_child_model_sha256'),('warm_receipt','warm_child_receipt_sha256'),('warm_review','warm_child_review_sha256')):
 desc=plan['inputs'][name];assert BINDING[name]==desc['path'] and desc['sha256']==c[key] and desc['kind']=='file' and Path(desc['path']).stat().st_size==desc['bytes'];check(Path(desc['path']),desc['sha256'])
assert c['warm_child_model_sha256']=='1e6c1cca50313e9fec4d89565f1e09d36137c376bc60bc1be3f7b193905326f4'
wr=read(Path(BINDING['warm_receipt']));wv=read(Path(BINDING['warm_review']))
with np.load(BINDING['warm_bank'],allow_pickle=False) as z:
 wm=json.loads(str(z['metadata_json']));finite_tree(wm);assert wm['schema']=='ns-obstacle-independent-residual-head-bank-v1' and wm['objective_variant']=='independent_residual_heads_same_trace_and_traction'
 raw=z['embedded_parent_npz'];assert raw.dtype==np.uint8 and raw.ndim==1 and raw.tobytes()==parent.read_bytes()
 aa=[]
 for i in range(4):
  for j,shape in enumerate(((21,64),(64,64),(64,2))):
   w0,b0=z[f'head{i}_W{j}'],z[f'head{i}_b{j}'];assert w0.shape==shape and b0.shape==(shape[1],) and w0.dtype==b0.dtype==np.float64 and np.isfinite(w0).all() and np.isfinite(b0).all();aa.extend((w0,b0))
 ct=np.concatenate([a.ravel() for a in aa])
assert ct.shape==(22792,) and digest(ct)==c['warm_child_selected_theta_sha256']==wr['selected_theta_sha256']==wv['selected_theta_sha256']
assert wr['bank_sha256']==wv['model_sha256']==c['warm_child_model_sha256'] and wv['passed'] is True and wv['model_receipt_sha256']==c['warm_child_receipt_sha256']
assert wr['parent_model_sha256']==wv['parent_model_sha256']==wm['parent_sha256']==c['parent_bank_sha256'] and wr['selected_theta_scope']=='child parameters only'
assert wr['selected_checkpoint']['objective']==c['parent_full_training_objective'] and c['prior_allocation_seconds']==wv['allocated_seconds'] and c['prior_adam_steps']==wv['adam_steps_recorded']==wm['actual_applied_adam_steps'] and c['prior_lbfgs_steps']==wv['accepted_lbfgs_steps']==wm['accepted_lbfgs_steps']
for key in ('teacher_identity','value_norms_squared','gradient_norms_squared','mode_weights','config'):assert wm[key]==pm[key]
assert wm['refinement']['boundary_teacher']['arrays']==c['boundary_teacher']['arrays'] and wm['refinement']['boundary_teacher']['normalization']==c['boundary_teacher']['normalization']
cm=arrays(P/'fixed-coordinate-map.npz',c['coordinate_map_arrays_sha256']);assert set(cm)=={'diagonal','anchor'}
D=cm['diagonal'];expected_D=np.tile(np.r_[np.full(5568,100.),np.ones(130)],4);assert D.shape==ct.shape and np.array_equal(D,expected_D) and np.array_equal(cm['anchor'],ct)
assert digest(D)==c['coordinate_diagonal_sha256'] and digest(ct)==c['coordinate_anchor_sha256']
assert c['coordinate_definition']=='theta=initial_child_theta+D*z; fixed positive diagonal; physical gradient stopping; same physical history safeguards'
check(B/'accuracy_v2/fixed_lbfgs_coordinates.py',c['coordinate_helper_sha256'])
assert release['hidden_coordinate_scale']==c['protocol']['hidden_coordinate_scale']==100.0 and release['output_coordinate_scale']==c['protocol']['output_coordinate_scale']==1.0
assert plan['coordinate_candidate_sha256']==release['coordinate_candidate_sha256']
candidate=read(B/'COORDINATE-CANDIDATE.json',plan['coordinate_candidate_sha256']);assert candidate['schema']=='ns-obstacle-fixed-coordinate-training-candidate-v1' and candidate['model_sha256']==c['warm_child_model_sha256'] and candidate['frozen_parent_sha256']==c['parent_bank_sha256'] and candidate['hidden_coordinate_scale']==100.0 and candidate['output_coordinate_scale']==1.0
coordinate_replay={'diagonal_sha256':digest(D),'anchor_sha256':digest(ct),'hidden_scale':100.,'output_scale':1.,'checkpoint_witnesses':[],'completed_trial_witnesses':0,'limits':'Only retained physical vectors and scalar coordinate witnesses; no intermediate direction/vector or fresh objective reconstruction.'}
weights=np.asarray(pm['mode_weights']);sigma=np.asarray(t['singular_values']);expected=np.sqrt(sigma/sigma[0]);expected/=expected.sum();assert np.allclose(weights,expected,rtol=1e-15,atol=0)
for k in ('value_norms_squared','gradient_norms_squared'):assert np.asarray(pm[k]).shape==(4,) and np.isfinite(pm[k]).all() and min(pm[k])>0
CP=B/'results/training-coefficients';cert=read(CP/'RESULT.json');assert cert==c['coefficient_covariance'];assert cert['schema']=='ns-obstacle-original-training-coefficient-energy-binding-v1' and cert['passed'] and cert['split']=='training' and cert['Re']==t['Re'] and cert['teacher_identity']==t and not cert['development_or_validation_read']
assert cert['tolerances']=={'gram':1e-10,'whitened_covariance_spectral':1e-10};check(B/'accuracy_v2/training_coefficients.py',cert['source_sha256'])
ca=arrays(CP/cert['arrays_file'],cert['arrays_sha256']);projection=arrays(CP/'projection-inputs.npz',ledger[str((CP/'projection-inputs.npz').relative_to(B))]);white=arrays(CP/cert['whitened_covariance_file'],cert['whitened_covariance_sha256'])
assert set(ca)=={'coefficients','gram','rhs','covariance','expected_covariance','singular_values','Re'} and set(projection)=={'gram','rhs','Re'} and set(white)=={'error'}
assert ca['gram'].shape==(4,4) and ca['rhs'].shape==ca['coefficients'].shape==(4,9);assert np.array_equal(ca['Re'],t['Re']) and np.array_equal(ca['singular_values'],sigma)
for k in projection:assert np.array_equal(projection[k],ca[k])
prepared=Path(plan['inputs']['prepared']['path']);manifest=read(prepared/'manifest.json',t['prepared_manifest_sha256']);vm=read(prepared/'volume/manifest.json',t['volume_manifest_sha256']);assert vm['units']=={'values':'dimensionless','gradients':'d/d(x/D)','weights':'dx/D**2'}
training=prepared/'training.npz';check(training,cert['training_file_sha256']);assert cert['training_file_sha256']==manifest['coefficient_files']['training.npz'];assert cert['training_array_sha256']==t['training_array_sha256']
with np.load(training,allow_pickle=False) as z:xx=z['points_physical'];ww=z['weights_dimensionless'];snap=z['velocity_corrections'];re=z['Re']
assert np.array_equal(re,t['Re']) and xx.shape==(1036497,2) and ww.shape==(1036497,) and snap.shape==(9,1036497,2) and all(a.dtype==np.float64 and np.isfinite(a).all() for a in (xx,ww,snap)) and np.min(ww)>0
h=hashlib.sha256()
for a in (xx,ww,snap):h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes())
assert h.hexdigest()==t['training_array_sha256'];assert digest(xx)==t['arrays']['points'] and digest(ww)==t['arrays']['weights']
ht=hashlib.sha256(str(((len(ww),2,4),np.dtype('float64').str)).encode());gram=np.zeros((4,4));rhs=np.zeros((4,9));cursor=0
for block in vm['fields']['pod_velocity']['blocks']:
 lo,hi=block['start'],block['end'];assert lo==cursor;cursor=hi;name='volume/'+block['file'];assert manifest['coefficient_files'][name]==block['sha256']==vm['files'][block['file']]
 a=arrays(prepared/name,block['sha256']);v=np.ascontiguousarray(a['values'][...,:4]);assert v.shape==(hi-lo,2,4);ht.update(v.tobytes());gram+=np.einsum('n,ncr,ncs->rs',ww[lo:hi],v,v);rhs+=np.einsum('n,ncr,snc->rs',ww[lo:hi],v,snap[:,lo:hi,:])
assert cursor==len(ww) and ht.hexdigest()==t['arrays']['values'];gram_gap=float(abs(gram-ca['gram']).max());rhs_gap=float(abs(rhs-ca['rhs']).max());assert np.allclose(gram,ca['gram'],rtol=2e-12,atol=2e-14) and np.allclose(rhs,ca['rhs'],rtol=2e-12,atol=2e-14)
coef=np.linalg.solve(ca['gram'],ca['rhs']);assert np.allclose(coef,ca['coefficients'],rtol=2e-12,atol=2e-14) and digest(ca['coefficients'])==cert['coefficients_sha256'];cov=ca['coefficients']@ca['coefficients'].T;expected_cov=np.diag(sigma*sigma);err=(cov-expected_cov)/(sigma[:,None]*sigma[None,:]);assert np.allclose(cov,ca['covariance'],rtol=2e-12,atol=2e-14) and np.array_equal(expected_cov,ca['expected_covariance']) and np.allclose(err,white['error'],rtol=1e-8,atol=2e-12)
ge=float(np.linalg.norm(ca['gram']-np.eye(4),2));ce=float(np.max(abs(cov-expected_cov))/sigma[0]**2);we=float(np.linalg.norm(err,2));assert ge<=1e-10 and we<=1e-10
for observed,recorded in ((ge,cert['gram_spectral_error']),(ce,cert['covariance_max_error_over_sigma0_squared']),(we,cert['whitened_covariance_spectral_error'])):assert np.isclose(observed,recorded,rtol=1e-6,atol=2e-12)
assert np.allclose(np.abs(np.diag(cov)-sigma*sigma)/(sigma*sigma),cert['diagonal_relative_errors'],rtol=1e-6,atol=2e-12) and np.allclose(err,cert['covariance_errors_over_sigma_products'],rtol=1e-6,atol=2e-12)
raw_coef=np.linalg.solve(gram,rhs);raw_cov=raw_coef@raw_coef.T;raw_white=(raw_cov-expected_cov)/(sigma[:,None]*sigma[None,:]);raw_ge=float(np.linalg.norm(gram-np.eye(4),2));raw_we=float(np.linalg.norm(raw_white,2));raw_coef_gap=float(abs(raw_coef-ca['coefficients']).max());assert raw_ge<=1e-10 and raw_we<=1e-10;assert np.allclose(raw_coef,ca['coefficients'],rtol=2e-12,atol=2e-14)
coefficient_replay=dict(raw_coefficient_max_gap=raw_coef_gap,raw_gram_spectral_error=raw_ge,raw_whitened_covariance_spectral_error=raw_we,raw_covariance=raw_cov.tolist(),saved_covariance=ca['covariance'].tolist(),raw_coefficients=raw_coef.tolist(),saved_coefficients=ca['coefficients'].tolist(),raw_gram_max_gap=gram_gap,raw_rhs_max_gap=rhs_gap,gram_spectral_error=ge,whitened_covariance_spectral_error=we,covariance_max_error_over_sigma0_squared=ce,training_points=cursor,semantic_teacher_values_sha256=ht.hexdigest(),scope='Original retained-four TRAIN coefficients only; excludes omitted modes and online coefficients.')
del snap,xx,ww
BT=B/'results/boundary-teacher';br=read(BT/'RESULT.json');assert br==c['boundary_teacher'] and br['schema']=='ns-obstacle-training-cylinder-traction-teacher-v1' and br['split']=='training' and br['Re']==t['Re'] and br['volume_teacher_identity']==t
assert br['gradient_units']=='d(u/U)/d(x/D)' and br['weight_units']=='ds/D' and not br['Re_in_target'] and not br['pressure_in_target'];check(B/'accuracy_v2/boundary_teacher.py',br['source_sha256'])
check(BT/'cylinder-teacher.npz',br['selected_arrays_sha256'])
with np.load(BT/'cylinder-teacher.npz',allow_pickle=False) as z:ba={k:z[k] for k in z.files}
assert ba['indices'].dtype==np.dtype('int64') and ba['indices'].ndim==1 and all(v.dtype==np.float64 for k,v in ba.items() if k!='indices')
assert set(ba)==set(br['arrays']) and all(digest(v)==br['arrays'][k] and np.isfinite(v).all() for k,v in ba.items())
check(prepared/'boundary.npz',manifest['coefficient_files']['boundary.npz']);assert br['boundary_file_sha256']==manifest['coefficient_files']['boundary.npz']
with np.load(prepared/'boundary.npz',allow_pickle=False) as z:
 ids=np.flatnonzero(z['labels']=='obstacle');expected={'indices':ids,'points_physical':z['points_physical'][ids],'points_dimensionless':z['points_dimensionless'][ids],'weights_dimensionless':z['weights_dimensionless'][ids],'normals_body':-z['normals_fluid'][ids],'teacher_gradient_dimensionless':z['pod_velocity_gradient_dimensionless'][ids,...,:4],'singular_values':sigma}
assert all(ba[k].shape==v.shape and np.array_equal(ba[k],v) for k,v in expected.items());assert br['prepared_manifest_sha256']==t['prepared_manifest_sha256'] and manifest['gradient_axes']=='point, component, derivative_coordinate, column';assert np.array_equal(ba['points_physical']/.1,ba['points_dimensionless']);assert manifest['scales']['D']==.1 and manifest['units']['boundary_weights']=='ds/D' and manifest['units']['all_exported_field_gradients']=='derivative with respect to x/D'
bw=ba['weights_dimensionless'];bn=ba['normals_body'];bg=ba['teacher_gradient_dimensionless'];boundary_count=len(bw);assert boundary_count>0 and min(bw)>0 and bg.shape==(boundary_count,2,2,4) and np.allclose(np.sum(bn*bn,axis=1),1,rtol=0,atol=1e-10)
bt=np.einsum('nijr,nj->nir',bg+bg.swapaxes(1,2),bn);traction_norms=np.einsum('n,nir,nir->r',bw,bt,bt);traction_den=float((sigma*sigma)@traction_norms);assert np.isfinite(traction_den) and traction_den>0 and c['traction_weight']==1
assert np.allclose(traction_norms,br['normalization']['mode_traction_norms_squared'],rtol=3e-12,atol=2e-14) and np.isclose(traction_den,br['normalization']['global_training_traction_energy'],rtol=3e-12,atol=2e-14);assert np.allclose(sigma*sigma*traction_norms/traction_den,br['normalization']['normalized_mode_energy_weights'],rtol=3e-12,atol=2e-14)
boundary_measure=float(bw.sum());boundary_replay=dict(points=boundary_count,measure_dimensionless=boundary_measure,teacher_mode_norms_squared=traction_norms.tolist(),global_teacher_energy=traction_den,normalized_mode_energy_weights=(sigma*sigma*traction_norms/traction_den).tolist(),exact_original_subset=True)
runtime=read(P/'RUNTIME.json',receipt['runtime_sha256']);assert runtime['backend']=='gpu' and runtime['x64'] and runtime['matmul_precision']=='highest'
assert final['status'] in ('job_deadline','gradient_stationary','line_search_stopped','iteration_limit') and not final['validation_seen'] and final['retention_errors']==[] and not final['all_intermediate_vectors_retained'] and not final['exact_optimizer_continuation_claim'];assert final['cumulative_adam_steps']==c['prior_adam_steps']==wm['actual_applied_adam_steps'];assert final['cumulative_lbfgs_steps']==c['prior_lbfgs_steps']+final['accepted_steps'];assert c['prior_lbfgs_steps']==wm['accepted_lbfgs_steps']
objective_gaps=[]
teacher_trace_reference=None
trace_summaries=[]
def objective(metrics,f):
 global teacher_trace_reference
 names=('value_relative_squared_by_mode','gradient_relative_squared_by_mode','divergence_error_relative_squared_by_mode','divergence_error_norm_squared_by_mode','predicted_divergence_norm_squared_by_mode','teacher_divergence_norm_squared_by_mode','predicted_divergence_relative_squared_by_mode','teacher_divergence_relative_squared_by_mode')
 a={k:np.asarray(metrics[k]) for k in names};assert all(v.shape==(4,) and np.isfinite(v).all() and v.min()>=0 for v in a.values());ng=np.asarray(pm['gradient_norms_squared'])
 for raw,norm in [('divergence_error_norm_squared_by_mode','divergence_error_relative_squared_by_mode'),('predicted_divergence_norm_squared_by_mode','predicted_divergence_relative_squared_by_mode'),('teacher_divergence_norm_squared_by_mode','teacher_divergence_relative_squared_by_mode')]:assert np.allclose(a[raw]/ng,a[norm],rtol=3e-12,atol=2e-14)
 base=float(weights@(a['value_relative_squared_by_mode']+a['gradient_relative_squared_by_mode']));added=float(weights@a['divergence_error_relative_squared_by_mode']);parent_combined=base+added
 terr=np.asarray(metrics['traction_error_norm_squared_by_mode']);tpred=np.asarray(metrics['traction_predicted_norm_squared_by_mode']);tteach=np.asarray(metrics['traction_teacher_norm_squared_by_mode']);integ=np.asarray(metrics['traction_integrated_traction_error_by_component_mode']);assert terr.shape==tpred.shape==tteach.shape==(4,) and integ.shape==(2,4) and all(np.isfinite(v).all() for v in (terr,tpred,tteach,integ)) and min(terr)>=0 and min(tpred)>=0
 assert np.allclose(tteach,traction_norms,rtol=3e-12,atol=2e-14);numerator=float((sigma*sigma)@terr);tr= numerator/traction_den;assert np.isclose(metrics['traction_weighted_numerator'],numerator,rtol=3e-12,atol=2e-14)
 assert np.all(np.sum(integ*integ,axis=0)<=boundary_measure*terr+2e-12*np.maximum(1,boundary_measure*terr));tcross=(tpred+tteach-terr)/2;assert np.all(abs(tcross)<=np.sqrt(tpred*tteach)+2e-12*np.maximum(1,tpred+tteach))
 for actual,expected in ((metrics['parent_trace_combined_objective'],parent_combined),(metrics['traction_objective_unweighted'],tr),(metrics['traction_objective_weighted'],tr)):assert np.isclose(actual,expected,rtol=3e-12,atol=2e-14)
 pn,tn,cn,dot=[float(metrics[k]) for k in ('parent_trace_gradient_l2','traction_gradient_l2_unweighted','combined_gradient_l2','parent_traction_gradient_dot_unweighted')];assert np.isfinite([pn,tn,cn,dot]).all() and min(pn,tn,cn)>=0 and abs(dot)<=pn*tn+2e-12*max(1,pn*tn);assert np.isclose(cn*cn,pn*pn+tn*tn+2*dot,rtol=3e-12,atol=2e-14)
 combined=parent_combined+tr
 for observed,expected in ((metrics['base_objective'],base),(metrics['supervised_divergence_objective_unweighted'],added),(metrics['supervised_divergence_objective_weighted'],added),(metrics['combined_objective'],combined),(f,combined)):assert np.isclose(observed,expected,rtol=3e-12,atol=2e-14)
 objective_gaps.append(abs(combined-f));teacher=a['teacher_divergence_norm_squared_by_mode']
 if teacher_trace_reference is None:teacher_trace_reference=teacher.copy()
 else:assert np.allclose(teacher,teacher_trace_reference,rtol=3e-12,atol=2e-14)
 predicted=a['predicted_divergence_norm_squared_by_mode'];error=a['divergence_error_norm_squared_by_mode'];gradient_error=a['gradient_relative_squared_by_mode']*ng;assert np.all(error<=2*gradient_error+2e-12*np.maximum(1,2*gradient_error));cross=(predicted+teacher-error)/2
 assert np.all(abs(cross)<=np.sqrt(predicted*teacher)+2e-12*np.maximum(1,predicted+teacher))
 return dict(base=base,added_trace=added,parent_trace_combined=parent_combined,added_traction=tr,traction_error_norm_squared=terr.tolist(),traction_predicted_norm_squared=tpred.tolist(),traction_teacher_norm_squared=tteach.tolist(),integrated_traction_error=integ.tolist(),combined=combined,predicted_trace_norm_squared=predicted.tolist(),teacher_trace_norm_squared=teacher.tolist(),trace_error_norm_squared=error.tolist(),inferred_predicted_teacher_trace_inner_product=cross.tolist())
checkpoints=final['checkpoints'];assert checkpoints[0]['label']=='initial' and checkpoints[0]['iteration']==0 and checkpoints[-1]['label']=='final' and checkpoints[-1]['iteration']==final['accepted_steps'];assert [r['iteration'] for r in checkpoints if r['label']=='periodic']==list(range(1000,final['accepted_steps']+1,1000))
semantic={};saved={}
for row in checkpoints:
 stem=f"state-{row['iteration']:05d}-{row['label']}";assert read(P/(stem+'.json'))==row and row['array_file']==stem+'.npz';a=arrays(P/row['array_file'],row['array_sha256']);n=len(row['pair_rho']);assert n<=10 and set(a)=={'theta','gradient'}|{k+str(i) for i in range(n) for k in ('s_','y_')};assert all(v.shape==ct.shape for v in a.values());assert digest(a['theta'])==row['theta_sha256'] and digest(a['gradient'])==row['gradient_sha256'];assert float(np.max(abs(a['gradient'])))==row['gradient_inf'];objective(row['metrics'],row['objective'])
 for i,rho in enumerate(row['pair_rho']):assert np.isclose(rho,1/float(a['s_'+str(i)]@a['y_'+str(i)]),rtol=2e-12) and rho>0
 semantic[stem]={k:digest(v) for k,v in a.items()};saved[stem]=a
initial=checkpoints[0];assert initial['theta_sha256']==digest(ct) and np.array_equal(saved['state-00000-initial']['theta'],ct) and initial['full_evaluations']==2
cache=read(P/'DEVICE-CACHE.json');eq=read(P/'INITIAL-CACHE-EQUIVALENCE.json');ea=arrays(P/'initial-streamed-cached-equivalence.npz',eq['arrays_sha256'])
assert set(ea)=={'theta','streamed_objective','cached_objective','streamed_gradient','cached_gradient'} and np.array_equal(ea['theta'],ct)
assert all(ea[k].shape==() for k in ('streamed_objective','cached_objective')) and all(ea[k].shape==ct.shape for k in ('streamed_gradient','cached_gradient'))
assert np.array_equal(ea['cached_gradient'],saved['state-00000-initial']['gradient']) and float(ea['cached_objective'])==initial['objective']
assert eq['passed'] and eq['tolerance']==1e-10 and eq['parent_parameters_in_optimizer'] is False and eq['child_parameters']==22792
sgap=float(abs(ea['streamed_gradient']-ea['cached_gradient']).max());fgap=float(abs(ea['streamed_objective']-ea['cached_objective']))
assert sgap==eq['gradient_max_abs_gap'] and fgap==eq['objective_gap'] and sgap<=1e-10 and fgap<=1e-10*max(1,abs(float(ea['streamed_objective'])))
assert eq['cached_metrics']==initial['metrics'];objective(eq['streamed_metrics'],float(ea['streamed_objective']))
keys=set(eq['streamed_metrics'])-{'seconds'};assert set(eq['component_equivalence'])==keys and all(eq['component_equivalence'].values())
for k in keys:assert np.allclose(eq['streamed_metrics'][k],eq['cached_metrics'][k],rtol=1e-10,atol=1e-10)
assert cache['parent_sha256']==c['parent_bank_sha256'] and cache['parent_theta_sha256']==digest(pt) and cache['teacher_identity']==t and cache['boundary_teacher']==br
expected_ranges=lambda n:[[i,min(n,i+c['config']['block_points'])] for i in range(0,n,c['config']['block_points'])]
assert cache['volume_ranges']==expected_ranges(1036497) and cache['boundary_ranges']==expected_ranges(boundary_count)
expected_payload=(1036497*51+boundary_count*37)*8
assert cache['logical_payload_bytes']==expected_payload<=cache['payload_limit_bytes']==2**30 and cache['payload_is_not_peak_memory_bound'] is True
for k in ('parent_volume_values_sha256','parent_volume_physical_gradient_sha256','parent_boundary_physical_gradient_sha256'):assert len(cache[k])==64 and all(ch in '0123456789abcdef' for ch in cache[k])
cache_replay={'initial_objective_gap':fgap,'initial_child_gradient_max_gap':sgap,'logical_payload_bytes':expected_payload,'parent_cache_hashes':{k:cache[k] for k in cache if k.startswith('parent_')},'scope':'Saved equivalence arrays and cache manifest checked; parent cache fields not independently evaluated.'}
igr=read(P/'INITIAL-COMPONENT-GRADIENTS.json');iga=arrays(P/'initial-component-gradients.npz',igr['arrays_sha256']);assert set(iga)=={'base','trace','traction_unweighted'} and all(v.shape==ct.shape for v in iga.values()) and igr['theta_sha256']==initial['theta_sha256']
pg=iga['base']+iga['trace'];cg=pg+iga['traction_unweighted'];assert np.allclose(cg,ea['streamed_gradient'],rtol=3e-12,atol=2e-14)
for key,v in [('base_gradient_l2',iga['base']),('trace_gradient_l2',iga['trace']),('traction_gradient_l2_unweighted',iga['traction_unweighted']),('traction_gradient_l2_weighted',iga['traction_unweighted']),('parent_combined_gradient_l2',pg),('new_combined_gradient_l2',cg)]:assert np.isclose(igr[key],np.linalg.norm(v),rtol=3e-12,atol=2e-14)
assert np.isclose(eq['streamed_metrics']['parent_traction_gradient_dot_unweighted'],pg@iga['traction_unweighted'],rtol=3e-12,atol=2e-14)
component_gradient_replay=dict(array_hashes={k:digest(v) for k,v in iga.items()},norms={k:v for k,v in igr.items() if k.endswith('_l2') or 'gradient_l2_' in k},initial_sum_max_gap=float(abs(cg-ea['streamed_gradient']).max()),scope='Initial saved component vectors only; no fresh autodiff or intermediate component-vector reconstruction.')
base_replay=read(P/'PARENT-BASE-REPLAY.json');assert base_replay['passed'] and base_replay['objective_changed_from_parent'] is False and base_replay['cross_objective_reduction_claim'] is False and base_replay['parent_objective']==c['parent_full_training_objective']
assert base_replay['metrics']==eq['streamed_metrics'] and base_replay['initial_combined_objective']==float(ea['streamed_objective']) and base_replay['initial_base_objective']==eq['streamed_metrics']['base_objective'] and base_replay['initial_parent_trace_combined_objective']==eq['streamed_metrics']['parent_trace_combined_objective']
assert abs(base_replay['initial_combined_objective']-c['parent_full_training_objective'])<=1e-10*max(1,abs(base_replay['initial_combined_objective']))
trials=lines('LINE-SEARCH.jsonl');accepted=lines('ACCEPTED.jsonl');assert len(accepted)==final['accepted_steps'];prior=initial;count=2;groups={};history_size=0;history_sizes={0:0}
for row in trials:
 count+=1;assert row['full_evaluations']==count;assert row['step']==2.**(-row['attempt']) and row['slope']<0 and np.isfinite(row['direction_norm']) and row['direction_norm']>0;assert row['accepted']==bool(row['objective']<=row['prior_objective']+1e-4*row['step']*row['slope']);objective(row['metrics'],row['objective']);groups.setdefault(row['iteration'],[]).append(row)
assert sorted(groups)==list(range(len(groups)))
def coordinate_witness(cm):
 assert set(cm)=={'H0_gamma_z','slope_physical','direction_reset','physical_gradient_inf','scaled_gradient_inf_diagnostic','physical_stopping_required'}
 finite_tree(cm);assert cm['H0_gamma_z']>0 and cm['slope_physical']<0 and type(cm['direction_reset']) is bool and cm['physical_stopping_required'] is True
 assert cm['physical_gradient_inf']>0 and cm['physical_gradient_inf']<=cm['scaled_gradient_inf_diagnostic']<=100*cm['physical_gradient_inf']*(1+1e-14)
for row in trials:
 cm0=row['coordinate_metrics'];coordinate_witness(cm0);assert cm0['slope_physical']==row['slope'];coordinate_replay['completed_trial_witnesses']+=1
for i,rows0 in groups.items():
 assert all(r['coordinate_metrics']==rows0[0]['coordinate_metrics'] for r in rows0)
 prior_inf=initial['gradient_inf'] if i==0 else accepted[i-1]['gradient_inf'];assert rows0[0]['coordinate_metrics']['physical_gradient_inf']==prior_inf
 if rows0[-1]['accepted']:assert accepted[i]['coordinate_metrics']==rows0[-1]['coordinate_metrics'] and accepted[i]['direction_reset']==rows0[-1]['coordinate_metrics']['direction_reset']
for cp in checkpoints:
 stem=f"state-{cp['iteration']:05d}-{cp['label']}";a=saved[stem];g=a['gradient'];witness={'iteration':cp['iteration'],'label':cp['label'],'physical_gradient_inf':float(abs(g).max()),'coordinate_gradient_inf':float(abs(D*g).max()),'coordinate_displacement_l2':float(np.linalg.norm((a['theta']-ct)/D)),'curvature_coordinate_inner_product_gaps':[]}
 for k in range(len(cp['pair_rho'])):
  sp,yp=a['s_'+str(k)],a['y_'+str(k)];gap=abs(float(sp@yp)-float((sp/D)@(D*yp)));assert np.isclose(sp@yp,(sp/D)@(D*yp),rtol=1e-12,atol=1e-25);witness['curvature_coordinate_inner_product_gaps'].append(gap)
 if cp['iteration'] in groups:
  cm0=groups[cp['iteration']][0]['coordinate_metrics'];assert np.isclose(cm0['physical_gradient_inf'],witness['physical_gradient_inf'],rtol=1e-14) and np.isclose(cm0['scaled_gradient_inf_diagnostic'],witness['coordinate_gradient_inf'],rtol=1e-14)
 coordinate_replay['checkpoint_witnesses'].append(witness)

for i,rows in groups.items():
 assert [r['attempt'] for r in rows]==list(range(len(rows))) and len(rows)<=20 and sum(r['accepted'] for r in rows)<=1 and not any(r['accepted'] for r in rows[:-1]);assert all(r['prior_objective']==prior['objective'] and r['slope']==rows[0]['slope'] and r['direction_sha256']==rows[0]['direction_sha256'] and r['cumulative_accepted_lbfgs_steps']==c['prior_lbfgs_steps']+i for r in rows)
 if rows[-1]['accepted']:
  r=rows[-1];a=accepted[i];assert a['iteration']==i+1 and a['prior_theta_sha256']==prior['theta_sha256'] and a['prior_gradient_sha256']==prior['gradient_sha256'] and a['prior_objective']==prior['objective'];assert a['full_post_update_objective']==r['objective'] and a['theta_sha256']==r['theta_sha256'] and a['gradient_sha256']==r['gradient_sha256'];assert all(a[k]==r[k] for k in ('step','attempt','slope','direction_sha256','direction_norm','gradient_inf','metrics','full_evaluations'));assert a['cumulative_lbfgs_steps']==c['prior_lbfgs_steps']+i+1;assert a['curvature_kept']==bool(a['curvature_sy']>a['curvature_threshold']);assert np.isclose(a['curvature_threshold'],1e-12*np.sqrt(a['curvature_ss']*a['curvature_yy']),rtol=3e-12,atol=1e-30);assert 0<=a['history_size']<=10;history_size=min(10,(0 if a['direction_reset'] else history_size)+int(a['curvature_kept']));assert a['history_size']==history_size;history_sizes[i+1]=history_size;prior=dict(objective=r['objective'],theta_sha256=r['theta_sha256'],gradient_sha256=r['gradient_sha256'])
 else:assert i==len(accepted) and i==max(groups) and final['status'] in ('job_deadline','line_search_stopped')
assert count==final['full_evaluations'];last=checkpoints[-1];assert all(last[k]==prior[k] for k in ('objective','theta_sha256','gradient_sha256'));assert last['objective']==min([initial['objective']]+[a['full_post_update_objective'] for a in accepted]);assert receipt['selected_checkpoint']=={'objective':last['objective'],'checkpoint':last['array_file']} and receipt['selected_theta_sha256']==last['theta_sha256'] and receipt['config_sha256']==provenance[str(P/'CONFIG.json')]
for row in checkpoints:
 expected_history=history_sizes[row['iteration']]
 if row['label']=='final' and final['status'] in ('job_deadline','line_search_stopped'):assert len(row['pair_rho']) in (expected_history,0)
 else:assert len(row['pair_rho'])==expected_history
 if row['iteration']:
  a=accepted[row['iteration']-1];assert row['theta_sha256']==a['theta_sha256'] and row['gradient_sha256']==a['gradient_sha256'] and row['objective']==a['full_post_update_objective']
model=P/receipt['bank_file'];check(model,receipt['bank_sha256'])
with np.load(model,allow_pickle=False) as z:
 md=json.loads(str(z['metadata_json']));finite_tree(md);assert set(z.files)=={'metadata_json','embedded_parent_npz'}|{f'head{i}_{k}{j}' for i in range(4) for j in range(3) for k in ('W','b')}
 raw=z['embedded_parent_npz'];assert raw.dtype==np.uint8 and raw.ndim==1 and raw.tobytes()==parent.read_bytes() and hashlib.sha256(raw.tobytes()).hexdigest()==c['parent_bank_sha256']
 aa=[]
 for i in range(4):
  for j,shape in enumerate(((21,64),(64,64),(64,2))):
   w0,b0=z[f'head{i}_W{j}'],z[f'head{i}_b{j}'];assert w0.shape==shape and b0.shape==(shape[1],) and w0.dtype==b0.dtype==np.float64 and np.isfinite(w0).all() and np.isfinite(b0).all();aa.extend((w0,b0))
 theta=np.concatenate([v.ravel() for v in aa])
assert md['schema']=='ns-obstacle-independent-residual-head-bank-v1' and md['head_architecture']==[21,64,64,2] and md['head_count']==4 and md['parent_sha256']==c['parent_bank_sha256'] and md['FE_fallback'] is False
assert np.array_equal(theta,saved[f"state-{last['iteration']:05d}-final"]['theta']) and md['objective_variant']==receipt['objective_variant']=='fixed_coordinate_residual_heads_same_trace_and_traction'
assert receipt['selection']=='minimum accepted combined objective' and receipt['selected_base_objective']==last['metrics']['base_objective'] and receipt['selected_theta_scope']=='child parameters only' and receipt['parent_model_sha256']==c['parent_bank_sha256'] and receipt['optimizer_counts_scope']=='warm child restart with reset curvature; historical lineage retained' and receipt['warm_restart'] is True and receipt['warm_child_model_sha256']==c['warm_child_model_sha256']
assert md['refinement']==c and md['new_adam_steps']==0 and md['new_lbfgs_steps']==len(accepted) and md['selected_checkpoint']==receipt['selected_checkpoint'] and md['head_seed']==c['protocol']['head_seed']
assert md['actual_applied_adam_steps']==receipt['cumulative_adam_steps']==final['cumulative_adam_steps'] and md['accepted_lbfgs_steps']==receipt['cumulative_lbfgs_steps']==final['cumulative_lbfgs_steps'] and md['status']==final['status']
for k in ('features','envelope','geometry','teacher_identity','config','value_norms_squared','gradient_norms_squared','mode_weights','value_units','gradient_coordinates'):assert md[k]==pm[k]
replay=read(P/'SELECTED-FULL-REPLAY.json',receipt['selected_full_replay_sha256']);assert replay['passed'] and replay['selected_model_sha256']==receipt['bank_sha256'] and replay['selected_theta_sha256']==digest(theta);objective(replay['metrics'],replay['objective']);assert np.isclose(replay['objective_gap'],abs(replay['objective']-last['objective']),rtol=1e-12,atol=1e-16);assert replay['objective_gap']<=1e-10*max(1,abs(last['objective'])) and replay['gradient_max_gap']<=1e-10;assert receipt['total_full_evaluations_including_serialized_replay']==count+1 and receipt['physical_admission'] is False and receipt['optimizer_status']==final['status']
ra=arrays(P/'selected-uncached-full-replay.npz',replay['raw_replay_arrays_sha256']);assert set(ra)=={'selected_theta','cached_objective','uncached_objective','cached_gradient','uncached_gradient'}
assert np.array_equal(ra['selected_theta'],theta) and np.array_equal(ra['cached_gradient'],saved[f"state-{last['iteration']:05d}-final"]['gradient']) and ra['uncached_gradient'].shape==theta.shape
assert ra['cached_objective'].shape==ra['uncached_objective'].shape==() and float(ra['cached_objective'])==last['objective'] and float(ra['uncached_objective'])==replay['objective']
assert float(abs(ra['uncached_gradient']-ra['cached_gradient']).max())==replay['gradient_max_gap']
assert set(replay['component_replay'])==set(last['metrics'])-{'seconds'} and all(replay['component_replay'].values())
for k in replay['component_replay']:assert np.allclose(last['metrics'][k],replay['metrics'][k],rtol=1e-10,atol=1e-10)
cache_replay['final_uncached_objective_gap']=replay['objective_gap'];cache_replay['final_uncached_child_gradient_max_gap']=replay['gradient_max_gap']
events=lines('EVALUATIONS.jsonl');byid={};completed=[];incomplete=[];pending={};completed_counter=0
identity_keys=('evaluation_id','phase','theta_sha256','accepted_steps_before','completed_full_evaluations_before','elapsed_seconds','cached_teacher_and_parent')
for event in events:
 eid=event['evaluation_id'];assert isinstance(eid,int) and eid>=1
 if event['status']=='pending':
  assert eid==len(byid)+1 and not pending and event['completed_full_evaluations_before']==completed_counter
  byid[eid]=[event];pending[eid]=event
 else:
  assert eid in pending and event['status'] in ('completed','incomplete');first=pending.pop(eid);assert all(event[k]==first[k] for k in identity_keys);byid[eid].append(event);assert event['completed_blocks']>=0
  if event['completed_boundary_blocks']>0:assert event['completed_blocks']==(1036497+c['config']['block_points']-1)//c['config']['block_points']
  if event['status']=='completed':
   completed_counter+=1;assert event['full_evaluations']==completed_counter and event['completed_blocks']==(1036497+c['config']['block_points']-1)//c['config']['block_points'] and event['completed_boundary_blocks']==(boundary_count+c['config']['block_points']-1)//c['config']['block_points'];completed.append(event)
  else:
   assert 0<=event['completed_blocks']<=(1036497+c['config']['block_points']-1)//c['config']['block_points'] and 0<=event['completed_boundary_blocks']<=(boundary_count+c['config']['block_points']-1)//c['config']['block_points']
   assert event['full_objective_available'] is False and 'objective' not in event;fn=Path(event['exception_record_file']);assert fn.name==str(fn);failure=read(P/fn,event['exception_record_sha256'])
   assert failure['evaluation_identity']=={k:first[k] for k in identity_keys};incomplete.append(event)
assert not pending and len(byid)==final['initiated_evaluations']+1 and completed_counter==count+1 and len(incomplete)==len(byid)-completed_counter
assert completed[0]['phase']=='initial_parent' and completed[1]['phase']=='initial_cached' and completed[-1]['phase']=='serialized_selected_uncached' and all(e['phase']=='line_search' for e in completed[2:-1])
assert completed[0]['cached_teacher_and_parent'] is False and completed[-1]['cached_teacher_and_parent'] is False and all(e['cached_teacher_and_parent'] is True for e in completed[1:-1])
assert len(completed[2:-1])==len(trials)
assert completed[0]['theta_sha256']==digest(ct) and completed[0]['objective']==float(ea['streamed_objective']) and completed[0]['gradient_sha256']==digest(ea['streamed_gradient']) and completed[0]['gradient_inf']==float(abs(ea['streamed_gradient']).max())
for event,row in zip(completed[1:-1],[initial]+trials):
 for key in ('objective','theta_sha256','gradient_sha256','gradient_inf'):assert event[key]==row[key]
 if 'full_evaluations' in row:assert event['full_evaluations']==row['full_evaluations']
assert completed[-1]['objective']==replay['objective'] and completed[-1]['theta_sha256']==replay['selected_theta_sha256'] and completed[-1]['gradient_sha256']==digest(ra['uncached_gradient']) and completed[-1]['gradient_inf']==float(abs(ra['uncached_gradient']).max())
for event,row in zip(completed[2:-1],trials):assert event['accepted_steps_before']==row['iteration']
assert completed[0]['accepted_steps_before']==completed[1]['accepted_steps_before']==0 and completed[-1]['accepted_steps_before']==len(accepted)
if incomplete:
 assert final['status']=='job_deadline' and len(incomplete)==1 and incomplete[0]['phase']=='line_search' and incomplete[0]['accepted_steps_before']==len(accepted) and incomplete[0]['evaluation_id']==len(byid)-1
initial_components=objective(initial['metrics'],initial['objective']);selected_components=objective(last['metrics'],last['objective']);replay_components=objective(replay['metrics'],replay['objective'])
if final['status']=='line_search_stopped':assert len(groups[len(accepted)])==20 and not any(r['accepted'] for r in groups[len(accepted)])
if final['status']=='gradient_stationary':assert last['gradient_inf']<=1e-10
if final['status']=='iteration_limit':assert len(accepted)==10000
rejected_better=[{k:r[k] for k in ('iteration','attempt','objective','theta_sha256','gradient_sha256')} for r in trials if not r['accepted'] and r['objective']<last['objective']]
signal.alarm(0)
print(json.dumps(dict(schema='ns-obstacle-fixed-coordinate-residual-refinement-saved-audit-v1',passed=True,physical_admission=False,status=final['status'],objective_variant='fixed_coordinate_residual_heads_same_trace_and_traction',coordinate_replay=coordinate_replay,warm_child_model_sha256=c['warm_child_model_sha256'],cache_replay=cache_replay,parent_model_sha256=c['parent_bank_sha256'],selected_theta_scope='child parameters only',trainable_parameters=22792,coefficient_replay=coefficient_replay,boundary_replay=boundary_replay,component_gradient_replay=component_gradient_replay,initial_components=initial_components,selected_components=selected_components,serialized_replay_components=replay_components,evaluation_events=dict(initiated=len(byid),completed=completed_counter,incomplete=incomplete,pending_unclosed=len(pending)),accepted_steps=len(accepted),completed_trial_evaluations=len(trials),full_evaluations_including_replay=count+1,retained_checkpoints=len(checkpoints),terminal_curvature_reset_unresolved=bool(len(last['pair_rho'])!=history_sizes[last['iteration']]),initial_objective=initial['objective'],selected_objective=last['objective'],selection_semantics='last accepted state, minimum among initial and accepted COMBINED objective states',original_selection_string=c['selection'],completed_rejected_trials_better_than_final=rejected_better,selected_model_sha256=receipt['bank_sha256'],selected_theta_sha256=digest(theta),saved_modal_aggregation_max_gap=max(objective_gaps),checkpoint_semantic_arrays=semantic,teacher_identity=t,provenance=provenance,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=['No neural evaluation, training, native solve or validation access.', 'Warm child identity and retained-array algebra replayed; no fresh JAX objective/autodiff.', 'Embedded parent bytes checked exactly; final replay is saved JAX child-gradient evidence, not independent NumPy live inference qualification.', 'Optimizer cumulative counts represent lineage across different parameter spaces, not exact continuation.','Intermediate vectors are absent: scalar hash/Armijo chains checked, not independent intermediate vector or two-loop replay.','Saved full teacher objective/gradient replay receipt checked, not independently recomputed.','Pending/incomplete evaluation events are retained and checked; interrupted partial sums are not full measured objectives.','A terminal unaccepted or interrupted first trial may reset curvature without a retained reset scalar; final empty history is allowed but its cause remains unresolved.','Periodic/final curvature arrays checked internally; correspondence to missing intermediate vectors cannot be independently reconstructed.']),indent=2,allow_nan=False))
