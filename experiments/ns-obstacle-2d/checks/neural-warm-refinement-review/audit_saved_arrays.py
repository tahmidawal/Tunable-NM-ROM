"""Audit completed compact warm refinement; BINDING supplied only after terminal retention."""
from pathlib import Path
import hashlib,json,time,resource,signal
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
assert c['schema']=='ns-obstacle-compact-warm-LBFGS-v1' and c['protocol']=={'max_iterations':10000,'checkpoint_every':250,'history_size':10,'line_search_steps':20,'gradient_tolerance':1e-10,'warm_restart':True,'new_adam_steps':0}
assert c['protocol']==plan['refinement_protocol']==release['refinement_protocol']
assert c['warm_restart'] and c['curvature_reset'] and not c['exact_optimizer_continuation_claim'] and not c['all_intermediate_vectors_retained'];check(B/'accuracy_v2/refine.py',c['source_sha256']);assert c['source_sha256']==BINDING['trainer_sha256']
parent=Path(BINDING['parent_bank']);check(parent,c['parent_bank_sha256']);pr=read(Path(BINDING['parent_receipt']),c['parent_receipt_sha256']);review=read(Path(BINDING['parent_review']),c['parent_review_sha256']);assert review['passed'] and review['model_sha256']==c['parent_bank_sha256'] and review['model_receipt_sha256']==c['parent_receipt_sha256'];assert pr['selected_theta_sha256']==c['parent_selected_theta_sha256'] and pr['selected_checkpoint']['objective']==c['parent_full_training_objective']
assert c['prior_allocation_seconds']==review['allocated_seconds'] and c['prior_adam_steps']==review['adam_steps_recorded'] and c['prior_lbfgs_steps']==review['accepted_lbfgs_steps']
def bank(p,h):
 check(p,h)
 with np.load(p,allow_pickle=False) as z:
  md=json.loads(str(z['metadata_json']));finite_tree(md);assert np.array_equal(z['transform'],np.eye(4));n=md['layers'];assert set(z.files)=={'metadata_json','transform'}|{k+str(i) for i in range(n) for k in ('W','b')};a=[z[k+str(i)] for i in range(n) for k in ('W','b')];assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in a);theta=np.concatenate([v.ravel() for v in a])
 return md,theta
pm,pt=bank(parent,c['parent_bank_sha256']);assert digest(pt)==c['parent_selected_theta_sha256'];t=c['teacher_identity'];assert t==pm['teacher_identity'] and c['config']==pm['config'] and t['split']=='training' and t['Re']==[15+1.25*i for i in range(9)]
assert t['reference_admission_sha256']=='5c9402af9ddb2af0ad8925ef224a43ae72cb36209df99865908617c63b5b506f';assert t['prepared_manifest_sha256']=='7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6';assert t['volume_manifest_sha256']=='21563bc7a114147716d300e8f1981e664da5ae728db6b1aa21ee3aa8bd947866';assert t['training_array_sha256']=='6a65cbcd387c32b9ab70fc3956335a975e40b62646370fdf3413c7df4bfd8f30'
weights=np.asarray(pm['mode_weights']);sigma=np.asarray(t['singular_values']);expected=np.sqrt(sigma/sigma[0]);expected/=expected.sum();assert np.allclose(weights,expected,rtol=1e-15,atol=0)
for k in ('value_norms_squared','gradient_norms_squared'):assert np.asarray(pm[k]).shape==(4,) and np.isfinite(pm[k]).all() and min(pm[k])>0
runtime=read(P/'RUNTIME.json',receipt['runtime_sha256']);assert runtime['backend']=='gpu' and runtime['x64'] and runtime['matmul_precision']=='highest'
assert final['status'] in ('job_deadline','gradient_stationary','line_search_stopped','iteration_limit') and not final['validation_seen'] and final['retention_errors']==[] and not final['all_intermediate_vectors_retained'] and not final['exact_optimizer_continuation_claim'];assert final['cumulative_adam_steps']==c['prior_adam_steps']==pm['actual_applied_adam_steps'];assert final['cumulative_lbfgs_steps']==c['prior_lbfgs_steps']+final['accepted_steps'];assert c['prior_lbfgs_steps']==pm['accepted_lbfgs_steps']
objective_gaps=[]
def objective(metrics,f):
 ev=np.asarray(metrics['value_relative_squared_by_mode']);eg=np.asarray(metrics['gradient_relative_squared_by_mode']);assert ev.shape==eg.shape==(4,) and np.isfinite(ev).all() and np.isfinite(eg).all() and min(ev.min(),eg.min())>=0;v=float(weights@(ev+eg));assert np.isclose(v,f,rtol=3e-12,atol=2e-14);objective_gaps.append(abs(v-f))
checkpoints=final['checkpoints'];assert checkpoints[0]['label']=='initial' and checkpoints[0]['iteration']==0 and checkpoints[-1]['label']=='final' and checkpoints[-1]['iteration']==final['accepted_steps'];assert [r['iteration'] for r in checkpoints if r['label']=='periodic']==list(range(250,final['accepted_steps']+1,250))
semantic={};saved={}
for row in checkpoints:
 stem=f"state-{row['iteration']:05d}-{row['label']}";assert read(P/(stem+'.json'))==row and row['array_file']==stem+'.npz';a=arrays(P/row['array_file'],row['array_sha256']);n=len(row['pair_rho']);assert n<=10 and set(a)=={'theta','gradient'}|{k+str(i) for i in range(n) for k in ('s_','y_')};assert all(v.shape==pt.shape for v in a.values());assert digest(a['theta'])==row['theta_sha256'] and digest(a['gradient'])==row['gradient_sha256'];assert float(np.max(abs(a['gradient'])))==row['gradient_inf'];objective(row['metrics'],row['objective'])
 for i,rho in enumerate(row['pair_rho']):assert np.isclose(rho,1/float(a['s_'+str(i)]@a['y_'+str(i)]),rtol=2e-12) and rho>0
 semantic[stem]={k:digest(v) for k,v in a.items()};saved[stem]=a
initial=checkpoints[0];assert initial['theta_sha256']==digest(pt) and abs(initial['objective']-c['parent_full_training_objective'])<=1e-10*max(1,abs(initial['objective']))
trials=lines('LINE-SEARCH.jsonl');accepted=lines('ACCEPTED.jsonl');assert len(accepted)==final['accepted_steps'];prior=initial;count=1;groups={};history_size=0;history_sizes={0:0}
for row in trials:
 count+=1;assert row['full_evaluations']==count;assert row['step']==2.**(-row['attempt']) and row['slope']<0 and np.isfinite(row['direction_norm']) and row['direction_norm']>0;assert row['accepted']==bool(row['objective']<=row['prior_objective']+1e-4*row['step']*row['slope']);objective(row['metrics'],row['objective']);groups.setdefault(row['iteration'],[]).append(row)
assert sorted(groups)==list(range(len(groups)))
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
md,theta=bank(P/receipt['bank_file'],receipt['bank_sha256']);assert np.array_equal(theta,saved[f"state-{last['iteration']:05d}-final"]['theta']);assert md['refinement']==c and md['new_adam_steps']==0 and md['new_lbfgs_steps']==len(accepted) and md['selected_checkpoint']==receipt['selected_checkpoint']
assert md['actual_applied_adam_steps']==receipt['cumulative_adam_steps']==final['cumulative_adam_steps'] and md['accepted_lbfgs_steps']==receipt['cumulative_lbfgs_steps']==final['cumulative_lbfgs_steps'] and md['status']==final['status']
for k in ('schema','features','envelope','geometry','layers','teacher_identity','config','value_norms_squared','gradient_norms_squared','mode_weights','value_units','gradient_coordinates'):assert md[k]==pm[k]
replay=read(P/'SELECTED-FULL-REPLAY.json',receipt['selected_full_replay_sha256']);assert replay['passed'] and replay['selected_model_sha256']==receipt['bank_sha256'] and replay['selected_theta_sha256']==digest(theta);objective(replay['metrics'],replay['objective']);assert np.isclose(replay['objective_gap'],abs(replay['objective']-last['objective']),rtol=1e-12,atol=1e-16);assert replay['objective_gap']<=1e-10*max(1,abs(last['objective'])) and replay['gradient_max_gap']<=1e-10;assert receipt['total_full_evaluations_including_serialized_replay']==count+1 and receipt['physical_admission'] is False and receipt['optimizer_status']==final['status']
if final['status']=='line_search_stopped':assert len(groups[len(accepted)])==20 and not any(r['accepted'] for r in groups[len(accepted)])
if final['status']=='gradient_stationary':assert last['gradient_inf']<=1e-10
if final['status']=='iteration_limit':assert len(accepted)==10000
rejected_better=[{k:r[k] for k in ('iteration','attempt','objective','theta_sha256','gradient_sha256')} for r in trials if not r['accepted'] and r['objective']<last['objective']]
signal.alarm(0)
print(json.dumps(dict(schema='ns-obstacle-compact-warm-refinement-saved-audit-v1',passed=True,physical_admission=False,status=final['status'],accepted_steps=len(accepted),completed_trial_evaluations=len(trials),full_evaluations_including_replay=count+1,retained_checkpoints=len(checkpoints),terminal_curvature_reset_unresolved=bool(len(last['pair_rho'])!=history_sizes[last['iteration']]),initial_objective=initial['objective'],selected_objective=last['objective'],selection_semantics='last accepted state, minimum among initial and accepted states',original_selection_string=c['selection'],completed_rejected_trials_better_than_final=rejected_better,selected_model_sha256=receipt['bank_sha256'],selected_theta_sha256=digest(theta),saved_modal_aggregation_max_gap=max(objective_gaps),checkpoint_semantic_arrays=semantic,teacher_identity=t,provenance=provenance,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=['No neural evaluation, training, native solve or validation access.','Intermediate vectors are absent: scalar hash/Armijo chains checked, not independent intermediate vector or two-loop replay.','Saved full teacher objective/gradient replay receipt checked, not independently recomputed.','Deadline may interrupt an unrecorded partial full evaluation; only completed counted evaluations are audited.','A terminal unaccepted or interrupted first trial may reset curvature without a retained reset scalar; final empty history is allowed but its cause remains unresolved.','Periodic/final curvature arrays checked internally; correspondence to missing intermediate vectors cannot be independently reconstructed.']),indent=2,allow_nan=False))
