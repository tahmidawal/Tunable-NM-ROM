"""Bounded read-only optimizer/model closure audit, without teacher/JAX execution."""
from pathlib import Path
import hashlib,json,time,resource,signal
import numpy as np
start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('audit115s')));signal.alarm(115)
B=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided01');P=B/'results/fit';provenance={};ledger={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
 actual=sha(p)
 if h is not None:assert actual==h,(str(p),actual,h)
 if p.is_relative_to(B) and p!=B/'COLLECTION-SHA256SUMS':
  member=str(p.relative_to(B));assert member in ledger,member;assert actual==ledger[member]
 provenance[str(p)]=actual;return actual
def read(p,h=None):check(p,h);return json.loads(p.read_text())
def digest(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()
def arrays(p,h):
 check(p,h)
 with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 assert all(x.dtype==np.float64 and np.isfinite(x).all() for x in a.values());return a
check(B/'COLLECTION-SHA256SUMS','c024356a90c79ff4e8014383917050a721921ebee2542fe1818e251a4acc085e')
for line in (B/'COLLECTION-SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);n=n.removeprefix('./');assert n not in ledger;ledger[n]=h
release=read(B/'TRAINING-RELEASE.json','f2efa38f710c764152b7f03bd3859e5342f126f85999faf3b909bb541338ca98');assert release['authorized'] and not release['validation_allowed'] and not release['resume_allowed'] and not release['new_reference_solves_allowed']
plan=read(B/'PLAN.json',release['plan_sha256']);check(B/'RUNTIME-INVENTORY.json',release['runtime_inventory_sha256']);check(B/'RESOURCE-CHECK.json',release['resource_check_sha256']);pre=read(B/'PREFLIGHT.json');assert pre['release_sha256']==provenance[str(B/'TRAINING-RELEASE.json')] and pre['actual_full_preflight_passed'] and pre['actual_nine_training_teacher_preparation_verified']
c=read(P/'CONFIG.json','120fb11cafdd3ce2700e2c263f1737a0d3200b22db0a5741076fccbcff8289a0');cfg=c['config'];assert cfg==plan['config']==read(B/'CONFIG.json',plan['config_sha256']);assert cfg['split']=='training' and cfg['rank']==4 and cfg['seed']==29102026 and cfg['adam_steps']==4000 and cfg['lbfgs_steps']==300
for n,h in c['source_files'].items():check(B/'accuracy_v2'/n,h)
t=c['teacher_identity'];assert t['split']=='training' and t['Re']==[15+1.25*i for i in range(9)] and t['reference_qualification']==c['qualification']==release['reference_qualification']==plan['reference_qualification'];assert t['reference_admission_sha256']==release['reference_admission_sha256']=='5c9402af9ddb2af0ad8925ef224a43ae72cb36209df99865908617c63b5b506f'
assert t['prepared_manifest_sha256']==plan['prepared_manifest_sha256']=='7867c6428814e4ecf790bb59f624abcea10fef18b60c2b238fa6c4dffac6d4f6';assert t['volume_manifest_sha256']=='21563bc7a114147716d300e8f1981e664da5ae728db6b1aa21ee3aa8bd947866';assert t['training_array_sha256']=='6a65cbcd387c32b9ab70fc3956335a975e40b62646370fdf3413c7df4bfd8f30';assert c['inherited_loader_source_sha256']==release['pilot_source_sha256'];assert t['arrays']=={k:c[k] for k in ('points','weights','values','gradients')}
prepared=Path(plan['inputs']['prepared']['path']);pm=read(prepared/'manifest.json',t['prepared_manifest_sha256']);check(prepared/'volume/manifest.json',t['volume_manifest_sha256']);assert pm['training_array_sha256']==t['training_array_sha256'] and pm['Re']==t['Re'] and pm['split']=='training'
sigma=np.asarray(c['singular_values']);assert np.array_equal(sigma,np.asarray(t['singular_values'])) and np.array_equal(sigma,np.asarray(plan['teacher_singular_values']));weights=np.sqrt(sigma/sigma[0]);weights/=weights.sum();assert np.array_equal(weights,np.asarray(c['mode_weights']))
assert all(np.asarray(c[k]).shape==(4,) and np.min(c[k])>0 for k in ('value_norms_squared','gradient_norms_squared'))
receipt=read(P/'MODEL-RECEIPT.json','470ba0ef5fa43bb59980cd1affcf51474d97f427d3e4f6a944411265db868285');final=read(P/'FINAL.json','51d4fa434fe169ccf82ddde415cb6d8fb8b1b3604b87549b59d71eae26587b90');runtime=read(P/'RUNTIME.json',receipt['runtime_sha256']);assert runtime['backend']=='gpu' and runtime['x64'] and runtime['matmul_precision']=='highest' and runtime['device_float_dtype']=='float64' and set(runtime['input_dtypes'])=={'float64'} and all(d['platform']=='gpu' for d in runtime['devices'])
assert not final['validation_seen'] and not final['automatic_resume'] and final['retention_errors']==[] and final['status']=='iteration_limit' and final['full_evaluations']==338
assert receipt['config_sha256']==provenance[str(P/'CONFIG.json')] and receipt['physical_admission'] is False and receipt['ordering_applied'] is False
records={};data={};semantic_arrays={};objective_gap=0.
for i in range(1,323):
 stem=f'checkpoint-{i:05d}';j=read(P/(stem+'.json'));assert j['array_file']==stem+'.npz';a=arrays(P/j['array_file'],j['array_sha256']);assert set(a)==({'theta','gradient','adam_m','adam_v'} if i<=21 else {'theta','gradient'});assert a['theta'].ndim==1 and all(x.shape==a['theta'].shape for x in a.values());assert digest(a['theta'])==j['theta_sha256'];assert float(np.max(np.abs(a['gradient'])))==j['gradient_inf'];ev=np.asarray(j['metrics']['value_relative_squared_by_mode']);eg=np.asarray(j['metrics']['gradient_relative_squared_by_mode']);assert ev.shape==eg.shape==(4,) and np.isfinite(ev).all() and np.isfinite(eg).all() and min(ev.min(),eg.min())>=0
 objective=float(weights@(ev+eg));gap=abs(objective-j['full_post_update_objective']);objective_gap=max(objective_gap,gap);assert np.isclose(objective,j['full_post_update_objective'],rtol=2e-13,atol=2e-15)
 if i==1:assert j['phase']=='initial' and j['adam_step']==0 and j['label']=='initial or exact restart'
 elif i<=21:assert j['phase']=='adam' and j['adam_step']==(i-1)*200 and j['label']=='Adam post-update'
 else:assert j['phase']=='lbfgs' and j['adam_step']==4000 and j['label']==('L-BFGS initial' if i==22 else 'L-BFGS accepted')
 if i<=22:assert j['full_evaluations']==i
 records[stem]=j;data[stem]=a;semantic_arrays[stem]={k:digest(v) for k,v in a.items()}
assert {x.stem for x in P.glob('checkpoint-*.json')}==set(records) and {x.stem for x in P.glob('checkpoint-*.npz')}==set(records)
rng=np.random.default_rng(cfg['seed']);sizes=[21]+[cfg['width']]*cfg['depth']+[2*cfg['rank']];initial=np.concatenate([a.ravel() for n,m in zip(sizes[:-1],sizes[1:]) for a in (rng.normal(size=(n,m))/np.sqrt(n),np.zeros(m))]);assert np.array_equal(initial,data['checkpoint-00001']['theta']);assert not np.any(data['checkpoint-00001']['adam_m']) and not np.any(data['checkpoint-00001']['adam_v'])
best_adam=min(list(records)[:21],key=lambda k:records[k]['full_post_update_objective']);assert np.array_equal(data[best_adam]['theta'],data['checkpoint-00022']['theta']) and np.array_equal(data[best_adam]['gradient'],data['checkpoint-00022']['gradient'])
check(P/'LINE-SEARCH.jsonl');lines=[json.loads(x) for x in (P/'LINE-SEARCH.jsonl').read_text().splitlines()];assert len(lines)==316;groups={i:[] for i in range(300)}
for row in lines:assert row['iteration'] in groups;groups[row['iteration']].append(row)
pairs=[];pair_ids=[];trial_gap=0.;accepted=0;counter=22;line_summary=[]
for iteration in range(300):
 before=f'checkpoint-{22+iteration:05d}';after=f'checkpoint-{23+iteration:05d}';old=data[before];new=data[after];g=old['gradient'];theta=old['theta'];q=g.copy();alphas=[]
 for step,y,rho in reversed(pairs):alpha=rho*np.dot(step,q);alphas.append(alpha);q-=alpha*y
 if pairs:q*=np.dot(pairs[-1][0],pairs[-1][1])/np.dot(pairs[-1][1],pairs[-1][1])
 for (step,y,rho),alpha in zip(pairs,reversed(alphas)):q+=step*(alpha-rho*np.dot(y,q))
 direction=-q;slope=np.dot(g,direction)
 if not np.isfinite(slope) or slope>=0:direction=-g;slope=-np.dot(g,g);pairs=[];pair_ids=[]
 group=groups[iteration];assert 1<=len(group)<=cfg['line_search_steps'];assert [r['attempt'] for r in group]==list(range(len(group))) and sum(r['accepted'] for r in group)==1 and group[-1]['accepted'];f=records[before]['full_post_update_objective']
 for row in group:
  assert row['step']==2.**(-row['attempt']) and np.isfinite(row['objective']) and np.isfinite(row['gradient_inf']);assert bool(row['objective']<=f+1e-4*row['step']*slope)==row['accepted'];counter+=1
 row=group[-1];assert row['theta_sha256']==records[after]['theta_sha256'] and row['objective']==records[after]['full_post_update_objective'] and row['gradient_inf']==records[after]['gradient_inf'];assert counter==records[after]['full_evaluations']
 trial=theta+row['step']*direction;gap=float(np.max(abs(trial-new['theta'])));trial_gap=max(trial_gap,gap);assert np.allclose(trial,new['theta'],rtol=1e-10,atol=1e-12),(iteration,gap)
 step=new['theta']-theta;y=new['gradient']-g;sy=float(step@y)
 if sy>1e-12*np.linalg.norm(step)*np.linalg.norm(y):pairs.append((step,y,1/sy));pair_ids.append([before,after]);pairs=pairs[-cfg['history_size']:];pair_ids=pair_ids[-cfg['history_size']:]
 assert pair_ids==records[after]['curvature_state_pairs'];accepted+=1;line_summary.append({'iteration':iteration,'trials':len(group),'accepted_step':row['step'],'trial_max_abs_gap':gap})
assert counter==338 and accepted==300
best=min(records,key=lambda k:records[k]['full_post_update_objective']);expected_best={'objective':records[best]['full_post_update_objective'],'checkpoint':best};assert final['best']==receipt['selected_checkpoint']==expected_best;assert final['last_checkpoint']=='checkpoint-00322';latest=read(P/'LATEST.json');assert latest['last']==final['last_checkpoint'] and latest['best']==expected_best
bank=P/receipt['bank_file'];check(bank,'faab6651618ae9d6afc79dbd91ad6e9c7a1a3101907d93d9e33ab37dfba2e4c6');assert receipt['bank_sha256']==provenance[str(bank)]
with np.load(bank,allow_pickle=False) as z:
 metadata=json.loads(str(z['metadata_json']));assert set(z.files)=={'metadata_json','transform'}|{f'{k}{i}' for i in range(cfg['depth']+1) for k in ('W','b')};assert np.array_equal(z['transform'],np.eye(4));layers=[]
 for i,(n,m) in enumerate(zip(sizes[:-1],sizes[1:])):
  w,b=z['W'+str(i)],z['b'+str(i)];assert w.shape==(n,m) and b.shape==(m,) and w.dtype==b.dtype==np.float64 and np.isfinite(w).all() and np.isfinite(b).all();layers.extend([w.ravel(),b])
 theta=np.concatenate(layers)
assert np.array_equal(theta,data[best]['theta']) and digest(theta)==receipt['selected_theta_sha256']=='a1f8d0a50e8847dcbc6ef90d1d21e24db8490351eafcdd51af9a2dc514e498f1'
for k,v in c.items():assert metadata[k]==v
assert metadata['schema']=='ns-obstacle-pod-guided-neural-v2' and metadata['features']=={'fourier_frequencies':[1.,2.,4.,8.],'radial_widths':[.1,.2,.4]} and metadata['envelope']=='bounded-exponential-distances-D0.1'
assert metadata['geometry']=={'kind':'channel2d','inlet':0.,'outlet':2.2,'center':[.2,.2],'obstacle_radius':.05,'height':.41,'pipe_radius':None} and metadata['layers']==cfg['depth']+1
assert metadata['actual_applied_adam_steps']==4000 and metadata['accepted_lbfgs_steps']==300 and metadata['selected_checkpoint']==expected_best and metadata['trained'] and metadata['status']=='iteration_limit' and metadata['FE_fallback'] is False and metadata['value_units']=='u/U' and metadata['gradient_coordinates']=='physical'
cont=read(P/'CONTINUATION.json');a=arrays(P/'CONTINUATION.npz',cont['array_sha256']);assert set(a)=={'theta','gradient','adam_m','adam_v'};assert np.array_equal(a['theta'],data['checkpoint-00322']['theta']) and np.array_equal(a['gradient'],data['checkpoint-00322']['gradient']);assert np.array_equal(a['adam_m'],data['checkpoint-00021']['adam_m']) and np.array_equal(a['adam_v'],data['checkpoint-00021']['adam_v']);assert cont['gradient_matches_theta'] and cont['theta_sha256']==cont['gradient_theta_sha256']==digest(a['theta']);assert cont['serial']==322 and cont['full_evaluations']==338 and cont['completed_lbfgs_iterations']==300 and cont['adam_step']==4000 and cont['phase']=='lbfgs';assert cont['config_sha256']==provenance[str(P/'CONFIG.json')] and cont['rng_state']==records['checkpoint-00322']['rng_state'];assert cont['curvature_state_pairs']==[[str(P/x) for x in pair] for pair in pair_ids]
closure=cont['checkpoint_closure'];needed={x for pair in cont['curvature_state_pairs'] for x in pair}|{cont['best']['checkpoint'],cont['last_measured_checkpoint']};assert set(closure)==needed and len(closure)==11
for stem,row in closure.items():assert Path(stem).parent==P;check(Path(stem+'.json'),row['record_sha256']);check(Path(stem+'.npz'),row['array_sha256'])
assert Path(cont['best']['checkpoint']).name==best and cont['best']['objective']==expected_best['objective'] and Path(cont['last_measured_checkpoint']).name==final['last_checkpoint']
signal.alarm(0)
print(json.dumps({'schema':'ns-obstacle-neural-guided-fit-saved-array-audit-v1','passed':True,'physical_admission':False,'validation_accessed':False,'checkpoint_count':322,'adam_steps_recorded':4000,'adam_measured_post_update_checkpoints':20,'accepted_lbfgs_steps':accepted,'line_search_trials':len(lines),'full_evaluations':counter,'continuation_closure_checkpoints':len(closure),'selected':expected_best,'selected_theta_sha256':digest(theta),'bank_sha256':provenance[str(bank)],'objective_from_saved_modal_metrics_max_abs_gap':objective_gap,'two_loop_trial_max_abs_gap':trial_gap,'initial_seed_exact':True,'final_per_mode_metrics':records[best]['metrics'],'line_search_replay':line_summary,'checkpoint_semantic_array_sha256':semantic_arrays,'teacher_identity':t,'provenance':provenance,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'limitations':['No neural/JAX/native execution, optimization, new solve or validation access.','Saved objective and gradients not recomputed against teacher arrays; saved modal metric aggregation only.','Adam inter-step updates and sampled minibatch gradients not replayed.','Rejected trial parameter arrays not retained; their Armijo decisions replayed from recorded objective and independently derived direction.','Accepted rederived vectors allow floating-point BLAS roundoff; actual checkpoint/accepted-row hashes remain exact.','Training objective is not full-field flow accuracy or physical admission.']},indent=2,allow_nan=False))
