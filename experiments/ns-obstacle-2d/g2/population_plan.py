"""Generate a conservative finite population coverage/cost/disk proposal from evidence."""
from pathlib import Path
import datetime,hashlib,json,shutil
B=Path(__file__).resolve().parents[1]
C=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle/experiments/ns-obstacle')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def size(p):return sum(x.stat().st_size for x in p.rglob('*') if x.is_file()) if p.is_dir() else p.stat().st_size
specs=[('p4_g4_h1','artifacts/g2-p4trend/results/dfg_h1'),('p4_g4_h2','artifacts/g2-p4trend/results/dfg_h2'),('p4_g4_h3','artifacts/g2-p4trend/results/dfg_h3'),('p3_g3_h3','artifacts/g2-h3/results/dfg_h3'),('p3_g4_h3','artifacts/g2-p4controls/results/dfg_p3_g4_h3')]
population={'train':[15,16.25,17.5,18.75,20,21.25,22.5,23.75,25],'development':[16.875,23.125],'validation':[16,18,22,24]}
quad=B/'artifacts/g2-solverquad01/results/quadrature_plus4/result.json';solver=B/'artifacts/g2-solverquad01/results/solver_100x/result.json'
q=read(quad);s=read(solver);fine=read(B/specs[2][1]/'result.json')
readout_keys=('reference_diagnostics_and_export','trace_certificate_and_reload')
measured_q_readout_ratio=sum(q['reference_phase_seconds'][k] for k in readout_keys)/sum(fine['reference_phase_seconds'][k] for k in readout_keys)
common_points=read(B/'artifacts/g2-p4trend/results/comparison_dfg/comparison.json')['comparisons'][0]['point_count']
quad_points=read(B/'artifacts/g2-solverquad01/results/comparison_solver_quad/comparison.json')['comparisons'][0]['point_count']
quad_point_ratio=quad_points/common_points;states=[];coverage=[]
for name,folder in specs:
 path=B/folder;r=read(path/'result.json');phase=r['reference_phase_seconds'];base=phase['total']+2.0;readout=sum(phase[k] for k in readout_keys);compute=phase['total']-readout
 estimates={'baseline':base,'solver_100x':base+.25*phase['newton_query'],'quadrature_plus4':1.25*compute+max(quad_point_ratio,measured_q_readout_ratio)*readout+2.0}
 manifest=read(path/'EXPORT-MANIFEST.json');retained_bytes=sum((path/member).stat().st_size for member in manifest['files'])+(path/'EXPORT-MANIFEST.json').stat().st_size
 states.append({'state':name,'baseline_directory':folder,'result_sha256':sha(path/'result.json'),'manifest_sha256':sha(path/'EXPORT-MANIFEST.json'),'immutable_mesh_sha256':sha(path/'mesh.vol.bin'),'layout_sha256':read(path/'fe_layout.json')['sha256'],'actual_Re20_phase_seconds':phase,'actual_export_member_bytes':retained_bytes,'prospective_per_parameter_variant_seconds':estimates,'physics_admission_scope':'Only adopted P4/g4/h3 must be admitted; other states supply their own diagnostic contrasts, never donated metrics.'})
 for split,parameters in population.items():
  for Re in parameters:
   variants={}
   for variant,seconds in estimates.items():
    retained=Re==20 and (variant=='baseline' or name=='p4_g4_h3')
    variants[variant]={'status':'retained' if retained else 'direct_run_required','planning_seconds_before_contingency':seconds,'source_directory':folder if retained and variant=='baseline' else ('artifacts/g2-solverquad01/results/'+variant if retained else None),'on_original_common_rule':Re==20 and variant=='baseline'}
   coverage.append({'Re':Re,'split':split,'state':name,'variants':variants,'independent_R':'reuse compatible same-rule retained replay where verified; otherwise fresh coefficient evaluation required, no extra PDE solve','noise_policy':'eta=S+Q+max(R,1e-12); no parameter or endpoint inheritance'})
pending=[v for row in coverage for v in row['variants'].values() if v['status']=='direct_run_required'];base_pending=sum(v['planning_seconds_before_contingency'] for v in pending);contingency=1.25
comparison_sources=[(B/'artifacts/g2-p4trend/results/p4_progress.json','comparison_dfg',3),(B/'artifacts/g2-crossp02/results/crossp_progress.json','comparison_p_geometry',3)]
# Discover the exact completed cross-p stage label rather than assume its spelling.
comparison_records=[]
for path,label,count in comparison_sources:
 data=read(path);matches=[x for x in data['completed'] if x['name']==label]
 if not matches:matches=[x for x in data['completed'] if 'comparison' in x['name'] or 'common' in x['name']]
 if len(matches)!=1:raise ValueError('ambiguous measured comparison stage')
 comparison_records.append({'path':str(path.relative_to(B)),'sha256':sha(path),'stage':matches[0]['name'],'seconds':matches[0]['seconds'],'state_count':count,'seconds_per_state':matches[0]['seconds']/count})
scalar_seconds=max(x['seconds_per_state'] for x in comparison_records)
# Five Re20 baseline common fields already exist; retained coefficient replays are not PDE solves.
common_field_evaluations=len(coverage)*3-5;fresh_replays_upper=len(coverage);scalar_comparison_seconds=(common_field_evaluations+fresh_replays_upper)*scalar_seconds*contingency
ledger_path=C/'state/job-ledger.json';ledger=read(ledger_path);charged={}
for row in ledger['allocations']:
 charged[int(row['job_id'])]=float(row.get('elapsed_seconds') or row.get('wall_limit_seconds',1200))*int(row.get('gpus',1))
charged.update({6617371:840,6617599:214});charged.setdefault(6617721,1200);charged.setdefault(6617840,1200)
existing_and_reserved=sum(charged.values());native_unoptimized=base_pending*contingency
# Disk envelope intentionally assumes no unmeasured cross-state compression/dedup gain.
raw_per_parameter=sum(x['actual_export_member_bytes']*3 for x in states);extra_quad=sum(x['actual_export_member_bytes']*(quad_point_ratio-1) for x in states)
future_raw_envelope=(raw_per_parameter+extra_quad)*14+sum(x['actual_export_member_bytes']*2 for x in states if x['state']!='p4_g4_h3')
old_names=['g2-ladder01','g2-ladder02','g2-comparison01','g2-h3','g2-p4trend','g2-p4controls','g2-crossp01','g2-crossp02']
cold=[]
for name in old_names:
 for path in [B/'runs'/name,B/'runs'/(name+'-collection.tar.gz'),B/'staging'/name/'payload']:
  if path.exists():cold.append({'path':str(path.relative_to(B)),'bytes':size(path),'precondition':'Verify matching RETENTION archive/member/chunk hashes and durable code-only mirror before duplicate-only removal; preserve canonical artifact evidence.'})
free=shutil.disk_usage(B).free
plan={'status':'prospective_not_population_release','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'noise_protocol':{'path':str(C/'reviews/RESOLVED-NOISE-PROTOCOL.md'),'sha256':sha(C/'reviews/RESOLVED-NOISE-PROTOCOL.md')},'population':population,'baseline_states':states,'coverage':coverage,'counts':{'parameters':sum(map(len,population.values())),'baseline_state_rows':len(coverage),'total_baseline_and_S_Q_states':len(coverage)*3,'retained_states':len(coverage)*3-len(pending),'pending_PDE_states':len(pending),'new_common_rule_evaluations_before_replay':common_field_evaluations,'fresh_replay_upper_count':fresh_replays_upper},'common_rule':{'source':'artifacts/g2-p4trend/results/dfg_h3/full_rule.npz','sha256':sha(B/'artifacts/g2-p4trend/results/dfg_h3/full_rule.npz'),'point_count':common_points,'policy':'Use every original P4 h3 point/weight, adopted-P4 denominator per parameter. Re-evaluate higher-quadrature coefficient states on this rule; preserve existing higher-rule comparison separately. No masking, pressure shifting or remeshing the shared training mesh.'},'cost':{'cross_job_speed_claim':False,'purpose':'conservative allocation planning from actual phase evidence','contingency_multiplier':contingency,'solver_variant_extra_Newton_phase_fraction':.25,'quad_compute_multiplier':1.25,'quad_readout_multiplier':max(quad_point_ratio,measured_q_readout_ratio),'actual_fine_solver_result_sha256':sha(solver),'actual_fine_quad_result_sha256':sha(quad),'pending_native_seconds_without_batch_speedup':native_unoptimized,'scalar_comparison_records':comparison_records,'scalar_common_fields_and_replay_seconds':scalar_comparison_seconds,'existing_and_reserved_campaign_seconds':existing_and_reserved,'ledger_snapshot_sha256':sha(ledger_path),'export_batch_proof_max_seconds':1200,'combined_unoptimized_seconds':native_unoptimized+scalar_comparison_seconds+existing_and_reserved+1200,'campaign_ceiling_seconds':12*3600,'fits_unoptimized_before_any_ROM_training':native_unoptimized+scalar_comparison_seconds+existing_and_reserved+1200<=12*3600,'ROM_training_and_paired_timing_reserve':'must be supplied by coordinator/ROM owner before admission; not silently zero','speedup_assumed':False},'critical_cost_work':[{'step':'review and run exact export-batch01 saved-state proof','source_commit':'d57f23e9a','status':'source review pending; no speed credited'},{'step':'reuse existing certified original-rule element/reference-coordinate arrays on each immutable mesh; build native batched evaluation and fresh-process replay','status':'source/native equivalence and measured phase cost still required','scope':'Geometry mapping cache may be shared across parameters only because exact binary mesh and original physical rule hashes match; this is not parameterwise physics/noise inheritance.'},{'step':'regenerate budget from measured controls; split jobs to1100s runner/1200s allocation with reserve for collection','status':'required before population release'}],'disk':{'free_bytes_at_generation':free,'minimum_untouched_reserve_bytes':10*1024**3,'submission_free_guard_bytes':22*1024**3,'future_export_raw_envelope_bytes_before_new_common_fields':future_raw_envelope,'future_common_fields_uncompressed_upper_bytes':(common_field_evaluations+fresh_replays_upper)*common_points*3*8,'current_duplicate_only_candidates':cold,'duplicate_only_reclaim_bytes':sum(x['bytes'] for x in cold),'policy':'Current full-expanded-plus-archive-plus-chunks retention is too expensive for the whole population. Before population release, review archive-only lossless retention with pinned commit/chunk/member hashes, explicit recovery, and temporary rehydration; no Git history rewrite/pruning, no unique evidence deletion, no assumed dedup gain. Keep at most a bounded current batch warm; immutable inputs/dependencies need not be recopied into every future archive if their exact recoverable source is pinned and verified.','archive_only_source_or_recovery_ready':False},'domain_coverage':{'measured_control_Re':20,'jobs':[6617599,6617721,6617840],'common_original_domain_comparison':'pending exact package after all three collections','scope':'Domain/outlet flow evidence is separate; disclose remeshing and compare pressure drops. No assumption that Re20 proves parameterwise domain adequacy. Coordinator must explicitly fix family domain-coverage policy before population release; additional direct controls would add to this budget, not be hidden.'},'admission':False}
path=B/'g2/POPULATION-NOISE-RESOURCE-PLAN.json';path.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'counts':plan['counts'],'cost':plan['cost'],'disk_free_GiB':free/1024**3,'duplicate_reclaim_GiB':sum(x['bytes'] for x in cold)/1024**3,'future_export_envelope_GiB':future_raw_envelope/1024**3},indent=2))
