"""Independent arithmetic and retention audit of the frozen confirmation outputs."""
from pathlib import Path
import json,hashlib,re
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 run=HERE/'runs/c01';out=run/'output';s=json.loads((out/'summary.json').read_text());replay=json.loads((out/'centroid_replay.json').read_text());cohort=json.loads((HERE/'COHORT_REGISTRATION.json').read_text())
 checks={};checks['expected_16_complete_cases']=[c['index'] for c in s['cases']]==list(range(16))
 checks['backend_precision']=s['backend']=='gpu' and s['precision']=='float64' and s['matmul_precision']=='highest'
 checks['parameter_hash']=s['parameter_sha256']==cohort['parameter_sha256']
 checks['explicit_parameters']=np.array_equal(np.asarray([c['parameters'] for c in s['cases']]),np.asarray(cohort['parameters']))
 checks['cohort_disjointness']=all(c['exact_tuple_overlap']==0 and c['rounded_12dp_tuple_overlap']==0 for c in cohort['comparisons'])
 checks['all_qualification_flags']=all(v for k,v in s['qualification'].items() if k!='cases_completed')
 checks['development_metric_replay']=replay['qualified']
 checks['source_driver_hash']=s['driver_sha256']==sha(run/'experiments/jcp-phase-confirm/run_mechanism.py')
 checks['frozen_config_hash']=sha(HERE/'configs/confirm.json')==sha(run/'experiments/jcp-phase-confirm/configs/confirm.json')
 checks['frozen_method_executed']=s['config']['n']==64 and s['config']['rank']==64 and s['config']['k']==8 and all({(q['arm'],q['rows'],q['dt'],q['gtol']) for q in c['runs']}=={(a,584,.01,g) for a in ('head_common','span_common','span_native') for g in (1e-7,1e-9)} for c in s['cases'])
 arrays=[];probe_rows=[];fit_failure_count=0;total_optional_fits=0;case_errors=[]
 for c in s['cases']:
  path=out/f"case_{c['index']:03d}"
  for item in c['runs']:
   arr=np.load(path/item['state_file']);rec=json.loads((path/item['records_file']).read_text());nsteps=round(s['config']['horizon']/item['dt'])
   arrays.append(arr['states'].shape[0]==nsteps+1 and arr['centers'].shape==(nsteps+1,3) and arr['coefficients'].shape==(nsteps+1,64) and all(np.isfinite(arr[k]).all() for k in arr.files))
   arrays.append(np.array_equal(arr['time'],np.arange(nsteps+1)*item['dt']))
   errors=np.array([m['relative_error'] for m in item['metrics']]);arrays.append(abs(float(errors[1:].max())-item['evolved_worst'])<1e-15)
   arrays.append(len(rec['records'])==nsteps and all(q['reason']=='stationarity' and q['gradient']<=item['gtol'] for q in rec['records']))
   for f in rec['span_head_fits']:
    total_optional_fits+=1;fit_failure_count+=f['starts'][f['best_index']]['reason']!='stationarity'
  probes=json.loads((path/'reset_m584_dt0.01.json').read_text())
  for p in probes:
   probe_rows.append({'case':c['index'],'step':p['step'],'stationary':p['span_result']['reason']=='stationarity','objective_nonincrease':p['span_result']['residual_norm']<=p['head_seed_residual_norm']*(1+1e-12),'span_error_larger':p['span_physical_error']>p['head_physical_error']})
  h=next(x for x in c['runs'] if x['arm']=='head_common' and x['gtol']==1e-9);sp=next(x for x in c['runs'] if x['arm']=='span_common' and x['gtol']==1e-9)
  case_errors.append({'case':c['index'],'head_max':h['evolved_worst'],'span_max':sp['evolved_worst'],'paired_span_over_head':sp['evolved_worst']/h['evolved_worst']})
 checks['state_arrays_and_arithmetic']=all(arrays)
 checks['same_previous_state_probe_stationarity']=all(p['stationary'] for p in probe_rows)
 checks['same_previous_state_objective_nonincrease']=all(p['objective_nonincrease'] for p in probe_rows)
 concentrations=np.asarray([v for c in s['cases'] for a in c['runs'] for m in a['metrics'] for key in ('pred_centroid_concentration','truth_centroid_concentration') for v in m[key]])
 checks['concentrations_finite_in_unit_interval']=bool(np.isfinite(concentrations).all() and concentrations.min()>=0 and concentrations.max()<=1+1e-14)
 logs='\n'.join(p.read_text(errors='replace') for p in (run/'logs').glob('*'))
 bad=re.findall(r'.*(?:captured.?large.?constant|out of memory|CUDA_ERROR|No space left|RESOURCE_EXHAUSTED|Traceback).*',logs,re.I)
 checks['no_failure_log_signatures']=not bad;checks['gpu_preflight']='jax_backend=gpu' in logs;checks['success_exit']='RUN_EXIT=0' in logs
 for label in ('INPUTS','OUTPUTS'):
  failures=[]
  for line in (run/f'{label}.sha256').read_text().splitlines():
   checksum,name=line.split(maxsplit=1);name=name.lstrip('*');p=run/name
   if not p.is_file() or sha(p)!=checksum:failures.append(name)
  checks[f'{label.lower()}_checksums']=not failures
 report={'schema':'phase-confirm-audit-v1','passed':all(checks.values()),'evidence_integrity_passed':all(v for k,v in checks.items() if k!='all_qualification_flags'),'scientific_confirmation_qualified':bool(s['qualification']['passed']),'initial_fit_failed_case_indices':[c['index'] for c in s['cases'] if not c['head_ic_fit']['qualified']],'checks':checks,'case_errors':case_errors,'probe_counts':{'total':len(probe_rows),'stationary':sum(p['stationary'] for p in probe_rows),'objective_nonincrease':sum(p['objective_nonincrease'] for p in probe_rows),'span_physical_error_larger':sum(p['span_error_larger'] for p in probe_rows)},'optional_fixed_frame_head_fit_nonstationary':fit_failure_count,'optional_fixed_frame_head_fit_total':total_optional_fits,'optional_fit_interpretation':'best-found upper bounds, not globally optimal projection; not used in phase-shape panel','minimum_fresh_concentration_all_runs':float(concentrations.min()),'centroid_descriptive_ambiguity_entries':int((concentrations<.1).sum()),'bad_log_signatures':bad,'source_summary_sha256':sha(out/'summary.json'),'source_replay_sha256':sha(out/'centroid_replay.json'),'job_id':s['job_id'],'source_commit':s['source_commit'],'gpu':s['gpu']}
 (HERE/'CONFIRMATION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('case_errors',)},indent=2))
 if not report['evidence_integrity_passed']:raise SystemExit(1)
if __name__=='__main__':main()
