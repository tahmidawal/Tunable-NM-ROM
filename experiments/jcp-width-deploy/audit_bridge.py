"""Recompute gates and actual nonzero-weight stencil closure from retained arrays."""
from pathlib import Path
import json,hashlib
import numpy as np
L=Path(__file__).resolve().parent;D=L/'checks/bridge';D.mkdir(parents=True,exist_ok=True)
s=json.loads((L/'runs/b01/output/summary.json').read_text());collect=json.loads((L/'runs/b01/collection.json').read_text());assert collect['scientifically_qualified']
report={'scope':s['scope'],'fresh_data_used':s['fresh_data_used'],'rules':{},'source_sha256':hashlib.sha256((L/'runs/b01/output/summary.json').read_bytes()).hexdigest(),'no_pruning_threshold':True,'eligibility_uses_exact_positive_weights':True,'no_sparse_rollout_or_timing_claim':True}
for name,rec in s['rules'].items():
 z=np.load(L/f'runs/b01/output/{name}-rule.npz');w=z['dimensionless_weights'];ix=z['indices'];n=256
 assert np.isfinite(w).all() and np.all(w>=0);active=np.zeros((n-1)**2,dtype=bool);active[ix[w>0]]=True;active=active.reshape(n-1,n-1)
 closure=active.copy();closure[:-1]|=active[1:];closure[1:]|=active[:-1];closure[:,:-1]|=active[:,1:];closure[:,1:]|=active[:,:-1]
 actual=int(closure.sum());proposed=int(z['closure'].size);support=int(active.sum())
 gates=rec['gates'].copy();gates['site_economy']=actual/((n-1)**2)<=.8
 assert support==rec['actual_positive_weights'];assert all(abs(v['direction_mass_norm']-1)<=1e-9 for v in rec['probes'])
 assert all((v['case'],v['time_index']) not in {(int(c),int(t)) for c,t in s['fit_pairs']} for v in rec['moment_records'] if not v['fit_snapshot'])
 recomputed=dict(heldout_moments=max(x['relative_error'] for x in rec['moment_records'] if not x['fit_snapshot'])<=1e-2,residual_fidelity=max(x['residual_initial_norm'] for x in rec['probes'])<=1e-5,action_fidelity=max(x['action_relative'] for x in rec['probes'])<=1e-3,cold_start=all(x['rank']==512 and x['field_difference_initial_norm']<=1e-4 for x in rec['cold_start']))
 assert all(gates[k]==v for k,v in recomputed.items())
 r=dict(proposed_support_count=len(ix),actual_nonzero_support_count=support,exact_zero_weights=int(np.sum(w==0)),proposed_stencil_count=proposed,actual_nonzero_stencil_count=actual,dense_count=(n-1)**2,actual_stencil_fraction=actual/((n-1)**2),weight_range=[float(w.min()),float(w.max())],fit_optimizer_success=rec['fit']['success'],fit_optimizer_message=rec['fit']['message'],fit_iterations=rec['fit']['iterations'],projected_gradient_inf=rec['fit']['projected_gradient_inf'],training_moment_max=rec['fit']['training_moment_max'],heldout_moment_max=rec['heldout_moment_max'],residual_initial_norm_max=rec['probe_residual_max'],jacobian_action_relative_max=rec['probe_action_max'],cold_start_field_difference_max=rec['cold_start_max'],cold_start=rec['cold_start'],gates=gates,fidelity_passed=all(v for k,v in gates.items() if k!='site_economy'),admitted=all(gates.values()),actual_vs_proposed_gate_changed=gates['site_economy']!=rec['gates']['site_economy'])
 report['rules'][name]=r
report['any_rule_admitted']=any(x['admitted'] for x in report['rules'].values());report['downstream_rollout_gate_closed']=not report['any_rule_admitted']
(D/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
