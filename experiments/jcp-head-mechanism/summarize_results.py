"""Generate all reported numbers from retained run JSONs; no manually entered results."""
from pathlib import Path
import argparse,json
import numpy as np

def stats(values):
    a=np.asarray(values,dtype=float)
    return {'count':len(a),'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max()),'mean':float(a.mean())} if len(a) else {'count':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    p=args.run;s=json.loads((p/'summary.json').read_text());tight=min(s['config']['solver']['gtols'])
    result={'qualification':s['qualification'],'job_id':s['job_id'],'groups':[],'reference':[],'containment':{},'fits':{}}
    contain=[];fits=[];steps=[];reproduced=[];gram_ranks={};gaps=[]
    for c in s['cases']:
        result['reference'].append(c['reference_check']);gaps.extend(x['max_gap'] for x in c['tight_tighter'])
        for setting in s['config']['test_settings']:
            row={'case':c['index'],**setting}
            for arm in s['config']['arms']:
                a=next(x for x in c['runs'] if x['arm']==arm and x['rows']==setting['rows'] and x['dt']==setting['dt'] and x['gtol']==tight)
                row[arm]=a['evolved_worst']
                row[arm+'_phase_worst']=max(float(np.linalg.norm(x['phase_displacement'])) for x in a['metrics'])
                row[arm+'_shape_aligned_worst']=max(x['shape_aligned_relative_error'] for x in a['metrics'][1:])
                row[arm+'_divergence_worst']=max(x['relative_divergence'] for x in a['metrics'])
                row[arm+'_energy_error_worst']=max(abs(x['energy']-x['truth_energy'])/x['truth_energy'] for x in a['metrics'])
                raw=json.loads((p/f"case_{c['index']:03d}"/a['records_file']).read_text());steps.extend(raw['records']);fits.extend(raw['span_head_fits'])
                for rec in raw['records']:gram_ranks.setdefault(arm,[]).append(rec['physical_tangent']['rank'])
                if a.get('pilot_reproducibility'):reproduced.extend(a['pilot_reproducibility']['physical_field_gaps'])
            row['span_over_head']=row['span_common']/row['head_common'];result['groups'].append(row)
            contain.extend(json.loads((p/f"case_{c['index']:03d}"/f"reset_m{setting['rows']}_dt{setting['dt']:g}.json").read_text()))
    result['by_setting']=[]
    for setting in s['config']['test_settings']:
        rows=[x for x in result['groups'] if x['rows']==setting['rows'] and x['dt']==setting['dt']]
        group={**setting,'cases':len(rows),'head_better_count':sum(x['head_common']<x['span_common'] for x in rows),'paired_error_ratio':stats([x['span_over_head'] for x in rows])}
        for arm in s['config']['arms']:
            group[arm]=stats([x[arm] for x in rows]);group[arm]['above_half_percent']=sum(x[arm]>.005 for x in rows)
        result['by_setting'].append(group)
    valid=[x for x in contain if x['span_result']['reason']=='stationarity']
    result['containment']={'count':len(contain),'stationary':len(valid),'stationary_lower_residual':sum(x['span_result']['residual_norm']<=x['head_seed_residual_norm'] for x in valid),'stationary_worse_error':sum(x['span_physical_error']>x['head_physical_error'] for x in valid),'stationary_error_ratio':stats([x['span_physical_error']/x['head_physical_error'] for x in valid])}
    result['fits']={'count':len(fits),'chosen_stationary':sum(x['starts'][x['best_index']]['reason']=='stationarity' for x in fits),'distance_semantics':'All best-found distances are upper bounds, including converged local fits; no certified global separation.'}
    result['tight_rollout_steps']={'count':len(steps),'max_gradient':max(x['gradient'] for x in steps),'guarded_acceptances':sum(h.get('acceptance_reason')=='roundoff_gradient_improvement' for x in steps for h in x['history'])}
    result['physical_ranks']={key:stats(vals) for key,vals in gram_ranks.items()};result['tight_tighter_max_gap']=max(gaps);result['pilot_reproducibility']=stats(reproduced)
    geo=[x['shared_state_geometry'] for x in valid if 'shared_state_geometry' in x]
    if geo:
        result['same_state_geometry']={'count':len(geo),'chain_rule_max_relative':max(x['analytic_chain_relative'] for x in geo),'span_rank':stats([x['span_rank'] for x in geo]),'head_rank':stats([x['head_rank'] for x in geo]),'least_visible_outside_head_fraction':stats([x['least_visible_span_outside_head_fraction'][0] for x in geo]),'actual_step_linearized_outside_head_fraction':stats([x['actual_step_linearized_outside_head_fraction'] for x in geo]),'head_to_span_smallest_singular_ratio':stats([x['head_physical_singular_values'][-1]/x['span_physical_singular_values'][-1] for x in geo]),'scope':'The singular-value restriction inequality is algebraic. These finite local samples are not a uniform stability bound; step fractions are tangent-linearized.'}
    (p/'analysis_compact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','reference')},indent=2))
if __name__=='__main__':main()
