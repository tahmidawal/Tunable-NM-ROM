"""Case-balanced costs and the frozen conditional-interval matching rule; no solver calls."""
import numpy as np

def summarize(result):
    ids=result['cohort']['eligible_cases'];summaries={}
    for sid,subject in result['subjects'].items():
        rows=result['case_results'][sid]
        errors=[rows[str(j)]['score']['e'] for j in ids]
        finite=all(e is not None and np.isfinite(e) for e in errors)
        rec=dict(eligible_cases=ids,eligible_count=len(ids),invalid_count=sum(not rows[str(j)]['valid'] for j in ids),
                 median_error=float(np.median(errors)) if finite else None,worst_error=max(errors) if finite else None,
                 qualified=subject['qualification']['status']=='qualified',costs={})
        for contract,field,mask in [('prepared_native','given_native_seconds','native_timing_admitted'),
                                    ('descriptor','descriptor_seconds','timing_admitted')]:
            percase={j:[r[field] for r in result['timing_repetitions'] if r['subject']==sid and r['case']==j and r['role']=='B' and r[mask]] for j in ids}
            covered=all(percase[j] for j in ids)
            rec['costs'][contract]=dict(coverage_complete=covered,covered_cases=[j for j in ids if percase[j]],
                case_medians={j:float(np.median(v)) for j,v in percase.items() if v},
                median_seconds=float(np.median([np.median(percase[j]) for j in ids])) if covered else None,
                admitted_repetitions=sum(map(len,percase.values())),
                rejected_repetitions=sum(r['subject']==sid and r['role']=='B' and not r[mask] for r in result['timing_repetitions']))
        if finite:
            med=rec['median_error'];rec['errors_above_five_times_median']=sum(e>5*med for e in errors)
            rec['errors_above_one_percent']=sum(e>.01 for e in errors)
            rec['errors_above_five_percent']=sum(e>.05 for e in errors)
        summaries[sid]=rec
    result['summaries']=summaries
    result['matching']=dict(predicate='Per-case intervals[max(e-epsilon,0),e+epsilon]; joint median(upper competitor)<=median(lower subject) AND max(upper competitor)<=max(lower subject); identical eligible cases.',
                            uncertainty='conditional Richardson intervals, not rigorous error bounds',subjects={})
    for sid,s in result['subjects'].items():
        if s['config']['kind'] not in ('bank','POD'):continue
        own=summaries[sid];result['matching']['subjects'][sid]={}
        for family in ('PS','FD','POD'):
            choices=[cid for cid,sub in result['subjects'].items() if cid!=sid and sub['config']['kind']==family]
            result['matching']['subjects'][sid][family]={}
            for contract in ('descriptor','prepared_native'):
                usable=[];checks={}
                for cid in choices:
                    comp=summaries[cid]
                    if not(own['qualified'] and comp['qualified'] and own['costs'][contract]['coverage_complete'] and comp['costs'][contract]['coverage_complete']):
                        checks[cid]=dict(status='unresolved qualification or timing coverage');continue
                    e=np.asarray([result['case_results'][sid][str(j)]['score']['e'] for j in ids]);f=np.asarray([result['case_results'][cid][str(j)]['score']['e'] for j in ids])
                    eps=np.asarray([result['case_results'][sid][str(j)]['score']['epsilon'] for j in ids])
                    assert np.array_equal(eps,[result['case_results'][cid][str(j)]['score']['epsilon'] for j in ids])
                    median=bool(np.median(f+eps)<=np.median(np.maximum(e-eps,0)))
                    worst=bool(np.max(f+eps)<=np.max(np.maximum(e-eps,0)))
                    checks[cid]=dict(median_match=median,worst_match=worst,joint_match=median and worst,
                                     status='matched' if median and worst else 'unresolved overlap or insufficient accuracy')
                    if median and worst:usable.append(cid)
                pick=min(usable,key=lambda c:summaries[c]['costs'][contract]['median_seconds']) if usable else None
                result['matching']['subjects'][sid][family][contract]=dict(selected=pick,checks=checks,
                    competitor_over_subject_cost_ratio=(summaries[pick]['costs'][contract]['median_seconds']/own['costs'][contract]['median_seconds']) if pick else None,
                    scope='fastest jointly matching member of this frozen finite tested set')
    result['frontiers']={}
    for contract in ('descriptor','prepared_native'):
        result['frontiers'][contract]={}
        for metric in ('median_error','worst_error'):
            points=[]
            for sid,r in summaries.items():
                c=r['costs'][contract]
                if not(r['qualified'] and c['coverage_complete'] and r[metric] is not None):continue
                points.append(dict(subject=sid,cost=c['median_seconds'],error=r[metric]))
            for p in points:p['nondominated']=not any(q['cost']<=p['cost'] and q['error']<=p['error'] and (q['cost']<p['cost'] or q['error']<p['error']) for q in points)
            result['frontiers'][contract][metric]=points
    return result
