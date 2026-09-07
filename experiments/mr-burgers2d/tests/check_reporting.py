"""CPU checks for additive-margin gating and the intended paired statistic."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reports'))
from generate_rollout import summarize,select
rows=[]
settings=[('adequate_rom','rom',.045,[2,2,6,4],2),('margin_miss','rom',.047,[.2]*4,2),('failed_cold','rom',.01,[.1]*4,3),('fom','fom',.045,[1,2,3,4],None)]
for name,method,err,times,reason in settings:
    for c,t in enumerate(times):
        for rep in range(3):
            errors=dict(fixed_initial_max=err)
            rows.append(dict(name=name,method=method,case=c,rep=rep,seconds=t,output_intervals=256,solver_intervals=256,dt=.005,
                finite=True,nonlinear_tolerance_satisfied=True,physical_error=errors,dense_physical_error=errors,
                ic_reason=reason,stop_reasons=[2] if method=='rom' else []))
metric=dict(uncertainty=[dict(case=c,conservative_difference_sum=.004) for c in range(4)],
    order_audit=[dict(case=c,asymptotic_decrease_observed=True,empirical_richardson_estimate=.003) for c in range(4)])
d=dict(config=dict(cases=4,reps=3),declared_subjects=[dict(name=x[0]) for x in settings],invocations=rows,observation_intervals=256,reference_metrics_by_output={'256':metric})
summary,_=summarize(d);selected=select(d,summary)
r=next(r for r in selected if r['norm']=='dense' and r['target']==.05)
assert r['rom']['name']=='adequate_rom' and r['fom']['name']=='fom' and r['paired_ratio']==.75
assert all(r['paired_ratio'] is None for r in selected if r['target']==.01)
d['invocations']=rows[:-1]
try:summarize(d)
except AssertionError:pass
else:raise AssertionError('missing repetition not rejected')
print('PASS: additive reference margin, failed cold exclusion, casewise ratio, incomplete repetition guard')
