"""Scientific figures generated only from saved, same-population pilot observations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
raw=HERE/'runs/pilot/collected/output/result.json'
j=json.loads(raw.read_text());dest=HERE/'plots';dest.mkdir(exist_ok=True)
ids=j['config']['cases']
points=[len(np.load(HERE/'inputs'/f'HR~ecsw~512~{a}.npz')['nodes']) for a in ['k8','k16']]
field=[max(v['field_distance'][f'{a}_to_none'] for v in j['cases'].values()) for a in ['k8','k16']]
grad=[max(max(v[a]['whole_residual']['delta_g']) for v in j['cases'].values()) for a in ['k8','k16']]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for a,y,bar,label in zip(ax,[field,grad],[j['config']['admission']['field_distance'],j['config']['admission']['gradient']],
                         ['Maximum rollout distance','Maximum whole-residual gradient discrepancy']):
    a.plot(points,y,'o-',color='#9c2f2f',lw=2,ms=7,label='Frozen ECSW supports')
    a.axhline(bar,color='#374151',ls='--',label=f'Frozen admission bar ({bar:g})')
    a.set_xscale('log',base=2);a.set_yscale('log');a.set_xticks(points,[str(x) for x in points])
    a.set_xlabel('Weighted residual points');a.set_ylabel(label)
    for x,v in zip(points,y):a.annotate(f'{v/bar:.2f}× bar',(x,v),xytext=(0,8),textcoords='offset points',ha='center')
    a.grid(alpha=.2,which='both')
for a in ax:a.legend(frameon=False,loc='lower left')
fig.suptitle(f'Frozen POD hyper-reduction fails the new admission contract\nSelected validation cases {", ".join(map(str,ids))}; diagnostic scope')
fig.savefig(dest/'pod_pilot_convergence.png',dpi=180)
plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
colors={'k8':'#bb3c39','k16':'#dd8d29','none':'#285d7d'}
done=set();rejected=0
for panel in j['timing']:
    if not panel['drift_accepted']:rejected+=1;continue
    for v in panel['invocations']:
        a=v['arm'];label={'k8':'k8: HR unqualified','k16':'k16: HR unqualified','none':'Full residual: Q1 passes on pilot'}[a]
        ax.scatter(v['seconds']*1000,v['e0']['e']*100,marker='x' if a!='none' else 'o',
                   color=colors[a],s=35,label=label if a not in done else None,alpha=.8)
        done.add(a)
ax.set_xlabel('Full given-field query time (ms)');ax.set_ylabel('E0 error (%)')
ax.set_title(f'Same-invocation POD accuracy and cost\nTimed cases {j["config"]["timing_cases"]}; job {j["job"]}; rejected drift panels: {rejected}')
ax.grid(alpha=.2);ax.legend(frameon=False)
fig.savefig(dest/'pod_pilot_accuracy_query.png',dpi=180)
plt.close(fig)
(dest/'PROVENANCE.json').write_text(json.dumps(dict(source=str(raw.resolve()),cases=ids,timed_cases=j['config']['timing_cases'],
    point_counts=points,field_to_none=field,gradient_discrepancy=grad,drift_rejected_panels=rejected,
    note='Convergence uses the same four cases for every point; timing scatter uses each actual timed invocation.'),indent=2)+'\n')
print(dest)
