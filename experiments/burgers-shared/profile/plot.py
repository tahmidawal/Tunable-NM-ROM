"""Additive stage bars from each arm's median-total repetition; all timings retained."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
SRC=HERE/'result.json'
x=json.loads(SRC.read_text());assert x['complete']
OUT=HERE.parent/'plots/profile';OUT.mkdir(parents=True,exist_ok=True)
phases=x['phase_names'];arms=['direct','affine'];modes=['fused','split']
r={(t['arm'],t['mode'],t['case'],t['block']):t for t in x['timings']}
assert len(r)==168
sums={(arm,mode):np.array([sum(r[arm,mode,c,b]['seconds'] for c in range(6))*1000 for b in range(7)]) for arm in arms for mode in modes}
selected={arm:int(np.argsort(sums[arm,'split'])[3]) for arm in arms}
stages={arm:{p:sum(r[arm,'split',c,selected[arm]]['phases'][p] for c in range(6))*1000 for p in phases} for arm in arms}
for arm in arms:
 np.testing.assert_allclose(sum(stages[arm].values()),np.median(sums[arm,'split']),rtol=1e-12)
for t in x['timings']:
 if t['mode']=='split':
  assert min(t['phases'].values())>=0
  np.testing.assert_allclose(sum(t['phases'].values()),t['seconds'],rtol=1e-12)
summary=dict(job_id=x['job_id'],source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
 method='Sum six query times within each repetition; select median-total repetition separately for each arm. All components in each stack come from that repetition.',
 selected_block=selected,stage_ms=stages,totals_ms={f'{a}_{m}':sums[a,m].tolist() for a in arms for m in modes},
 median_ms={f'{a}_{m}':float(np.median(sums[a,m])) for a in arms for m in modes},
 diagnostic_runtime_ratio={a:float(np.median(sums[a,'split']/sums[a,'fused'])) for a in arms},
 max_state_gap=max(t.get('max_state_gap',0.) for t in x['timings']),caveats=x['caveats'])
summary['outlier_counts_tukey']={}
for (a,m),v in sums.items():
 q1,q3=np.quantile(v,[.25,.75]);iqr=q3-q1
 summary['outlier_counts_tukey'][f'{a}_{m}']=int(np.sum((v<q1-1.5*iqr)|(v>q3+1.5*iqr)))
summary['excluded_repetitions']=0
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,6),layout='constrained')
colors=['#757575','#b7bdc8','#d87916','#9a749d','#21835b','#2265a8','#dfdfdf']
labels=['Initial fit','History + warm-start prediction','16-coordinate solve','Transfer + acceptance check','64-coordinate solve','Output decoding + recording','Host loop / other']
bottom=np.zeros(2)
for p,color,label in zip(phases,colors,labels):
 vals=np.array([stages[a][p] for a in arms]);ax.bar([0,1],vals,bottom=bottom,color=color,label=label,width=.55)
 for i,v in enumerate(vals):
  if v>max(np.median(sums[a,'split']) for a in arms)*.055:ax.text(i,bottom[i]+v/2,f'{v:.1f}',ha='center',va='center',fontsize=10,color='black' if p in ['host_other','predictor_history'] else 'white')
 bottom+=vals
for i,v in enumerate(bottom):ax.text(i,v+max(bottom)*.02,f'{v:.1f} ms',ha='center',weight='bold')
ax.set(xticks=[0,1],xticklabels=['Direct K=64','16 → 64'],ylabel='Milliseconds summed over all 6 trajectories',title='Measured stages in synchronized diagnostic',ylim=(0,max(bottom)*1.15))
ax.legend(loc='upper left',bbox_to_anchor=(0,-.10),frameon=False,ncols=2,fontsize=9)
for mode,off,color,label in [('fused',-.18,'#2265a8','Original fused execution'),('split',.18,'#9a749d','Synchronized diagnostic')]:
 vals=np.array([np.median(sums[a,mode]) for a in arms]);pos=np.array([0,1])+off
 bx.bar(pos,vals,width=.32,color=color,label=label)
 for i,a in enumerate(arms):
  v=sums[a,mode];bx.scatter(pos[i]+np.linspace(-.07,.07,7),v,c='#333333',s=13,zorder=5)
  bx.text(pos[i],v.max()+max(bottom)*.02,f'{vals[i]:.1f}',ha='center')
bx.set(xticks=[0,1],xticklabels=['Direct K=64','16 → 64'],ylabel='Milliseconds summed over all 6 trajectories',title='Original versus stage-timed execution',ylim=(0,max(v.max() for v in sums.values())*1.15))
bx.legend(loc='upper left',bbox_to_anchor=(0,-.10),frameon=False,fontsize=10)
fig.suptitle('Where the time goes: direct K=64 versus 16 → 64',fontsize=17)
fig.supxlabel(f"Job {x['job_id']} · same GPU · 7 repetitions · 50 time steps per trajectory\nStage timing includes dispatch/synchronization and omits alternation monitoring.\nIt is not a decomposition of the fused run. Setup/compilation excluded.",fontsize=10)
fig.savefig(OUT/'stage-time-breakdown.png',dpi=170);fig.savefig(OUT/'stage-time-breakdown.pdf')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
