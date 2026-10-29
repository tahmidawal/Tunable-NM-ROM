"""Actual within-job full-query timings, no inferred stage timing."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
src=p.parent/'checks/source-results/step.json'
x=json.loads(src.read_text())
t={(r['arm'],r['case'],r['block']):1000*r['seconds'] for r in x['timings']}
a={arm:np.array([[t[arm,c,b] for b in range(7)] for c in range(6)]) for arm in ['direct','affine']}
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
for arm,offset,color,label in [('direct',-.19,'#2265a8','Direct K=64'),('affine',.19,'#21835b','16-coordinate stage → K=64')]:
    v=a[arm];med=np.median(v,axis=1);pos=np.arange(1,7)+offset
    ax.bar(pos,med,width=.35,color=color,alpha=.85,label=label)
    ax.errorbar(pos,med,yerr=[med-v.min(1),v.max(1)-med],fmt='none',ecolor='#333333',capsize=4,lw=1)
    for i in range(6):
        ax.scatter(pos[i]+np.linspace(-.10,.10,7),v[i],s=16,c='#333333',alpha=.65,zorder=4)
        ax.text(pos[i],v[i].max()+2,f'{med[i]:.1f}',ha='center',fontsize=11)
ax.set(xlabel='Development case',ylabel='Full trajectory query time (milliseconds)',xticks=np.arange(1,7),ylim=(0,max(v.max() for v in a.values())*1.2),title='Measured time: direct K=64 versus 16 → 64\nBars: medians · dots: all 7 repetitions · whiskers: minimum–maximum')
ax.legend(loc='upper left',frameon=False)
fig.supxlabel('Job 6619857 · same H200 GPU · 50 time steps per query\nIncludes initial fitting, solves, guards and field decoding; excludes compilation/setup. Stage times were not recorded.',fontsize=10)
fig.savefig(p/'runtime-ms.png',dpi=170);fig.savefig(p/'runtime-ms.pdf')
s=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),unit='milliseconds',repetitions={k:v.tolist() for k,v in a.items()},case_medians={k:np.median(v,axis=1).tolist() for k,v in a.items()},stage_times_available=False)
(p/'runtime-ms.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s['case_medians']))
