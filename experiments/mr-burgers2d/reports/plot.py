"""Plot complete-query accuracy/cost from one GPU job's retained invocations."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args();d=json.loads(Path(a.input).read_text())
group=defaultdict(list)
for r in d['invocations']:group[r['name']].append(r)
meshes=sorted(set(r['output_intervals'] for r in d['invocations']))
margin=max(r['conservative_difference_sum'] for r in d['reference_uncertainty'])
rich=[r['empirical_richardson_estimate'] for r in d.get('reference_order_audit',[]) if r['empirical_richardson_estimate'] is not None]
if rich:margin=max(margin,max(rich))
fig,axs=plt.subplots(1,len(meshes),figsize=(5*len(meshes),4.6),sharey=True,squeeze=False)
colors={'fom':'#2468a2','rom1':'#cd6634','rom4':'#198761'}
for ax,L in zip(axs[0],meshes):
    seen=set()
    for name,rows in group.items():
        r=rows[0]
        if r['output_intervals']!=L:continue
        key='fom' if r['method']=='fom' else 'rom'+str(r.get('ic_starts',1))
        label={'fom':'FOM, including coarser grids','rom1':'ROM, one initial guess','rom4':'ROM, four initial guesses'}[key]
        y=max(x['physical_error']['fixed_initial_max'] for x in rows)+margin
        x=1000*np.median([x['seconds'] for x in rows])
        ax.scatter(x,y,s=52 if key=='fom' else 70,c=colors[key],marker='o' if key=='fom' else 'D',edgecolors='white',linewidths=.6,label=label if key not in seen else None)
        seen.add(key)
    for target in [.1,.05,.01]:ax.axhline(target,color='#aeb7c1',lw=.7,ls='--',zorder=0)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_title(f'{L} intervals: dense requested output')
    ax.set_xlabel('Complete query, median milliseconds');ax.grid(alpha=.15)
axs[0,0].set_ylabel('Worst validation error + empirical reference margin')
handles,labels=axs[0,-1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
fig.suptitle(f"Burgers 2D · frozen K{d['mesh_setup'][0]['K']}/R{d['mesh_setup'][0]['R']} · job {d['job_id']}")
fig.tight_layout(rect=(0,.10,1,.93));fig.savefig(a.output);fig.savefig(Path(a.output).with_suffix('.png'),dpi=170);plt.close(fig)
print(a.output)
