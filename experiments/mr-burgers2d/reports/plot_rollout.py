"""Two explicit accuracy norms from the same complete-query invocations."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_rollout import summarize
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args();d=json.loads(Path(a.input).read_text())
summary,_=summarize(d);meshes=sorted({r['intervals'] for r in summary})
fig,axes=plt.subplots(2,len(meshes),figsize=(5*len(meshes),8),sharey=True,squeeze=False)
styles={'fom':('#2468a2','o','FOM, including coarser grids'),'edge':('#cd6634','s','ROM, original edge fitting'),'fixed_gauss':('#198761','D','ROM, fixed physical Gauss')}
for row,norm in enumerate(['common','dense']):
    for ax,L in zip(axes[row],meshes):
        met=d['reference_metrics_by_output'][str(d['observation_intervals'] if norm=='common' else L)]
        margin={r['case']:r['conservative_difference_sum'] for r in met['uncertainty']}
        for r in met['order_audit']:
            if r['empirical_richardson_estimate'] is not None:margin[r['case']]=max(margin[r['case']],r['empirical_richardson_estimate'])
        seen=set()
        for r in summary:
            if r['intervals']!=L or not r['valid']:continue
            key='fom' if r['method']=='fom' else r['cold_rule'];color,marker,label=styles[key]
            y=max(e+margin[c] for c,e in r[norm]['case_errors'].items())
            ax.scatter(r['ms'],y,s=52 if key=='fom' else 70,c=color,marker=marker,edgecolors='white',linewidths=.6,label=label if key not in seen else None)
            if d['experiment']=='gauss_timestep_study' and key!='fom':
                ax.annotate(f"dt={r['dt']:g}",(r['ms'],y),xytext=(4,5),textcoords='offset points',fontsize=7,color=color)
            seen.add(key)
        for target in [.1,.05,.01]:ax.axhline(target,color='#aeb7c1',lw=.7,ls='--',zorder=0)
        ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.15)
        title=f'{L} requested intervals · '+(f"common {d['observation_intervals']}-interval norm" if norm=='common' else 'complete-grid norm')
        ax.set_title(title,fontsize=10);ax.set_xlabel('Complete query, case-median milliseconds')
    axes[row,0].set_ylabel('Worst case error + empirical reference margin')
handles,labels=axes[1,-1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
study='time-step study with frozen Gauss initialization' if d['experiment']=='gauss_timestep_study' else 'frozen Gauss initializer in full rollout'
fig.suptitle(f"Burgers 2D · {study} · job {d['job_id']}")
fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(a.output,metadata={'Date':None});fig.savefig(Path(a.output).with_suffix('.png'),dpi=170);plt.close(fig)
if Path(a.output).suffix=='.svg':
    Path(a.output).write_text('\n'.join(line.rstrip() for line in Path(a.output).read_text().splitlines())+'\n')
print(a.output)
