"""Standalone scientific plots from generated summary only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
d=json.loads((a.run/'summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,len(d['config']['intervals']),figsize=(10,4),sharey=True,layout='constrained')
for ax,n in zip(axs,d['config']['intervals']):
    for s in d['summaries']:
        if s['intervals']!=n:continue
        label=s['arm']+(f" τ={s['tau']:g}" if s['tau'] is not None else '')
        ax.scatter(s['median_seconds']*1000,s['physical_max'],s=55,label=label)
    for target in d['config']['targets']:ax.axhline(target,color='grey',alpha=.25,ls='--')
    ax.set_title(f'{n} intervals / {n+1} nodes per axis')
    ax.set_yscale('log');ax.set_xlabel('Median full host-to-host query (ms)')
    ax.grid(alpha=.15)
axs[0].set_ylabel('Worst development physical relative error')
axs[-1].legend(fontsize=8,loc='lower right')
fig.suptitle('Frozen Poisson model versus direct sine-transform solvers')
fig.savefig(a.run/'accuracy-cost.png',dpi=180)
fig.savefig(a.run/'accuracy-cost.pdf')
plt.close(fig)
