"""Publication-style standalone figures from follow-up JSON."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
d=json.loads((a.run/'summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,n in zip(axes,d['config']['intervals']):
    selected=sorted((s for s in d['summaries'] if s['intervals']==n and s['arm']=='rom_modular' and s['tau']==0),key=lambda s:s['requested_modes'])
    ax.plot([s['retained_modes'] for s in selected],[s['same_grid_max'] for s in selected],'o-',label='Worst weak-solve error')
    oracle=[o for o in d['oracles'] if o['intervals']==n]
    ax.axhline(max(o['full_bank_same_grid_error'] for o in oracle),color='#287a4b',ls='--',label='Worst full-bank projection')
    ax.axhline(max(o['best_same_grid_error'] for o in oracle),color='#ab4b19',ls=':',label='Worst best head oracle')
    ax.axhline(.05,color='grey',alpha=.5,label='5% target')
    ax.set_title(f'{n} intervals per axis');ax.set_xlabel('Actual retained weak modes');ax.grid(alpha=.15)
axes[0].set_ylabel('Worst development same-grid relative error');axes[-1].legend(fontsize=8)
fig.savefig(a.run/'testspace-and-representation.png',dpi=180);fig.savefig(a.run/'testspace-and-representation.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
colors={64:'#3274a1',128:'#e1812c',256:'#3a923a'}
for ax,n in zip(axes,d['config']['intervals']):
    for s in d['summaries']:
        if s['intervals']!=n:continue
        arm=s['arm'];color=colors.get(s['requested_modes'],'#444444')
        marker='o' if arm=='rom_fused' else ('x' if arm=='rom_modular' else 's')
        ax.scatter(s['latency_seconds']*1000,s['physical_max'],color=color,marker=marker,s=45)
    ax.axhline(.1,color='grey',ls='--',alpha=.4);ax.axhline(.05,color='grey',ls='--',alpha=.4)
    ax.set_yscale('log');ax.set_title(f'{n} intervals per axis');ax.set_xlabel('Median of source-median complete query (ms)');ax.grid(alpha=.15)
axes[0].set_ylabel('Worst development physical relative error')
fig.suptitle('Circles: fused ROM; crosses: segmented ROM; squares: direct DST')
fig.savefig(a.run/'accuracy-cost.png',dpi=180);fig.savefig(a.run/'accuracy-cost.pdf');plt.close(fig)
