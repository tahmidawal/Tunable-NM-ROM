"""Standalone complete-query and paired-factorial figures from audited JSON."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run
d=json.loads((run/'summary.json').read_text());cfg=d['config'];assert d['audit']['passed']
colors=plt.get_cmap('tab10').colors
fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout='constrained')
for ax,n in zip(axs,cfg['intervals']):
    for s in d['summaries']:
        if s['intervals']!=n or s['cohort']!='all_development' or s['panel']!='primary':continue
        if s['model']:
            color=colors[cfg['models'].index(s['model'])];marker='o' if s['initialization']==cfg['initialization'][0] else 'D'
            filled=s['projection']==cfg['projection'][1]
            ax.scatter(1000*s['latency_seconds'],s['adjusted_max'],s=70,marker=marker,
                facecolors=color if filled else 'none',edgecolors=color)
            if s['invalid_count']:ax.scatter(1000*s['latency_seconds'],s['adjusted_max'],marker='x',color='black',s=90)
        else:
            ax.scatter(1000*s['latency_seconds'],s['adjusted_max'],marker='*',s=110,color='black')
            ax.annotate(s['arm'],(1000*s['latency_seconds'],s['adjusted_max']),xytext=(4,4),textcoords='offset points',fontsize=8)
    for eps in cfg['targets']:ax.axhline(eps,color='.6',lw=.6,ls=':')
    ax.set_yscale('log');ax.set_xlabel('Full query, median of case medians (ms)');ax.set_title(f'{n} intervals');ax.grid(alpha=.15)
for i,model in enumerate(cfg['models']):axs[0].scatter([],[],color=colors[i],label=model.replace('_',' '))
axs[0].legend(frameon=False,fontsize=8);axs[0].set_ylabel('Worst empirically adjusted physical error')
fig.suptitle('Poisson development accuracy and cost\nCircle: mean start; diamond: nearest; filled: DST projection; hollow: sine products; cross: invalid')
for ext in ('png','pdf'):fig.savefig(run/('speed-cost-error.'+ext),dpi=180)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
names=('thin_over_dst_projection','mean_over_nearest_initialization')
for ax,contrast in zip(axs,names):
    items=[r for r in d['factorial_cost_contrasts'] if r['contrast']==contrast]
    labels=[]
    for i,r in enumerate(items):
        values=np.asarray(r['per_case_cost_ratios']);jitter=np.linspace(-.17,.17,len(values))
        color=colors[cfg['models'].index(r['model'])]
        ax.scatter(values,i+jitter,s=13,color=color,alpha=.65)
        ax.scatter(r['median_case_cost_ratio'],i,marker='|',s=150,color='black')
        held={'mean_training_code':'mean','nearest_cached_scaled_weak_prediction':'nearest',
            'skinny_sine_products':'sine products','forward_dst_and_gather':'DST gather'}[r['held_fixed']]
        labels.append(f"{r['intervals']} / {r['model'].replace('original_','')} / {held}")
    ax.set_yticks(range(len(items)),labels,fontsize=8);ax.invert_yaxis();ax.axvline(1,color='.5',ls=':',lw=.8);ax.grid(axis='x',alpha=.15)
    ax.set_title('Projection change' if contrast==names[0] else 'Initialization + existing stop')
    ax.set_xlabel('Sine-product / DST-projection query ratio' if contrast==names[0] else 'Mean-start / nearest-start query ratio')
fig.suptitle('Paired complete-query effects: points are case-median ratios; black marks are medians\nRatios above one favor the candidate procedure; accuracy and failure gates remain separate')
for ext in ('png','pdf'):fig.savefig(run/('speed-paired-effects.'+ext),dpi=180)
plt.close(fig)
