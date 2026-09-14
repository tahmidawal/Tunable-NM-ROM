"""Standalone, data-generated training and common-observation accuracy figures."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
d=json.loads((run/'summary.json').read_text());cfg=d['config'];n=max(cfg['intervals'])
models=cfg['arms'];colors=plt.get_cmap('tab10').colors
labels={name:name.replace('_','\n') for name in models}
fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=True,layout='constrained')
for ax,cohort in zip(axs,('existing_development','fresh_development')):
    rows=[next(s for s in d['summaries'] if s['intervals']==n and s['cohort']==cohort and s['model']==model and s['arm']=='rom_modular' and s['tau']==0) for model in models]
    values=np.array([[c['physical_error'] for c in r['cases']] for r in rows])
    for i in range(values.shape[1]):ax.plot(range(len(models)),values[:,i],color='.75',lw=.7,alpha=.55,zorder=1)
    for i,(model,row) in enumerate(zip(models,rows)):
        for case in row['cases']:
            marker='o' if case['solver_valid'] else 'x'
            ax.scatter(i,case['physical_error'],color=colors[i],s=24,marker=marker,zorder=2)
        ax.scatter(i,np.median(values[i]),color='black',marker='_',s=180,zorder=3)
    for eps in cfg['targets']:ax.axhline(eps,color='.55',lw=.6,ls=':')
    ax.set_yscale('log');ax.set_xticks(range(len(models)),[labels[m] for m in models],fontsize=8)
    ax.set_title(cohort.replace('_',' ').capitalize());ax.grid(axis='y',alpha=.15)
    ax.set_xlabel('Fixed final checkpoint')
axs[0].set_ylabel('Common-observation relative field discrepancy')
fig.suptitle(f"Poisson stationary generic control, {n} intervals\nPoints: sources; black marks: medians; crosses: solver-invalid outputs")
for ext in ('png','pdf'):fig.savefig(run/('accuracy-factorial.'+ext),dpi=180)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,objective in zip(axs,('global','relative')):
    for i,coverage in enumerate(('original','expanded')):
        record=next(r for r in d['training']['arms'] if r['tag']==coverage+'_'+objective)
        ax.plot([r['step'] for r in record['progress']],[r['loss'] for r in record['progress']],label=coverage+' coverage',color=colors[i])
    ax.set_yscale('log');ax.set_xlabel('Optimizer updates');ax.set_title(objective.capitalize()+' normalization')
    ax.grid(alpha=.15);ax.legend(frameon=False)
axs[0].set_ylabel('Sampled training objective (including Gram penalty)')
fig.suptitle('Fixed continuation schedules; training loss is not validation error')
for ext in ('png','pdf'):fig.savefig(run/('training-objectives.'+ext),dpi=180)
plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,5),layout='constrained')
markers={'rom_modular':'o','rom_fused':'s','rom_gj':'^'}
for i,model in enumerate(models):
    rs=[r for r in d['summaries'] if r['intervals']==n and r['cohort']=='all_development' and r['model']==model]
    for row in rs:
        valid=not row['invalid_count'] and row['valid_endpoint']
        ax.scatter(row['latency_seconds']*1000,max(c['adjusted_error'] for c in row['cases']),
            marker=markers[row['arm']],color=colors[i],s=50,facecolors=colors[i] if valid else 'none')
    ax.scatter([],[],color=colors[i],label=model.replace('_',' '))
for arm in ('dst','dst_coarse128'):
    r=next(s for s in d['summaries'] if s['intervals']==n and s['cohort']=='all_development' and s['arm']==arm)
    ax.scatter(r['latency_seconds']*1000,max(c['adjusted_error'] for c in r['cases']),marker='*',color='black',s=110)
    ax.annotate(arm,(r['latency_seconds']*1000,max(c['adjusted_error'] for c in r['cases'])),xytext=(4,4),textcoords='offset points',fontsize=8)
for eps in cfg['targets']:ax.axhline(eps,color='.6',lw=.6,ls=':')
ax.set_yscale('log');ax.set_xlabel('Complete-query latency, median of case medians (ms)')
ax.set_ylabel('Worst empirically reference-adjusted error')
ax.set_title(f"Poisson development cost and error, {n} intervals\nCircle: modular; square: fused; triangle: guarded kernel; hollow: invalid")
ax.legend(frameon=False,fontsize=8,loc='best');ax.grid(alpha=.15)
for ext in ('png','pdf'):fig.savefig(run/('cost-error-factorial.'+ext),dpi=180)
plt.close(fig)
