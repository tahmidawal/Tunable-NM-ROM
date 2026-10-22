"""Cost and solver iterations against mesh, per bank width r and time scheme (2D, P1 timing job 5352971, H200).
Source: worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/time/archive/output/p1time_time/result.json
(invocations: t_evolve, t_decode, it per subject x case x repeat). Frozen-mode subjects only. Writes mesh_cost_iters.png + mesh_cost_iters.json."""
import json, collections, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
SRC='/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/time/archive/output/p1time_time/result.json'
d=json.load(open(SRC)); assert d['backend']=='gpu' and d['complete']
HERE=os.path.dirname(os.path.abspath(__file__))
acc=collections.defaultdict(lambda: dict(ev=[],de=[],it=[]))
for inv in d['invocations']:
    if not inv.get('valid'): continue
    mesh,R,rule,pol,sch=inv['B'].split('|')
    if pol!='frozen': continue
    a=acc[(R,sch,int(mesh))]; a['ev'].append(inv['t_evolve']*1e3); a['de'].append(inv['t_decode']*1e3); a['it'].append(sum(inv['it']))
out={}
for (R,sch,N),a in sorted(acc.items()):
    out[f'{R}|{sch}|{N}']=dict(n=len(a['ev']),evolve_ms_median=float(np.median(a['ev'])),decode_ms_median=float(np.median(a['de'])),
        iters_per_run_mean=float(np.mean(a['it'])),iters_per_run_min=int(min(a['it'])),iters_per_run_max=int(max(a['it'])))
json.dump(dict(source=SRC,job=d['job_id'],gpu=d['gpu'],commit=d['commit'],steps_per_run=50,rows=out),open(os.path.join(HERE,'mesh_cost_iters.json'),'w'),indent=1)
cols={'R128':'#1F6F8B','R256':'#2E6B4F','R384':'#C8641E','R512':'#8A6A12'}; ls={'BE':'-','CNR':'--'}; mk={'BE':'o','CNR':'s'}
fig,ax=plt.subplots(1,3,figsize=(16,5.8))
for (R,sch) in sorted({(k.split('|')[0],k.split('|')[1]) for k in out}):
    Ns=sorted(int(k.split('|')[2]) for k in out if k.startswith(f'{R}|{sch}|'))
    g=lambda f:[out[f'{R}|{sch}|{N}'][f] for N in Ns]
    lab=f"r={R[1:]} {'BE' if sch=='BE' else 'CN-R'}"
    for i,f in enumerate(['evolve_ms_median','iters_per_run_mean','decode_ms_median']):
        ax[i].plot(Ns,g(f),ls[sch],marker=mk[sch],color=cols[R],label=lab,ms=7)
for a,t,y in zip(ax,['Solve (evolve) time','LM iterations per run (50 steps)\n(Jacobian builds = 50 + this)','Output decode time (6 fields)'],['ms, median','iterations, mean over cases','ms, median']):
    a.set_xscale('log',base=2); a.set_xticks([256,1024,4096]); a.set_xticklabels(['256²','1024²','4096²']); a.set_title(t,fontsize=13); a.set_ylabel(y); a.set_xlabel('mesh'); a.grid(alpha=.3)
ax[1].set_ylim(45,60); ax[1].axhline(50,color='#999',lw=1,ls=':'); ax[1].text(300,50.3,'1 iteration per step',color='#777',fontsize=9)
ax[2].set_yscale('log'); ax[0].set_ylim(bottom=0)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=8,fontsize=11,frameon=False)
fig.suptitle('2D Burgers, frozen test modes: solve time and iterations flat in the mesh for r = 128–384 (r = 512 timed at 1024² only); only decode grows. H200, 6 dev cases × 3 repeats',fontsize=13)
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(os.path.join(HERE,'mesh_cost_iters.png'),dpi=150)
for k,v in out.items(): print(k,v)
