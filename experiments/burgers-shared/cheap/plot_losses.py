"""Display separately recorded losses beside timings; no joint cost-accuracy claim."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;SRC=HERE/'evidence';OUT=HERE.parent/'plots/cheap'
x=json.loads((SRC/'result.json').read_text());assert x['complete']
arms=['direct','full_b1','full_b2'];names=['Direct K=64','1 coarse update → K=64','2 coarse updates → K=64']
colors=['#2265a8','#d87916','#21835b'];markers=['o','s','^'];data={};hashes={}
for arm in arms:
 losses=[]
 for c in range(6):
  path=SRC/f'{arm}_{c}.npz';hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path) as z:losses.append((z['residual_norm_fresh']**2).tolist())
 secs=[[1000*t['seconds'] for t in sorted(x['timings'],key=lambda t:t['block']) if t['arm']==arm and t['case']==c] for c in range(6)]
 assert all(len(v)==7 for v in secs)
 data[arm]=dict(losses=losses,runtime_ms=secs,median_runtime_ms=np.median(secs,axis=1).tolist())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained');ax,bx,cx,dx=axs.flat
for ai,(arm,label,color,marker) in enumerate(zip(arms,names,colors,markers)):
 pos=np.arange(1,7)+(ai-1)*.23
 ms=np.array(data[arm]['runtime_ms']);med=np.median(ms,axis=1)
 ax.bar(pos,med,width=.22,color=color,label=label)
 ax.errorbar(pos,med,yerr=[med-ms.min(1),ms.max(1)-med],fmt='none',ecolor='#444444',capsize=2,lw=.8)
 loss=np.array(data[arm]['losses'])[:,-1]
 bx.scatter(pos,loss,c=color,marker=marker,s=65,label=label)
ax.set(title='Completed solves: full-query runtime',xlabel='Development case',ylabel='Milliseconds (median; whiskers min–max)',xticks=range(1,7),ylim=(0,145))
ax.legend(frameon=False,fontsize=10)
bx.set(title='Completed solves: returned loss at t = 0.25',xlabel='Development case',ylabel=r'Fine solver loss $\|r_{50}\|_2^2$ (log scale)',xticks=range(1,7),yscale='log')
bx.grid(axis='y',alpha=.15,which='both')
for c in range(6):
 vals=[data[a]['losses'][c][-1] for a in arms]
 bx.text(c+1,max(vals)*1.23,f'{min(vals):.3g}–{max(vals):.3g}' if max(vals)/min(vals)>1.001 else f'≈ {np.median(vals):.4g}',ha='center',fontsize=9)
bx.set_ylim(.00025,.065)
configs=['fine64','full','q64','q32m64'];labels=['K64\n16,384 pts','K16\n16,384 pts','K16\n4,096 pts','K16\n1,024 pts; M64']
v=[np.median(next(c['seconds'] for c in x['components'] if c['config']==n and c['operation']=='full_residual_jacobian'))*1e6 for n in configs]
cx.bar(range(4),v,color=[colors[0],colors[2],'#bdbdbd','#bdbdbd'])
for i,y in enumerate(v):cx.text(i,y+8,f'{y:.1f}',ha='center')
cx.set(title='Isolated calculations: residual + Jacobian cost',ylabel='Microseconds per synchronized call',xticks=range(4),xticklabels=labels,ylim=(0,max(v)*1.2))
qnames=['q64','q32m64'];err=[100*x['qualification'][n]['max_relative_errors'][0] for n in qnames]
dx.bar([0,1],err,color='#bdbdbd',width=.5)
for i,y in enumerate(err):dx.text(i,y+.08,f'{y:.2f}%',ha='center')
dx.axhline(.1,color='#a33a32',linestyle='--',label='Registered panel threshold: 0.1%')
dx.set(title='Reduced quadrature: residual discrepancy, NOT final loss',ylabel='Worst relative residual difference (%)',xticks=[0,1],xticklabels=['K16 / 4,096 points','K16 / 1,024 points; M64'],ylim=(0,4.5))
dx.text(.5,3.9,'No completed trajectories → final losses unavailable',ha='center',fontsize=10)
dx.legend(frameon=False,fontsize=9)
fig.suptitle('Burgers 16 → 64: runtime and recorded losses',fontsize=19)
fig.supxlabel('Job 6647637 · Top: unchanged full-quadrature fine target; each trajectory uses its own accepted history.\nRuntime: 7 repetitions/case. Loss: separate diagnostic trajectories with matching returned fields; no joint cost–accuracy claim.\nBottom: component timing at case1 initial state; residual discrepancies over 30 states against SAME-mode full quadrature. Loss is not reference-solution error.',fontsize=10)
fig.savefig(OUT/'timing-and-losses.png',dpi=170);fig.savefig(OUT/'timing-and-losses.pdf');plt.close(fig)
summary=dict(source_result_sha256=hashlib.sha256((SRC/'result.json').read_bytes()).hexdigest(),hashes=hashes,arms=data,loss_definition='Squared norm of fresh weighted weak residual at each returned step; not physical error.',limitations='Timings and losses from separate matched-output runs, displayed separately. No within-iteration losses or final trajectories for reduced rules.')
(OUT/'timing-and-losses.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Generated timing-and-losses.png/pdf/json')
