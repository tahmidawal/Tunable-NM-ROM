"""Plot retained checkpoints only; no solver execution or interpolated iteration trace."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT.parent / '2026-10-28-burgers-shared-step/experiments/burgers-shared-step/artifacts/screen01/output/dev6'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
blue, orange, green = '#2265a8', '#d87916', '#21835b'
rows = []
source_hashes = {}
for c in range(6):
    paths = [SOURCE / f'{a}_case{c}.npz' for a in ('direct', 'affine')]
    for p in paths:
        source_hashes[str(p.relative_to(ROOT.parent))] = hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(paths[0]) as d, np.load(paths[1]) as a:
        # At the first BE step both objectives have precisely the same history and IC.
        np.testing.assert_array_equal(d['states'][0], a['states'][0])
        np.testing.assert_array_equal(a['seed_codes'][0, :64], a['states'][0])
        si = a['seed_info']
        rows.append(dict(case=c+1, direct_iterations=int(d['step_it'][0]),
            coarse_iterations=int(si[0,1]), refine_iterations=int(a['step_it'][0]),
            initial_loss=float(si[0,7]), coarse_loss=float(si[0,8]),
            direct_final_loss=float(d['residual_norm_fresh'][0]**2),
            refine_final_loss=float(a['residual_norm_fresh'][0]**2),
            direct_total=int(d['step_it'].sum()), coarse_total=int(si[:,1].sum()),
            refine_total=int(a['step_it'].sum()),
            coarse_reduction_fraction_median=float(np.median(1-si[:,8]/si[:,7]))))
fig, axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
for ax,r in zip(axes.flat,rows):
    n, nc, nf = r['direct_iterations'], r['coarse_iterations'], r['refine_iterations']
    ax.scatter([0],[r['initial_loss']],color='#555555',marker='D',s=65,label='Shared starting loss',zorder=5)
    ax.scatter([n],[r['direct_final_loss']],color=blue,marker='o',s=100,label='Direct K=64: finished',zorder=5)
    ax.scatter([nc],[r['coarse_loss']],color=orange,marker='s',s=80,label='After 16-coordinate stage',zorder=5)
    ax.scatter([nc+nf],[r['refine_final_loss']],color=green,marker='^',s=100,label='After K=64 refinement',zorder=5)
    ax.set(yscale='log',xlabel='Cumulative solver updates (not elapsed time)',ylabel='Loss = squared weighted residual norm',
           title=f"Case {r['case']}: direct {n} updates; staged {nc} + {nf}",xlim=(-.7,max(n,nc+nf)+1))
    ax.set_xticks(sorted(set([0,n,nc,nc+nf])))
    ax.grid(alpha=.16,which='both')
fig.suptitle('Direct K=64 versus 16-coordinate initialization → K=64\nFirst time step: identical starting state and objective\nSaved checkpoints only; intermediate iteration losses were not recorded (job 6619857)',fontsize=14)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
fig.savefig(HERE/'loss-checkpoints.png',dpi=160)
fig.savefig(HERE/'loss-checkpoints.pdf')
plt.close(fig)

result=json.loads((SOURCE/'result.json').read_text())
totals={k:sum(r[k] for r in rows) for k in ['direct_total','coarse_total','refine_total']}
ratios=np.array(result['gate']['per_case_paired_median_ratios'])
fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
labels=['Direct64','16 → 64']
d,c,f=[totals[k] for k in ['direct_total','coarse_total','refine_total']]
for ax,vals,title in [(axes[0],(d,c,f),'Solver updates'),(axes[1],(d+300,c+300,f+300),'Jacobian evaluations')]:
    vd,vc,vf=vals
    ax.bar(0,vd,color=blue); ax.bar(1,vc,color=orange,label='16-coordinate stage'); ax.bar(1,vf,bottom=vc,color=green,label='64-coordinate refinement')
    ax.text(0,vd/2,str(vd),ha='center',color='white',fontsize=14)
    ax.text(1,vc/2,str(vc),ha='center',color='white',fontsize=14)
    ax.text(1,vc+vf/2,str(vf),ha='center',color='white',fontsize=14)
    ax.text(1,vc+vf,str(vc+vf),ha='center',va='bottom')
    ax.set(xticks=[0,1],xticklabels=labels,ylabel='Count across 6 cases × 50 time steps',title=title,ylim=(0,max(vd,vc+vf)*1.15))
axes[1].legend(frameon=False,loc='upper left',fontsize=9)
axes[2].bar(np.arange(1,7),ratios,color=green)
axes[2].axhline(1,color=blue,lw=2,label='Direct64 baseline')
for i,r in enumerate(ratios,1): axes[2].text(i,r+.025,f'{r:.2f}×',ha='center')
axes[2].set(xlabel='Development case',ylabel='Staged / direct runtime',title='Median paired full-query runtime',ylim=(0,1.75),xticks=np.arange(1,7))
axes[2].legend(frameon=False,loc='lower right')
fig.suptitle('The smaller stage saves fine updates, but adds more total work',fontsize=17)
fig.savefig(HERE/'work-breakdown.png',dpi=160);fig.savefig(HERE/'work-breakdown.pdf');plt.close(fig)
summary=dict(source_job=6619857,loss='squared norm of existing weighted weak residual; not physical solution error',
    scope='First time step only for loss comparisons: identical IC/history/objective. Saved endpoints, not per-iteration traces.',
    coarse_stage='First16 coordinates optimized; last48 held at original predictor, not standalone zero-tail D16.',
    rows=rows,totals=totals,median_paired_runtime_ratio=float(np.median(ratios)),
    cost_exclusions='Update/J bars exclude initial fitting and guards. Full-query timings include them; 600 extra guard residuals.',
    hashes=source_hashes)
(HERE/'loss-checkpoints.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
