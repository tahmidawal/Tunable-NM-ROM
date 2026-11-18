"""Generate every reported number and plot from retained evidence, never hand tables."""
from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;out=H/'reports';out.mkdir(exist_ok=True)
G=json.loads((H/'results/global01/output/result.json').read_text())
D=json.loads((H/'results/diag01/output/result.json').read_text())
C=json.loads((H/'results/children01/output/result.json').read_text())
L=json.loads((H/'results/local01/output/result.json').read_text())
selected=next(c for c in C['checkpoints'] if c['update']==C['selected_update'])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,ax=plt.subplots(figsize=(8,5))
def curve(xs,ys,label,color,style='-'):
    ax.plot(xs,100*np.asarray(ys),style+'o',label=label,color=color,lw=1.7,ms=4)
curve([16,32,64,128],[G['global_direct'][str(k)]['worst'] for k in (16,32,64,128)],'Global learned direct bank','#355d8c')
curve([16,32,64],[G['heads'][str(k)]['worst_projection'] for k in (16,32,64)],'Global nonlinear Tunable','#30343b')
curve([16,32,64,128],[G['pod_control'][str(k)]['worst'] for k in (16,32,64,128)],'Training-only POD span','#8a8f97',':')
for kind,color,style in [('uniform','#bd8141','--'),('prescribed','#bd8141','-')]:
    rows=[r for r in D['comparisons'] if r['rank']==4 and r['allocation']==kind]
    curve([r['active_variables'][0] for r in rows],[r['projection']['worst'] for r in rows],f'Direct local / {kind}',color,style)
for r in selected['comparisons']:
    ax.scatter(r['active_variables'][0],100*r['projection']['worst'],marker='D',s=45,color='#5a7e55',
               label=f'Nonlinear children / {r["allocation"]}')
ax.axhline(.5,color='#a53c3c',lw=1,ls=':',label='0.5% advancement gate')
ax.set(xlabel='Total independently adjustable variables',ylabel='Worst relative L2 projection error (%)',
       title='First representation comparison — development and alignment')
ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.15);fig.tight_layout()
fig.savefig(out/'representation-capacity.png');fig.savefig(out/'representation-capacity.svg');plt.close(fig)
w=np.load(H/'results/diag01/output/witness.npz');x=w['x'];ref=w['reference'];root=w['root_before'];joint=w['joint_root'];child=w['children'];pred=w['reconstruction']
fig,axes=plt.subplots(2,3,figsize=(11,6.6),layout='constrained')
items=[(ref,'Reference field',False),(root,'Parent before adding children',False),(root-ref,'Parent signed error',True),
       (joint,'Jointly refitted parent',False),(child,'Sum of child corrections',True),(pred-ref,'Remaining signed error',True)]
for ax,(v,title,signed) in zip(axes.flat,items):
    lim=np.max(np.abs(v));im=ax.pcolormesh(x,x,v.T,shading='auto',cmap='RdBu_r' if signed else 'viridis',
             vmin=-lim if signed else 0,vmax=lim if signed else float(ref.max()))
    ax.set(title=title,xlabel='x',ylabel='y',aspect='equal');fig.colorbar(im,ax=ax,shrink=.82,label='u')
fig.suptitle(f'{str(w["case"])} at t={float(w["time"]):.3f}: optimized direct local reconstruction\n64 × 64 Gauss points; current-time relative L2 error {100*float(w["normalized_error"]):.3f}%')
fig.savefig(out/'parent-children-witness.png');fig.savefig(out/'parent-children-witness.svg');plt.close(fig)
rows=[]
for k in (16,32,64):rows.append((f'Global nonlinear k{k}',k,G['heads'][str(k)]['worst_projection']))
rows.append(('Global direct128',128,G['global_direct']['128']['worst']))
for r in D['comparisons']:
    if r['cap']==128:rows.append((f'Direct local r{r["rank"]}, {r["allocation"]}',r['active_variables'][0],r['projection']['worst']))
for r in selected['comparisons']:rows.append((f'Nonlinear child4, {r["allocation"]}',r['active_variables'][0],r['projection']['worst']))
table='\n'.join(f'| {name} | {k} | {100*err:.5f}% |' for name,k,err in rows)
span='\n'.join(f'| {r["kind"]} | {r["free_linear_coefficients"]} | {100*r["projection"]["worst"]:.5f}% |' for r in D['optimistic_span_diagnostics'])
text=f'''# Burgers tree embeddings: first representation evidence

Provisional development screening for the new study. These results concern reconstructed fields, not PDE trajectories or complete solve speed. All numbers below are generated from saved job evidence.

**Decision: do not advance these checkpoints to PDE performance claims.** No tested tree passes the all-snapshot 0.5% representation gate. R1 is registered to repair the training-code optimization gap before interpreting the early stop as a learning plateau. Final-test fields remain unopened.

![Representation capacity](representation-capacity.png)

| Method | Active variables | Worst development/alignment error |
|---|---:|---:|
{table}

Errors use the current snapshot's reference L2 norm. Panels contain16 random development trajectories and12 alignment trajectories, with11 snapshots each. These first comparisons use64-by64 Gauss integration; independent finer scoring and actual4096 output are being evaluated separately. Uniform and prescribed allocations share the same local checkpoint and parent. Prescribed trees use only the known initial field. A best-found nonlinear fit is not a certified global optimum.

![Parent and corrections](parent-children-witness.png)

The parent is refitted jointly with children, so the summed child correction adds to the refitted parent shown separately. Colors show physical field/error values; they are not a pointwise1% guarantee.

Even optimistic span diagnostics fail0.5% on the initial checkpoint. These controls release the parent to128 linear coefficients and every child to16 coefficients. They exceed the eligible128-variable budget and are used only to diagnose feature limitations.

| Optimistic local features | Free linear coefficients | Worst error |
|---|---:|---:|
{span}

The initial bank selected update{G['selected_bank_update']}; the direct local features selected update{L['selected_update']}; the nonlinear child head selected update{C['selected_update']}. Best and final checkpoints, raw fitting statuses and unsuccessful recipes are retained. The initial training optimized codes with Adam; a subsequent frozen-checkpoint audit found materially better training fits under explicit code refitting. This limits claims about training convergence, while preserving the measured finite-checkpoint failures.

The primary fine-upwind tested weak residual has an independent dense fixture. No fixed-tree trajectory, deployable adaptation, faithful CROM/SNF-ROM/masked-autoencoder confirmation, final-test result, or complete runtime claim has been produced at this stage. Those stages remain gated by representation evidence.

## Evidence and glossary

Sources: jobs7351850 (global),7351946 (direct local),7352069 (nonlinear children),7352154 (diagnostics). Raw arrays and file hashes are in the lane's results and manifests directories. Seed2026111851; float64/highest precision on verified A100 allocations. No timing is compared across allocations.

- Active variables: independently adjustable parent and retained-child numbers.
- Projection error: best reconstruction found with the reference field available, distinct from a PDE solve.
- Worst error: maximum over every listed trajectory and snapshot, including the alignment panel.
- Uniform: a complete tree at the deepest level fitting the variable cap.
- Prescribed: a connected tree chosen from the initial-field callable; no future solution access.
- Span diagnostic: a deliberately more permissive linear fit used to test whether a checkpoint contains useful field directions.
- R1: the prospectively registered repair that explicitly refits codes during weight training.
'''
(out/'2026-11-18-tree-embedding-representation.md').write_text(text)
print(out/'representation-capacity.png');print(out/'parent-children-witness.png')
