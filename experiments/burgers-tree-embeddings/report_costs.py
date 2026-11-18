"""Measured component-cost plots, explicitly separate from complete PDE runtime."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;out=H/'reports';out.mkdir(exist_ok=True)
a=json.loads((H/'results/audit01/output/result.json').read_text())
g=json.loads((H/'results/global01/output/result.json').read_text())
m=a['component_timing']['methods']
families=[]
families.append(('Global nonlinear','#30343b',[(f'global_head{k}',g['heads'][str(k)]['worst_projection']) for k in (16,32,64)]))
families.append(('Global direct','#355d8c',[(f'global_direct{k}',g['global_direct'][str(k)]['worst']) for k in (16,32,64,128)]))
families.append(('Nonlinear tree / prescribed','#5a7e55',[(f'tree{k}',next(r for r in a['comparisons'] if r['cap']==k and r['allocation']=='prescribed')['projection']['worst']) for k in (32,64,128)]))
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
for ax,key,label in zip(axes,['field_seconds','jvp_seconds'],['Field evaluation','Code-Jacobian vector product']):
    for family,color,items in families:
        xs=[np.median(m[n][key])*1e6 for n,_ in items];ys=[100*e for _,e in items]
        ax.plot(xs,ys,'o-',lw=1.5,color=color,label=family)
        for xx,yy,(n,_) in zip(xs,ys,items):ax.annotate(str(m[n]['active_variables']),(xx,yy),xytext=(3,4),textcoords='offset points',fontsize=8)
    ax.axhline(.5,ls=':',color='#a53c3c',lw=1)
    ax.set(xlabel=f'{label} (microseconds)',ylabel='Worst projection error (%)',title=label)
    ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('Initial checkpoint: cached 4096-point component costs\nSame GPU/allocation, five repetitions; labels are active variables; not PDE solve time',fontsize=11)
fig.savefig(out/'representation-component-cost.png',dpi=160);fig.savefig(out/'representation-component-cost.svg');plt.close(fig)
v=np.load(H/'results/audit01/output/actual4096.npz');ref=v['reference'];pred=v['prediction']
fig,axes=plt.subplots(1,4,figsize=(14,3.6),layout='constrained')
for ax,z,title,signed in zip(axes,[ref,pred,pred-ref,np.abs(pred-ref)],['Reference','Actual4096 reconstruction','Signed error','Absolute error'],[False,False,True,False]):
    d=z[::4,::4];lim=np.max(np.abs(z));vmax=lim if signed or title=='Absolute error' else ref.max()
    im=ax.imshow(d.T,extent=[0,1,0,1],origin='lower',cmap='RdBu_r' if signed else 'viridis',vmin=-lim if signed else 0,vmax=vmax)
    ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,shrink=.8,label='u')
fig.suptitle(f'{str(v["case"])} at t={float(v["time"]):.3f}; full4096 arrays retained, display sampled every4th node\nRelative L2={100*a["dense_witness"]["grids"]["4096"]["relative_l2"]:.4f}%; reference-assisted projection, no PDE solve',fontsize=10)
fig.savefig(out/'actual4096-witness.png',dpi=160);plt.close(fig)
print(out/'representation-component-cost.png');print(out/'actual4096-witness.png')
