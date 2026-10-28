"""Display retained FE fields; display bins are not numerical error metrics."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads((a.case/'EXPORT-MANIFEST.json').read_text())
for name in ['full_rule.npz','result.json']:
    assert sha(a.case/name)==m['files'][name],name
r=json.loads((a.case/'result.json').read_text());c=r['configuration']
with np.load(a.case/'full_rule.npz') as z:
    x=z['points'];w=z['weights'];u=z['velocity'];pressure=z['pressure']
assert all(np.isfinite(v).all() for v in [x,w,u,pressure]) and np.all(w>0)
nx,ny=660,123
ix=np.floor((x[:,0]-c['xmin'])/(c['xmax']-c['xmin'])*nx).astype(int)
iy=np.floor(x[:,1]/c['H']*ny).astype(int)
assert np.all((ix>=0)&(ix<nx)&(iy>=0)&(iy<ny))
bins=iy*nx+ix;weight=np.bincount(bins,weights=w,minlength=nx*ny)
values=[np.linalg.norm(u,axis=1),pressure]
fig,axes=plt.subplots(2,1,figsize=(12,5.8),layout='constrained')
for ax,v,label,cmap in zip(axes,values,['Velocity magnitude [m/s]','Kinematic pressure [m²/s²]'],['viridis','cividis']):
    numerator=np.bincount(bins,weights=w*v,minlength=nx*ny)
    mean=np.divide(numerator,weight,out=np.full(nx*ny,np.nan),where=weight>0).reshape(ny,nx)
    im=ax.imshow(mean,origin='lower',extent=[c['xmin'],c['xmax'],0,c['H']],cmap=cmap,aspect='equal',interpolation='nearest')
    ax.add_patch(Circle(c['center'],c['radius'],facecolor='white',edgecolor='black',linewidth=.6))
    ax.set(xlabel='x [m]',ylabel='y [m]',title=label)
    fig.colorbar(im,ax=ax,shrink=.8,pad=.015)
re=c['U']*c['D']/c['nu']
fig.suptitle(f'Original fixed channel · Re = {re:g} · retained P{c["velocity_order"]}/P{c["pressure_order"]} FE solution\nProvisional reference candidate — ROM training and full population qualification pending',fontsize=12)
fig.savefig(a.output/'fixed-channel-fields.png',dpi=180)
fig.savefig(a.output/'fixed-channel-fields.pdf')
metadata={'status':'provisional_FE_display_not_ROM_result_or_reference_admission','input_directory':str(a.case.resolve()),'input_hashes':{name:sha(a.case/name) for name in ['EXPORT-MANIFEST.json','full_rule.npz','result.json']},'all_volume_points_used':len(x),'display_bins':[nx,ny],'display_operation':'positive-quadrature-weighted pixel average; empty pixels masked; no interpolation or new PDE evaluation','physical_scope':'original fixed finite DFG channel; no domain independence claim','Re':re,'raw_ranges':{'speed':[float(values[0].min()),float(values[0].max())],'pressure':[float(pressure.min()),float(pressure.max())]},'script_sha256':sha(Path(__file__))}
(a.output/'PROVENANCE.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
