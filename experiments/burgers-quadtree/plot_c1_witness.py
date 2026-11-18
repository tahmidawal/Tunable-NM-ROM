"""Independent NumPy audit and physical-output plot, using all4096 values for metrics."""
from pathlib import Path
import hashlib,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
H=Path(__file__).resolve().parent;p=H/'results/c1cost/output/final4096_case9.npz';manifest=dict(line.split('  ',1)[::-1] for line in (p.parents[1]/'OUTPUTS.sha256').read_text().splitlines());assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['output/'+p.name]
d=np.load(p);ref=d['reference'];rom=d['local_hybrid'];classical=d['spectral'];assert ref.shape==rom.shape==classical.shape==(4096,4096)
ph=np.load(H/'results/c1data/output/case137.npz')['parameters'];x=np.arange(4096)/4095;gx=np.exp(-(x-ph[0])**2/(2*ph[2]**2));gy=np.exp(-(x-ph[1])**2/(2*ph[2]**2));gx[[0,-1]]=0;gy[[0,-1]]=0;norm=ph[3]*np.linalg.norm(gx)*np.linalg.norm(gy)
errors={name:float(np.linalg.norm(field-ref)/norm) for name,field in [('local_hybrid',rom),('spectral64_dt005',classical)]};row=next(x for x in json.loads((H/'evidence/c1cost-result.json').read_text())['rows'] if x['case']==9)
for name,value in errors.items():assert np.isclose(value,row['errors'][name][-1],rtol=1e-10,atol=1e-12)
wrong=float(np.linalg.norm(rom-ref)/np.linalg.norm(ref));assert not np.isclose(wrong,errors['local_hybrid'],rtol=1e-3)
(H/'evidence/c1cost-witness-audit.json').write_text(json.dumps(dict(case=9,shape=list(ref.shape),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),numpy_errors=errors,reported_errors={k:row['errors'][k][-1] for k in errors},wrong_final_field_normalization=wrong,normalization_negative_control=True),indent=2)+'\n')
fig,ax=plt.subplots(1,3,figsize=(12,3.6),layout='constrained');im=ax[0].imshow(ref[::8,::8].T,origin='lower',extent=(0,1,0,1),cmap='viridis');fig.colorbar(im,ax=ax[0],shrink=.8);ax[0].set_title('Converged reference')
limit=float(np.max(np.abs(rom-ref)))
for a,field,label in zip(ax[1:],[rom,classical],['Local hybrid error','Spectral control error']):
 im=a.imshow((field-ref)[::8,::8].T,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-limit,vmax=limit);fig.colorbar(im,ax=a,shrink=.8);a.set_title(label)
tile=[min(2,int(np.argmax(g)*4/4096)) for g in (gx,gy)]
for i in (tile[0],tile[0]+1):
 for j in (tile[1],tile[1]+1):ax[1].add_patch(Rectangle((i/4,j/4),.25,.25,fill=False,color='#303030',lw=.9))
for a in ax:a.set(xlabel='x',ylabel='y',xlim=(0,1),ylim=(0,1))
fig.suptitle('Actual 4096 × 4096 output · development case 9 · t = 0.25')
fig.supxlabel('Display sampled for plotting; audited errors use every output value. Boxes show selected patch cores.',fontsize=9)
(H/'figures').mkdir(exist_ok=True);fig.savefig(H/'figures/c1-dev9-output.png',dpi=160);plt.close(fig)
print(errors)
