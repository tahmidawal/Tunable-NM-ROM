"""Publish measured B6 neural diagnostics; never relabel reconstructions as solves."""
from pathlib import Path
import json,pickle,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parents[1]
CAN=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports')
ASSETS=CAN/'2026-11-19-neural-tree-assets';ASSETS.mkdir(exist_ok=True)
R=H/'results/b6parent/output';r=json.loads((R/'result.json').read_text())
with open(R/'all_development_fits.pkl','rb') as f:fits=pickle.load(f)
errors=np.array(r['independent_256_errors']).reshape(44,11)*100
history=r['history'];updates=[v['update'] for v in history]
fig,ax=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
ax[0].plot(updates,[100*v['subset_worst'] for v in history],label='Worst selection snapshot')
ax[0].plot(updates,[100*v['subset_median'] for v in history],label='Median selection snapshot')
ax[0].axhline(.5,c='black',ls='--',label='Representation target')
ax[0].axvline(r['selected_update'],c='grey',ls=':');ax[0].set(xlabel='Training updates',ylabel='Relative L2 error (%)',yscale='log',title='Actual coordinate neural parent — 8 embeddings')
ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
im=ax[1].imshow(errors,aspect='auto',origin='lower',extent=[-.0125,.2625,-.5,43.5],cmap='magma',vmin=0)
ax[1].set(xlabel='Time',ylabel='Development trajectory (32 random + 12 alignment)',title='All 484 reconstruction errors, independent 256² scoring')
fig.colorbar(im,ax=ax[1],label='Relative L2 error (%)')
fig.savefig(ASSETS/'parent-training-and-errors.png',dpi=180);plt.close(fig)
v=np.load(R/'worst_actual4096.npz');truth=v['reference'];prediction=v['prediction'];error=prediction-truth
fig,ax=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
limit=max(float(truth.max()),float(prediction.max()));elo=float(abs(error).max())
for axis,field,title in zip(ax,[truth,prediction,error],['Qualified numerical reference','Neural parent reconstruction','Signed error: model − reference']):
    signed=title.startswith('Signed');im=axis.imshow(field.T,origin='lower',extent=[0,1,0,1],
        cmap='RdBu_r' if signed else 'viridis',vmin=-elo if signed else 0,vmax=elo if signed else limit)
    axis.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=axis,shrink=.8)
fig.suptitle(f"{r['actual4096']['case']}, t={r['actual4096']['time']:g}: actual 4096×4096 arrays; parent error {100*r['actual4096']['relative_error']:.3f}%\nOffline reconstruction — not an embedding-space PDE trajectory",fontsize=11)
fig.savefig(ASSETS/'parent-actual4096-witness.png',dpi=180);plt.close(fig)
named=json.loads((H/'results/b6namedrefs/output/result.json').read_text())
rows='\n'.join(f"|{v['case']}|{100*max(v['allowance']):.7f}%|{'Pass' if v['provisional_ladder_pass'] else 'Fail'}|" for v in named['cases'])
doc=f'''# Actual nonlinear neural tree: development evidence

These are provisional reconstruction and reference diagnostics for the new coordinate neural model. No trained neural-tree PDE trajectory or CROM/SNF-ROM runtime comparison is admitted yet.

The first global neural parent uses eight nonlinear embeddings and three sine-network hidden layers. Its selected checkpoint, at update {r['selected_update']}, has worst independent development reconstruction error **{100*r['worst_256_projection']:.4f}%** over all 44 trajectories and 11 times. The worst actual 4096×4096 reconstruction gives **{100*r['actual4096']['relative_error']:.4f}%**, so the error is not a coarse scoring-grid artifact. The largest change between 128² and independent 256² scoring is {100*r['max_128_to_256_error_change']:.7f} percentage points.

The parent fails the 0.5% representation gate. {r['unfinished_full_fits']} of 484 embedding fits remain unfinished under the registered stopping criterion; these are explicitly retained. Most fits reach stationarity, which supports investigating decoder/training capacity and local corrections rather than assuming every large error is merely an unfinished optimizer. This does not establish an inherent neural-decoder limitation.

![Parent training and all-snapshot reconstruction errors](2026-11-19-neural-tree-assets/parent-training-and-errors.png)

![Actual-resolution parent reconstruction and error](2026-11-19-neural-tree-assets/parent-actual4096-witness.png)

The initial assembled-child recipe checks parent-fit stability first, initializes children from missing reference fields, and jointly trains embeddings and shared networks across tree depths. Uniform and prescribed allocations use the same checkpoint. Full independent scoring and useful independent child-direction checks are required before embedding-space PDE trajectories. The implemented nonlinear weak solver passes dense arithmetic and derivative fixtures; this is not an accuracy qualification of a trained model.

All twelve named development extensions have completed spectral space/time ladders. Independent finite-difference checks are still pending, so these references are not yet admitted for method claims.

|Named development case|Largest spectral reference allowance|Ladder result|
|---|---:|---|
{rows}

CROM and SNF-ROM source/environment qualification is running separately. CROM's supplied Burgers implementation is one-dimensional and forced; our two-dimensional viscous study requires a documented problem adapter. Native methods, verified ports and architecture-inspired ablations will remain distinct. No speed advantage over these methods is claimed.

The previous direct-coefficient tree result remains separate and negative on cost. These plots show the newly requested neural architecture, not the previous POD–Fourier or direct-coefficient models. Final tests remain unopened.

## Glossary

**Embedding:** small vector of independently adjustable model variables. **Parent:** global neural field covering the domain. **Child:** local neural correction with its own embedding. **Reconstruction:** fitting a known reference field offline; it is not a PDE prediction. **Relative L2 error:** physical squared-error integral, normalized by the current-time reference norm. **Stationarity:** the fitted objective's gradient is sufficiently small. **Allowance:** measured numerical-reference uncertainty. **Ladder:** progressively finer reference space/time settings. **Development:** exposed trajectories used to diagnose and select methods. **Native method:** author implementation with its intended training and evolution algorithm. **Actual resolution:** the model and reference arrays contain all 4096×4096 requested values; the overview image is reduced for display.
'''
name='2026-11-19-neural-tree-development.md'
(CAN/name).write_text(doc);(H/'reports'/name).write_text(doc)
shutil.copytree(ASSETS,H/'reports'/ASSETS.name,dirs_exist_ok=True)
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'result.json',R/'all_development_fits.pkl',R/'worst_actual4096.npz',CAN/name,*ASSETS.glob('*.png')]}
(H/'manifests/B6-neural-publication.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(CAN/name)
