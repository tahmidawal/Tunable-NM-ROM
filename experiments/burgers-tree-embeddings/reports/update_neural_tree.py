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

All twelve named development extensions passed spectral space/time ladders and independent finite-difference checks. They are qualified development references; this does not qualify any neural model.

|Named development case|Largest spectral reference allowance|Ladder result|
|---|---:|---|
{rows}

CROM and SNF-ROM reproduction qualification is incomplete. CROM's supplied Burgers implementation is one-dimensional and forced; our two-dimensional viscous study requires a documented problem adapter. Native methods, verified ports and architecture-inspired ablations will remain distinct. No speed advantage over these methods is claimed.

The previous direct-coefficient tree result remains separate and negative on cost. These plots show the newly requested neural architecture, not the previous POD–Fourier or direct-coefficient models. Final tests remain unopened.

## Glossary

**Embedding:** small vector of independently adjustable model variables. **Parent:** global neural field covering the domain. **Child:** local neural correction with its own embedding. **Reconstruction:** fitting a known reference field offline; it is not a PDE prediction. **Relative L2 error:** physical squared-error integral, normalized by the current-time reference norm. **Stationarity:** the fitted objective's gradient is sufficiently small. **Allowance:** measured numerical-reference uncertainty. **Ladder:** progressively finer reference space/time settings. **Development:** exposed trajectories used to diagnose and select methods. **Native method:** author implementation with its intended training and evolution algorithm. **Actual resolution:** the model and reference arrays contain all 4096×4096 requested values; the overview image is reduced for display.
'''
refit=json.loads((H/'results/b6refit/output/result.json').read_text())
feature=json.loads((H/'results/b6features/output/result.json').read_text())
span=json.loads((H/'results/b6span/output/result.json').read_text())
crom=json.loads((H/'results/b6cromdeps/output/result.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
rr=refit['checkpoints']
ax[0].plot([v['update'] for v in rr],[100*v['worst_error'] for v in rr],'o-',label='Joint tree refit')
ax[0].plot([v['update'] for v in rr],[100*v['max_root_error'] for v in rr],'s--',label='Damaged parent refit')
ax[0].axhline(100*r['worst_256_projection'],c='grey',label='Original parent')
ax[0].set(xlabel='Joint weight-training updates',ylabel='Worst subset error (%)',title='Initial nonlinear tree: training damage');ax[0].legend(fontsize=8)
fh=feature['history'];ax[1].plot([v['update'] for v in fh],[100*v['worst_independent_feature_error'] for v in fh],'o-')
ax[1].axhline(.5,c='black',ls='--');ax[1].set(xlabel='Feature training updates',ylabel='Feature-target error (%)',title='Final repair: neural features learned')
ss=span['allocations'];ax[2].bar(range(len(ss)),[100*v['worst_independent_error'] for v in ss],color=['steelblue','seagreen','darkorange'])
ax[2].set_xticks(range(len(ss)),[v['kind']+' depth'+str(v['max_depth']) for v in ss],rotation=15)
ax[2].patches[-1].set_hatch('//');ax[2].text(2,100*ss[-1]['worst_independent_error']+.1,'Unresolved scoring',ha='center',fontsize=8);ax[2].set_ylim(0,10);ax[2].axhline(.5,c='black',ls='--');ax[2].set(ylabel='Worst full-field error (%)',title='But assembled capacity gate fails')
for a in ax:a.grid(axis='y',alpha=.2)
fig.suptitle('Reconstruction diagnostics only — no admitted neural PDE solve')
fig.savefig(ASSETS/'neural-repair-diagnosis.png',dpi=180);plt.close(fig)
spanrows='\n'.join(f"|{v['kind']}|{v['max_depth']}|{100*v['worst_independent_error']:.4f}%|{100*v['max_scoring_change']:.4f} percentage points|" for v in ss)
best=refit['best'];chosen=next(v for v in crom['history'] if v['epoch']==crom['selected_epoch'])
extra=f"""## Completed bounded neural repairs

Curvature-aware parent fitting resolved the unfinished-fit prerequisite without relaxing the stopping criterion. The subsequent jointly trained parent/child network failed. Refitting all retained checkpoints gives best worst subset error **{100*best['worst_error']:.4f}%**; its parent alone is **{100*best['max_root_error']:.4f}%**. Every joint fit remains unfinished. All {best['child_directions_after_parent']['total']} tested child directions are independent after removing the parent span. This is evidence of training/co-adaptation damage and unsuccessful nonlinear fitting, not a proof that local neural children cannot work.

The final registered repair freezes the original parent and learns shared local coordinate-network features. Its independent feature-target fit reaches **{100*feature['best_independent_feature_error']:.4f}%**. However, direct projection into those assembled features with fixed original parent embeddings fails the physical capacity prerequisite on all-trajectory scoring:

|Allocation|Maximum depth|Worst independent256² error|Largest128²-to256² change|
|---|---:|---:|---:|
{spanrows}

These direct-feature coefficients are a capacity diagnostic, not nonlinear child embeddings. Depth4 is additionally unresolved under independent scoring. **Advancement decision: stop this final registered repair; do not train its smaller nonlinear child map or claim PDE/local-adaptation/runtime success.** This scoped failure does not establish a universal decoder limitation or a lower bound with jointly refitted parent embeddings.

![Training damage and final repair capacity failure](2026-11-19-neural-tree-assets/neural-repair-diagnosis.png)

Native CROM training selected epoch {crom['selected_epoch']}, with worst encoded subset error {100*chosen['encoded_subset_worst']:.4f}%. This is an unfinished reproduction diagnostic, not a fair performance ranking: native nonlinear projection, independent scoring and the2D dynamics adapter remain required. The retained recipe uses a lower learning-rate schedule than the author CLI defaults. SNF-ROM dependency installation succeeded but native import exposed a Lux/ChainRules precompile conflict; a single cache-disabled compatibility retry is registered. Neither comparator has an eligible complete solve time.

"""
doc=doc.replace('## Glossary',extra+'## Glossary')
supplement=''
npth=H/'results/b6namedneural/output/result.json'
if npth.exists():
    nr=json.loads(npth.read_text())
    if nr['status']=='complete':
        names=['Smooth','Compact pulse','Diagonal ridge','Two pulses']
        table='\n'.join(f"|{v['case']}|{100*v['worst_parent']:.3f}%|{100*v['worst_tree']:.3f}%|{v['unfinished_tree']}/11|" for v in nr['cases'])
        supplement+='## Tests on the requested Burgers families\n\nThese are independent256² offline reconstructions using retained failed neural checkpoints, not PDE solves. Reference checks passed for all twelve cases. No named case was added to training.\n\n|Case|Worst original-parent error|Worst retained nonlinear-tree error|Unfinished tree fits|\n|---|---:|---:|---:|\n'+table+'\n\n'
        fig,ax=plt.subplots(1,4,figsize=(16,3.8),constrained_layout=True)
        for j,axis in enumerate(ax):
            case=next(v for v in nr['cases'] if v['case']==f'named{j}_nu0.005')
            tt=[v['time'] for v in case['rows']]
            axis.plot(tt,[100*v['errors'][0] for v in case['rows']],label='Original neural parent')
            axis.plot(tt,[100*v['errors'][1] for v in case['rows']],label='Retained nonlinear tree')
            axis.axhline(.5,c='black',ls='--');axis.set(title=names[j],xlabel='Time',ylabel='Relative L2 error (%)',yscale='log');axis.grid(alpha=.2)
        ax[0].legend(fontsize=7);fig.suptitle('Named-case reconstruction errors, viscosity0.005 — not PDE trajectories')
        fig.savefig(ASSETS/'named-neural-errors.png',dpi=180);plt.close(fig)
        supplement+='![Named-family neural errors over time](2026-11-19-neural-tree-assets/named-neural-errors.png)\n\n'
        for j in range(4):
            fig,ax=plt.subplots(3,6,figsize=(15,7),constrained_layout=True)
            for c,it in enumerate(range(0,11,2)):
                v=np.load(npth.parent/f'named{j}_nu0.005_time{it:02d}_independent256.npz')
                xy=v['xy'];truth=v['reference'].reshape(256,256);pred=v['tree'].reshape(256,256);error=pred-truth
                for rr,(field,label) in enumerate([(truth,'Reference'),(pred,'Neural tree'),(error,'Signed error')]):
                    limit=max(float(truth.max()),float(pred.max()));el=max(float(abs(error).max()),1e-12)
                    im=ax[rr,c].pcolormesh(xy[:,0].reshape(256,256),xy[:,1].reshape(256,256),field,cmap='RdBu_r' if rr==2 else 'viridis',vmin=-el if rr==2 else 0,vmax=el if rr==2 else limit,shading='auto',rasterized=True)
                    ax[rr,c].set_aspect('equal');ax[rr,c].set_xticks([]);ax[rr,c].set_yticks([])
                    if rr==0:ax[rr,c].set_title(f't={it*.025:.2f}')
                    if c==0:ax[rr,c].set_ylabel(label)
                    fig.colorbar(im,ax=ax[rr,c],fraction=.046,pad=.02)
            fig.suptitle(names[j]+' — viscosity0.005; independent256² offline reconstruction, not a PDE solve')
            fn=f'named{j}-neural-reconstruction.png';fig.savefig(ASSETS/fn,dpi=150);plt.close(fig)
            supplement+=f'![{names[j]} reference, nonlinear tree and errors](2026-11-19-neural-tree-assets/{fn})\n\n'
cpth=H/'results/b6cromprojection/output/result.json'
if cpth.exists():
    cr=json.loads(cpth.read_text())
    if cr['status']=='complete':
        supplement+=f"## Native CROM projection diagnosis\n\nWorst independent subset error changes from{100*cr['worst_encoded']:.3f}% encoded to{100*cr['worst_projected']:.3f}% after native nonlinear projection. This retained-checkpoint diagnostic does not establish CROM's best attainable accuracy or a complete PDE runtime.\n\n"
spth=H/'results/b6snfnocache/output/result.json'
if spth.exists():
    sr=json.loads(spth.read_text())
    supplement+=f"## Native SNF environment outcome\n\nFinal bounded cache-disabled compatibility fixture passed: **{sr.get('passed',False)}**. No SNF training or complete PDE timing was performed. The pinned source, resolved dependency manifest and failure logs are retained; SNF-ROM is unavailable as an eligible comparator in this batch.\n\n"
doc=doc.replace('## Glossary',supplement+'## Glossary')
name='2026-11-19-neural-tree-development.md' 
(CAN/name).write_text(doc);(H/'reports'/name).write_text(doc)
shutil.copytree(ASSETS,H/'reports'/ASSETS.name,dirs_exist_ok=True)
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'result.json',R/'all_development_fits.pkl',R/'worst_actual4096.npz',*[H/('results/'+v+'/output/result.json') for v in ['b6refit','b6features','b6span','b6namedfd','b6cromdeps']],*[p for p in [npth,cpth,spth] if p.exists()],CAN/name,*ASSETS.glob('*.png')]}
(H/'manifests/B6-neural-publication.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(CAN/name)
