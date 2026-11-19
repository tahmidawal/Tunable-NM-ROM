"""Publish completed B7 parent and named-family evidence without PDE claims."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parents[1]
load=lambda name:json.loads((H/f'results/{name}/output/result.json').read_text())
initial=load('b7parentfull');final=load('b7parentrepair');named=load('b7namedparent');snf=load('b7snff64')
assert final['status']=='complete_stable_parent' and named['status']=='complete'
out=H/'reports/2026-11-19-neural-parent-assets';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for ax,r,title in zip(axes,[initial,final],['Initial full-data training','Final alternating-training repair']):
    history=r['history'];x=[v['update'] for v in history]
    for key,label in [('subset_worst','Development subset: worst'),('subset_median','Development subset: median'),('pilot_worst','Training pilot fields: stored codes, worst')]:
        ax.semilogy(x,[100*v[key] for v in history],label=label)
    ax.axhline(2,color='gray',ls='--',label='Parent prerequisite');ax.axhline(.5,color='black',ls=':',label='Assembled representation target')
    ax.set(title=title,xlabel='Training updates',ylabel='Relative L2 error (%)');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.savefig(out/'final-training-curves.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for r,label in [(initial,'Initial'),(final,'Final repair')]:
    e=np.asarray(r['independent_errors']).reshape(44,11)
    axes[0].semilogy(np.linspace(0,100,e.size),100*np.sort(e.ravel()),label=label)
    axes[1].plot(np.arange(11)*.025,100*e.max(0),label=label+' worst')
    axes[1].plot(np.arange(11)*.025,100*np.median(e,axis=0),'--',label=label+' median')
for ax in axes:ax.axhline(.5,color='black',ls=':');ax.set(ylabel='Relative L2 error (%)');ax.legend();ax.grid(alpha=.2)
axes[0].set(xlabel='Development snapshot percentile',title='All 484 independent reconstructions')
axes[1].set(xlabel='Snapshot time',title='Offline reconstruction errors across time')
fig.savefig(out/'final-errors.png',dpi=170);plt.close(fig)
plots=[];full_exports=[]
export_dir=H/'results/b7-parent-exports';export_dir.mkdir(exist_ok=True)
for ci,case in enumerate(named['cases']):
    if abs(case['nu']-.005)>1e-12:continue
    frames=[];rows=[r for r in case['rows'] if 'actual4096' in r]
    for row in rows:
        a=np.load(H/'results/b7namedparent/output'/row['actual4096']['file'])
        assert a['reference'].shape==(4096,4096) and a['prediction'].shape==(4096,4096)
        frames.append((a['reference'],a['prediction']));a.close()
    vmin=min(min(v[0].min(),v[1].min()) for v in frames);vmax=max(max(v[0].max(),v[1].max()) for v in frames);emax=max(np.abs(v[1]-v[0]).max() for v in frames)
    fig,axes=plt.subplots(3,len(frames),figsize=(17,8),constrained_layout=True)
    for k,((ref,pred),row) in enumerate(zip(frames,rows)):
        for j,field in enumerate([ref,pred,np.abs(pred-ref)]):
            kind=['reference','reconstruction','error'][j]
            export=export_dir/f'{case["case"]}_t{row["time"]:.2f}_{kind}.png'
            plt.imsave(export,field.T,origin='lower',vmin=0 if j==2 else vmin,vmax=emax if j==2 else vmax,cmap='magma' if j==2 else 'viridis')
            full_exports.append(dict(source=str(export),case=case['case'],time=row['time'],kind=kind,shape=[4096,4096],color_min=float(0 if j==2 else vmin),color_max=float(emax if j==2 else vmax)))
            im=axes[j,k].imshow(field[::8,::8].T,origin='lower',extent=[0,1,0,1],vmin=0 if j==2 else vmin,vmax=emax if j==2 else vmax,cmap='magma' if j==2 else 'viridis')
            axes[j,k].set_xticks([0,.5,1]);axes[j,k].set_yticks([0,.5,1])
            if j==0:axes[j,k].set_title(f't={row["time"]:.2f}')
            if j==1:axes[j,k].set_xlabel(f'L2 error {100*row["actual4096"]["relative_error"]:.3f}%')
            if k==0:axes[j,k].set_ylabel(['Reference','Neural reconstruction','Absolute error'][j])
            if k==len(frames)-1:fig.colorbar(im,ax=axes[j,:],shrink=.8)
    fig.suptitle(case['definition']['name']+' — actual 4096 × 4096 arrays, previewed at 512 × 512; offline fits, not PDE rollouts')
    p=out/f'named-{ci:02d}-actual4096-times.png';fig.savefig(p,dpi=150);plt.close(fig);plots.append((case,p));print('full_resolution_exports',case['case'],flush=True)
report=H/'reports/2026-11-19-neural-parent-training.md'
lines=['# Neural parent training, full-resolution reconstructions and advancement','',
       'Completed B7 parent-training evidence and named-family reconstruction diagnostics. These fields are independently fitted offline reconstructions; they are not neural PDE rollouts. Final-test solution fields remain sealed.','',
       '| Measurement | Initial full parent | Final training repair |','|---|---:|---:|']
for label,key in [('Worst development error, independent256²','worst_independent_error'),('Worst stored-code training error','worst_training_error'),('Maximum scoring change','max_scoring_change')]:
    lines.append(f'|{label}|{100*initial[key]:.6f}%|{100*final[key]:.6f}%|')
lines.extend([f'|Actual 4096 × 4096 worst witness|{100*initial["actual4096"]["relative_error"]:.6f}%|{100*final["actual4096"]["relative_error"]:.6f}%|',
 f'|Unfinished development fits|{initial["unfinished_fits"]}|{final["unfinished_fits"]}|','',
 f'The final actual 4096 witness is {final["actual4096"]["case"]} at time {final["actual4096"]["time"]:g}.','',
 '![Training curves](2026-11-19-neural-parent-assets/final-training-curves.png)','',
 '![All-snapshot and across-time errors](2026-11-19-neural-parent-assets/final-errors.png)','',
 f'Parent prerequisite passed: **{final["parent_quality_pass"]}**. Parent-alone 0.5% representation gate passed: **{final["parent_representation_pass"]}**. The original assembled 0.5% representation and 1% PDE thresholds are unchanged.'])
if not final['parent_quality_pass']:
    lines.extend(['','The two permitted parent/training repairs are exhausted. Stop parent development in this round; do not train children on this unqualified parent or claim an embedding-space PDE result. This is a scoped failure of the tested parent/training recipes, not proof that nonlinear local children cannot work.'])
lines.extend(['','The fixed-weight diagnosis showed that training-code optimization was incomplete, but finishing development embeddings did not remove the large error. Additional network training tests that remaining learned-decoder bottleneck. A small training-only pilot passing 0.5% was not evidence of held-out-family accuracy.','',
 '![Fixed-weight diagnosis](2026-11-19-neural-parent-assets/fit-diagnosis.png)','',
 '| Named development family | Viscosity | Worst reconstruction error | Unfinished fits |','|---|---:|---:|---:|'])
for c in named['cases']:lines.append(f'|{c["definition"]["name"]}|{c["nu"]:g}|{100*c["worst_error"]:.6f}%|{c["unfinished"]}/11|')
lines.extend(['','Named-family solutions were not added to training. Ridge and two-pulse cases test transfer beyond the original pulse/smooth training distribution. Viscosities 0.01 and 0.02 are also outside the training viscosity of 0.005. All 24 primary-viscosity previews below come from actual 4096-by-4096 reference and reconstruction arrays; the saved errors use the full arrays and current-time reference norm.'])
for c,p in plots:
    lines.extend(['',f'![{c["definition"]["name"]}]({out.name}/{p.name})','',
                  '| Time | Full-resolution reference | Full-resolution neural reconstruction | Full-resolution error |','|---:|---|---|---|'])
    times=sorted({r['time'] for r in full_exports if r['case']==c['case']})
    for t in times:
        selected=[r for r in full_exports if r['case']==c['case'] and r['time']==t]
        links=['['+r['kind']+']('+out.name+'/full-resolution/'+Path(r['source']).name+')' for r in selected]
        lines.append('|'+f'{t:.2f}'+'|'+'|'.join(links)+'|')

starts_path=H/'results/b7parentstarts/output/result.json'
if starts_path.exists():
    starts=json.loads(starts_path.read_text());assert starts['status']=='complete'
    lines.extend(['','Closing fixed-weight starting-point diagnosis (six selected worst snapshots only; no revised all-case qualification):','',
                  '| Snapshot | Registered-fit error | Best diagnostic-start error | Converged |','|---|---:|---:|---|'])
    for row in starts['rows']:
        converged=bool(row['statuses'][row['best_start']][-1])
        lines.append(f'|{row["index"]}|{100*row["original_error"]:.6f}%|{100*row["best_independent_error"]:.6f}%|{converged}|')
    ranks=[row['tangent_conditioning']['rank_at_relative_1e6'] for row in starts['rows']]
    ratios=[row['tangent_conditioning']['minimum_relative'] for row in starts['rows']]
    lines.extend(['',f'The trained parent retains {min(ranks)}–{max(ranks)} independent code directions out of 32 at these fits (relative singular-value cutoff $10^{{-6}}$). The minimum-to-maximum column-normalized singular-value ratio ranges from {min(ratios):.6f} to {max(ratios):.6f}. These checks do not indicate code-direction collapse at the tested fits.'])
    lines.extend(['','Starts include the retained solution, a training-PCA-projected code and field-nearest codes from 12 distinct training trajectories, selected by full physical field distance. Persisting error is evidence against these straightforward starting-point fixes, not a proof of global optimality or an intrinsic architecture limit.'])
lines.extend(['',f'Native SNF patched-environment qualification passed: **{snf["passed"]}**. The fixture verifies float64 GPU weights and gradients and a scalar-equivalent native Burgers RHS. It is not a trained SNF model. Earlier missing-adapter and precision-port failures are retained. CROM remains an unqualified reproduction with recorded training limitations. No fair complete-PDE-solve timing comparison against CROM or SNF-ROM is available.','',
 'Glossary: parent means the global nonlinear coordinate neural decoder; embedding means its independently fitted field code; reconstruction means fitting a known field offline; PDE rollout means advancing codes using the PDE without future solution fields; unfinished means the original stopping test failed; independent scoring uses 256² quadrature instead of 128² training quadrature; actual 4096 means all 4096² field values were produced and scored. The parent prerequisite allows residual-child development but does not replace the stricter assembled representation or PDE requirements.'])
report.write_text('\n'.join(lines)+'\n')
pub=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');dest=pub/out.name;dest.mkdir(exist_ok=True);records=[]
for p in [report,*out.glob('*.png')]:
    target=pub/p.name if p==report else dest/p.name;shutil.copy2(p,target)
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    assert sha(p)==sha(target);records.append(dict(source=str(p),published=str(target),sha256=sha(p)))
full_dest=dest/'full-resolution';full_dest.mkdir(exist_ok=True)
for row in full_exports:
    p=Path(row['source']);target=full_dest/p.name;shutil.copy2(p,target)
    row.update(published=str(target),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    assert row['sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
(H/'manifests/B7-parent-full-resolution-exports.json').write_text(json.dumps(full_exports,indent=2)+'\n')
(H/'manifests/B7-parent-publication.json').write_text(json.dumps(records,indent=2)+'\n');print(report)
