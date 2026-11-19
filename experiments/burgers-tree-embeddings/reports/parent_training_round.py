"""Generate B7 training diagnostics directly from retained run JSONs."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parents[1]
out=H/'reports/2026-11-19-neural-parent-assets';out.mkdir(exist_ok=True)
names=['b7parentpilot','b7parentpolish','b7parentfull']
r={n:json.loads((H/f'results/{n}/output/result.json').read_text()) for n in names}
fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for ax,name,title in zip(axes,names,['Training-only pilot: Adam','Same pilot: deterministic polish','Full matched-data training']):
    history=r[name]['history'];x=[v['update'] for v in history]
    for key,label in [('pilot_worst','Pilot fields, worst'),('pilot_median','Pilot fields, median'),('subset_worst','Development subset, worst'),('subset_median','Development subset, median')]:
        if key in history[0]:ax.semilogy(x,[100*v[key] for v in history],label=label)
    ax.axhline(.5,color='black',ls=':',label='Representation target (0.5%)');ax.set(title=title,xlabel='Optimizer updates',ylabel='Relative L2 error (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(out/'training-curves.png',dpi=170);plt.close(fig)
f=r['b7parentfull']
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
for key,label in [('training_errors','Training: stored embeddings'),('independent_errors','Development: fitted embeddings')]:
    values=np.sort(f[key]);ax.semilogy(np.linspace(0,100,len(values)),100*values,label=label)
ax.axhline(.5,color='black',ls=':',label='Representation target');ax.axhline(2,color='gray',ls='--',label='Parent prerequisite');ax.set(xlabel='Snapshot percentile',ylabel='Relative L2 error (%)',title='Full parent: all snapshots');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'all-snapshot-errors.png',dpi=170);plt.close(fig)
a=np.load(H/'results/b7parentfull/output/worst_actual4096.npz');ref=a['reference'];pred=a['prediction'];error=pred-ref
fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,image,title in zip(axes,[ref,pred,error],['Reference','Neural parent reconstruction','Signed error']):
    signed=title=='Signed error';v=max(abs(error.min()),abs(error.max())) if signed else ref.max()
    im=ax.imshow(image.T,origin='lower',extent=[0,1,0,1],cmap='coolwarm' if signed else 'viridis',vmin=-v if signed else 0,vmax=v);ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle(f'Actual 4096 × 4096 arrays; worst development snapshot, t={float(a["time"]):g}')
fig.savefig(out/'parent-actual4096.png',dpi=170);plt.close(fig)
report=H/'reports/2026-11-19-neural-parent-training.md'
report.write_text(f'''# Neural parent training and advancement

B7 parent-first round: retained pilot and initial full-data results. These are offline field reconstructions, not PDE trajectories. Further registered diagnosis/repair may supersede the full parent checkpoint.

The training-only pilot improved from {100*r['b7parentpilot']['worst_training_error']:.6f}% worst error to {100*r['b7parentpolish']['worst_training_error']:.6f}% with deterministic polishing. This establishes fit ability on those selected training fields only.

![Training curves](2026-11-19-neural-parent-assets/training-curves.png)

Across all full-data snapshots, worst stored-code training error is {100*f['worst_training_error']:.6f}%; worst independently scored development error is {100*f['worst_independent_error']:.6f}%. There are {f['unfinished_fits']} unfinished development fits. The maximum independent scoring change is {100*f['max_scoring_change']:.8f} percentage points. Parent prerequisite passed: {f['parent_quality_pass']}.

![All snapshot errors](2026-11-19-neural-parent-assets/all-snapshot-errors.png)

The actual4096 witness error is {100*f['actual4096']['relative_error']:.6f}%. The displayed image is a preview of the actual full-resolution arrays.

![Actual4096 witness](2026-11-19-neural-parent-assets/parent-actual4096.png)

No child or PDE advancement follows from these initial full-data results. CROM and SNF-ROM remain unqualified for a fair complete-solve speed comparison. No final-test fields are opened.

Glossary: parent is the global nonlinear coordinate neural decoder; embedding is its adjustable field code; stored-code training error uses codes learned with the network; fitted development error separately optimizes codes against held-out known fields; independent scoring uses256² quadrature points rather than the128² training rule; unfinished means the original convergence criterion was not satisfied. All relative errors use the current-time reference norm.
''')
# Diagnostic comparisons are generated from converged saved fits.
diagnostic=H/'results/b7parentdiagnosis/output/result.json'
if diagnostic.exists():
    d=json.loads(diagnostic.read_text());rows=d['rows']
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    xx=np.arange(len(rows));ax.bar(xx-.2,[100*v['stored_error'] for v in rows],.4,label='Before additional fitting')
    ax.bar(xx+.2,[100*v['newton_status'][0] for v in rows],.4,label='After converged refit')
    ax.set_xticks(xx,[v['group']+' '+str(v['index']) for v in rows],rotation=60,ha='right');ax.set(ylabel='Relative L2 error (%)',title='Fixed weights: embedding optimization diagnosis');ax.legend()
    fig.savefig(out/'fit-diagnosis.png',dpi=170);plt.close(fig)
    report.write_text(report.read_text()+"\nFixed-weight diagnosis converged all selected training/development fits. Training codes improve, while the worst held-out reconstruction remains poor. This implicates both training-code optimization and the learned decoder; it does not prove an architectural capacity limit.\n\n![Fixed-weight fit diagnosis](2026-11-19-neural-parent-assets/fit-diagnosis.png)\n")
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
for key,label in [('training_errors','Training'),('independent_errors','Development')]:
    e=np.array(f[key]).reshape(-1,11)
    ax.plot(np.arange(11)*.025,100*e.max(0),label=label+' worst')
    ax.plot(np.arange(11)*.025,100*np.median(e,axis=0),'--',label=label+' median')
ax.axhline(.5,color='black',ls=':',label='Representation target');ax.set(xlabel='Snapshot time',ylabel='Relative L2 error (%)',title='Offline reconstructions across time — not PDE rollouts');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'errors-across-time.png',dpi=170);plt.close(fig)
report.write_text(report.read_text()+"\n![Offline errors across time](2026-11-19-neural-parent-assets/errors-across-time.png)\n")
body=report.read_text();start=body.index('Glossary:');end=body.index('\n\n',start)
glossary=body[start:end];report.write_text(body[:start]+body[end:]+'\n\n'+glossary+'\n')
publication=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');dest=publication/out.name;dest.mkdir(exist_ok=True)
files=[report,*out.glob('*.png')];records=[]
for p in files:
    target=publication/p.name if p==report else dest/p.name;shutil.copy2(p,target)
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    assert sha(p)==sha(target);records.append(dict(source=str(p),published=str(target),sha256=sha(p)))
(H/'manifests/B7-parent-publication.json').write_text(json.dumps(records,indent=2)+'\n')
print(report)
