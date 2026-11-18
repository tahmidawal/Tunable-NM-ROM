"""Generate the continuing study report/figures exclusively from retained run arrays."""
from pathlib import Path
import hashlib,json,shutil,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;H=OUT.parent
sources={}
def read(name):
    p=H/'results'/name/'output/result.json';sources[str(p.relative_to(H))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def figure(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def pct(x):return f'{100*x:.4f}%'
def panel(err,labels,prefix):
    ix=[i for i,n in enumerate(labels) if (n.startswith('align') if prefix=='alignment' else n.startswith('dev') and ((int(n[3:])<16)==(prefix=='original')))]
    return float(np.max(np.asarray(err)[ix]))

old=read('auditr1');g0=read('global01');attr=read('b1attrib01');p16=read('b1parent16');resolved=read('b1resolved2')
alias=read('b1spanaudit');span=read('b1expanded');cache=read('b1cache')
labels=[r['case'] for r in cache['cases']['dev']]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

fig,ax=plt.subplots(figsize=(8.3,4.8))
xx=[int(k) for k in span['controls']];yy=[100*v['worst'] for v in span['controls'].values()]
ax.loglog(xx,yy,'o-',color='#546d9c',label='Global POD: expanded training data')
xx=[v['active_variables'] for v in cache['direct_controls'].values()];yy=[100*v['projection']['worst'] for v in cache['direct_controls'].values()]
ax.loglog(xx,yy,'o-',color='#00877c',label='Shared local POD + global parent')
for k,v in cache['direct_controls'].items():ax.annotate(f'{k}/node',(v['active_variables'],100*v['projection']['worst']),xytext=(4,7),textcoords='offset points',fontsize=8)
trained=[]
for name in ['b1global64','b1global128','b1child8','b1child16']:
    if not (H/'results'/name/'output/result.json').exists():continue
    r=read(name)
    if not r['status'].startswith('complete_'):continue
    trained.append((name,r))
    if 'global' in name:ax.scatter([r['latent']],[100*r['worst_projection']],marker='s',s=50,color='#8f54a0',label=f'Global nonlinear {r["latent"]}')
    else:ax.scatter([r['active_variables']],[100*r['worst_projection']],marker='D',s=50,label=f'Nonlinear child {r["latent"]}')
ax.axhline(.5,color='#b5413c',ls='--',lw=1,label='Representation gate (every snapshot)')
ax.set(xlabel='Total adjustable variables, including retained parents',ylabel='Worst current-time relative L2 error (%)',
       title='Expanded panel: shared local corrections versus global variables')
ax.grid(alpha=.17,which='both');ax.legend(fontsize=8,loc='lower left');figure(fig,'b1-representation-capacity')

fig,axs=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
for row,name,title in [(0,'auditr1','Original learned local model'),(1,'b1directaudit','Shared local POD24')]:
    f=H/'results'/name/'output/actual4096.npz';sources[str(f.relative_to(H))]=hashlib.sha256(f.read_bytes()).hexdigest()
    d=np.load(f);ref=d['reference'];pred=d['prediction'];error=pred-ref
    e=float(np.linalg.norm(error)/np.linalg.norm(ref));peak=max(abs(error.min()),abs(error.max()))
    values=[ref,pred,error];titles=['Reference','Reconstruction','Signed error']
    for col,(v,tt) in enumerate(zip(values,titles)):
        kw=dict(cmap='magma',vmin=0,vmax=ref.max()) if col<2 else dict(cmap='RdBu_r',vmin=-peak,vmax=peak)
        im=axs[row,col].imshow(v[::8,::8].T,origin='lower',extent=(0,1,0,1),**kw)
        axs[row,col].set_title(tt);axs[row,col].set(xlabel='x',ylabel='y');fig.colorbar(im,ax=axs[row,col],shrink=.7)
    axs[row,0].set_ylabel(f'{title}\n{str(d["case"])}, t={float(d["time"]):g}\ny')
    axs[row,2].set_title(f'Signed error · L2 {pct(e)}')
fig.suptitle('Actual 4096×4096 fields; each method’s worst witness (different cases)',fontsize=12)
figure(fig,'b1-actual-field-witnesses')

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
selected=next(c for c in p16['checkpoints'] if c['update']==p16['selected_update'])
train=[g0['heads']['16']['train_projection']['worst'],selected['train_projection']['worst']]
dev=[g0['heads']['16']['worst_projection'],p16['worst_projection']]
x=np.arange(2);axs[0].bar(x-.17,np.array(train)*100,width=.34,label='Training');axs[0].bar(x+.17,np.array(dev)*100,width=.34,label='Development + alignment')
axs[0].set(xticks=x,xticklabels=['Original parent','Normalized richer parent'],ylabel='Worst projection error (%)',title='Better training leaves a generalization gap')
axs[0].legend(fontsize=8);axs[0].text(.02,-.26,'Both have 16 global variables; feature banks also differ.',transform=axs[0].transAxes,fontsize=8)
grids=[128,256,512,2048,4096];values=[alias['scoring'][str(n)]['worst'] for n in grids[:3]]+[alias['dense_witness']['grids'][str(n)]['relative_l2'] for n in grids[3:]]
axs[1].loglog(grids,np.array(values)*100,'x-',color='#b5413c',label='Aliased fit (retracted)')
axs[1].loglog([2048,4096],[100*resolved['dense_witness']['errors'][str(n)] for n in [2048,4096]],'o-',color='#00877c',label='Resolved-metric refit')
axs[1].axhline(.5,color='gray',ls='--',lw=1);axs[1].set(xlabel='Independent scoring points per axis',ylabel='Current-time relative L2 error (%)',title='Dense scoring exposed a numerical failure')
axs[1].legend(fontsize=8);figure(fig,'b1-training-and-integration')

name='2026-11-18-tree-embedding-representation.md';old_report=OUT/name;archive=OUT/'B0-retained-representation.md'
if not archive.exists():shutil.copy2(old_report,archive)
bestold=min(old['direct_fine_projection'],key=lambda x:x['projection']['worst'])
lines=[]
for k,v in cache['direct_controls'].items():
    e=v['projection']['per_snapshot'];lines.append(f"| Direct local {k} | {v['active_variables']} | {pct(panel(e,labels,'original'))} | {pct(panel(e,labels,'new'))} | {pct(panel(e,labels,'alignment'))} | {pct(v['projection']['worst'])} |")
for n,r in trained:
    chosen=next(c for c in r['checkpoints'] if c['update']==r['selected_update']);e=chosen['projection']['per_snapshot']
    lines.append(f"| {'Global' if 'global' in n else 'Child'} nonlinear {r['latent']} | {r.get('active_variables',r['latent'])} | {pct(panel(e,r['dev_labels'],'original'))} | {pct(panel(e,r['dev_labels'],'new'))} | {pct(panel(e,r['dev_labels'],'alignment'))} | {pct(r['worst_projection'])} |")
ledger=json.loads((H/'cluster/ledger.json').read_text());terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
directaudit=read('b1directaudit')
allocation_rows='\n'.join(f"| {k} | {v['active_variables']} | {pct(v['projection']['worst'])} |" for k,v in directaudit['controls'].items())
charged=sum(r.get('elapsed_seconds',0) for r in ledger if r.get('state') in terminal)
live=[r['name'] for r in ledger if r.get('state') not in terminal]
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
text=f'''# Burgers tree embeddings: representation diagnosis and continuation

This generated report covers the original bounded failure and the user-authorized continuation. Representation results are development evidence; nonlinear compression, dense audits and fixed-tree physics are still being investigated. No complete PDE-runtime win or final-test result is established.

## Why the original best error was still high

The original best tree had {pct(bestold['projection']['worst'])} worst projection error at 128 adjustable variables. Its actual-grid witness remains {pct(old['dense_witness']['grids']['4096']['relative_l2'])}; that failure is real, not the subsequently retracted quadrature result.

The evidence supports several contributing limitations. The original parent alone has {pct(g0['heads']['16']['worst_projection'])} worst development error. Better-scaled training with a richer frozen feature span reduces selected training error from {pct(train[0])} to {pct(train[1])}, but development error remains {pct(dev[1])}. Thus optimization and generalization are both relevant; changing scaling and the feature bank together does not isolate one cause. The old head’s relative departure from its best affine map is {pct(attr['old_parent_nonlinearity']['relative_departure_from_best_affine'])}, so describing it as exactly linear would be wrong.

Attribution on the old fixed trees provisionally improves when parent and local coefficient restrictions are relaxed. Those sampled-integration diagnostics also increase variable counts and are not an additive causal decomposition. Stronger evidence is the resolved richer local span: its worst original-panel error is {pct(resolved['root_controls']['128']['projection']['worst'])} with {resolved['root_controls']['128']['variables']} coefficients, confirmed by actual output at {pct(resolved['dense_witness']['errors']['4096'])}. This rules out a universal error floor at the original value for the broader local representation. It does not yet establish efficient nonlinear compression.

![Training and numerical diagnosis](b1-training-and-integration.png)

## Expanded matched-data comparison

Training uses the registered prefix expansion to 128 trajectories. Development has 32 random trajectories and the same 12 alignment cases; every one of the 11 snapshot times is scored in the current-time physical norm. Final-test solution fields remain unopened. Shared local POD uses training coefficients only, fixed windows and retained parents. Its allocations use the known initial field and a sampled speed estimate, not evolved reference fields or a certified propagation bound.

| Representation | Total variables | Original random panel | Added random panel | Alignment panel | Worst all |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

The direct local entries use resolved physical Gram integration; actual-grid audits are separate. The strongest listed expanded global POD control has {len(np.load(H/'results/b1expanded/output/pod_basis.npz')['coefficients'])} coefficients and {pct(span['controls']['768']['worst'])} worst error. Spectral-feature nonlinear controls are architecture-inspired NM-ROMs, not faithful native coordinate-MLP Tunable reproductions. Larger spans and the planned independent NM-ROM competitors remain relevant.

![Error against active variables](b1-representation-capacity.png)

![Actual-grid witnesses](b1-actual-field-witnesses.png)

The image display is subsampled for readability from saved actual-grid arrays. Errors use the complete arrays. The two rows are different worst cases and are not a matched-case attribution test.

Independent fixed-code scoring confirms local POD24 on every snapshot at the denser grid: worst {pct(directaudit['dense_scoring']['2048']['worst'])}, with largest change from the resolved physical metric {directaudit['max_metric_2048_change']:.3g}. The actual-output witness is {pct(directaudit['dense_witness']['relative_l2'])}. All {directaudit['zero_child_jacobian']['independent_directions']} of {directaudit['zero_child_jacobian']['child_variables']} child directions remain numerically independent after removing the parent; their squared singular-value ratio is {directaudit['zero_child_jacobian']['smallest_to_largest']:.3g}.

The following allocation controls use exactly the same local checkpoint and coefficient dimension per node. Uniform and prescribed entries at each matched node count isolate an allocation/depth-distribution effect. The random allocation is a diagnostic control, not the deployed policy.

| Allocation and retained nodes | Total variables | Worst projection |
|---|---:|---:|
{allocation_rows}

## Retraction and numerical controls

The preliminary rich-span score {pct(alias['scoring']['128']['worst'])} was invalid as a gate result: unchanged coefficients gave {pct(alias['scoring']['256']['worst'])} on denser scoring. Too few integration points per small patch missed oscillatory directions. This apparent pass is retracted; its outputs are retained. The repaired metric resolves every cell-core and halo boundary using tensor integration. The independent integration-rule difference is {resolved['root_controls']['128']['max_integration_difference']:.3g}, with actual-grid agreement recorded above. A first repair implementation failed before scientific output due to the reference’s interval/mode indexing convention; its allocation and logs are retained.

The historical attribution JSON accidentally calls a selected update number `old_dev_worst`. It is not an error metric; this report reads the actual global error from its source run. The source label is corrected for future reruns, and the original raw file remains unchanged.

## Advancement and remaining work

The richer direct local space passes the representation screen and warrants dense audits and a separately labelled fixed-tree physics study. Shared nonlinear children are now trained against the full assembled physical metric, with jointly refitted parent and child variables at multiple depths. A small fixture verifies the overlap-aware fitting algebra and exact zero-child transfer. Trained zero-child directions and fixed-code actual output require separate qualification.

The focused continuation changes the training allocation mixture for its nonlinear compression pilot; it is explicitly documented in B1-SHARED-CHILD.md. Same-checkpoint uniform/prescribed controls remain required before an allocation-policy claim. No truth-assisted online refinement is allowed. The verified fine-upwind weak operator remains the primary PDE formulation. Initialization, assembly, failed trials, solving, fallback and requested output must all enter later complete-query timing.

Original recipes, unsuccessful repairs and the original decision remain in [the retained B0 report](B0-retained-representation.md). CROM, SNF-ROM, hyper-reduced masked autoencoders and conditional equation-driven CoLoRA have not yet been reproduced in confirmation; they are not counted as defeated. The older burgers-quadtree campaign remains paused and untouched. No merge or three-dimensional work.

## Accounting and provenance

Generated {stamp}. Terminal allocations recorded in the current ledger total {charged} GPU-seconds ({charged/3600:.6f} GPU-hours); nonterminal or not-yet-reconciled names: {', '.join(live) if live else 'none'}. Outstanding reservations are enforced by the submission helper, not omitted from its admission calculation. The finite B1 ceiling, recipe definitions, source commits, GPU preflights, raw arrays, checkpoint histories and checksum-verified retention records remain in this lane. This report and all figures are generated from saved data; their source hashes are recorded beside them.

## Glossary

- Representation/projection error: best reconstruction found while the reference field is available; this is not a PDE solve.
- Parent: global field retained when local corrections are added.
- Child variables: independently adjustable numbers controlling one local correction.
- POD: a basis learned by singular-value decomposition of training fields or coefficients.
- Gram metric: field inner products that retain overlapping-patch interactions.
- Current-time relative L2: physical root-mean-square error divided by the reference norm at that same time.
- Worst all: largest error across every snapshot and trajectory in all development panels.
- Alignment panel: prescribed cases near tree cell edges/corners, reported separately from random cases.
- Aliasing: integration samples miss variation between sample locations and report an incorrect fit error.
- Zero-child transfer: adding zero-valued correction codes leaves the existing field unchanged.
- Allocated GPU-hours: reserved hardware time actually elapsed, including setup and failures.
- Complete query: initialization, adaptation, assembly, solving, fallback and requested reconstruction together.
'''
old_report.write_text(text)
(OUT/'B1-report-source-hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
main=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');assets=main/'2026-11-18-tree-embedding-representation-assets';assets.mkdir(exist_ok=True)
for p in OUT.iterdir():
    if p.suffix in ('.png','.svg') or p.name==archive.name:
        shutil.copy2(p,assets/p.name);text=text.replace(f']({p.name})',f']({assets.name}/{p.name})')
(main/name).write_text(text)
print('Generated',old_report)
