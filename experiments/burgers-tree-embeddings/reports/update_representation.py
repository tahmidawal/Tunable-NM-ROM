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
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    svg=OUT/(name+'.svg');fig.savefig(svg,bbox_inches='tight',metadata={'Date':None});plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
def pct(x):return f'{100*x:.4f}%'
def panel(err,labels,prefix):
    ix=[i for i,n in enumerate(labels) if (n.startswith('align') if prefix=='alignment' else n.startswith('dev') and ((int(n[3:])<16)==(prefix=='original')))]
    return float(np.max(np.asarray(err)[ix]))

old=read('auditr1');g0=read('global01');attr=read('b1attrib01');p16=read('b1parent16');resolved=read('b1resolved2')
alias=read('b1spanaudit');span=read('b1expanded');cache=read('b1cache')
fullglobal=read('b1globalfull');refit8=read('b1child8refit')
headspan=read('b1headspan2')
stable=read('b1stableglobal');refit16=read('b1child16refit')
growth=read('b1childgrowth64')
labels=[r['case'] for r in cache['cases']['dev']]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'burgers-tree-embeddings'})

fig,ax=plt.subplots(figsize=(8.3,4.8))
xx=[int(k) for k in span['controls']];yy=[100*v['worst'] for v in span['controls'].values()]
ax.loglog(xx,yy,'o-',color='#546d9c',label='Global POD: expanded training data')
xx=[int(k) for k,v in fullglobal['POD'].items() if v['relative_training_singular_value']>1e-12]
yy=[100*fullglobal['POD'][str(k)]['projection']['worst'] for k in xx]
ax.loglog(xx,yy,'o-',color='#546d9c')
xx=[1024]+[int(k) for k,v in fullglobal['POD'].items() if v['relative_training_singular_value']<=1e-12]
yy=[100*fullglobal['POD'][str(k)]['projection']['worst'] for k in xx]
ax.loglog(xx,yy,'o--',mfc='white',color='#546d9c',label='Global extension into numerical-null directions')
ax.loglog([v['variables'] for v in fullglobal['complete_sine'].values()],
          [100*v['projection']['worst'] for v in fullglobal['complete_sine'].values()],
          '^:',color='#9a752c',label='Complete tensor sine space (data-independent)')
ax.loglog([int(k) for k in stable['controls']],[100*v['worst'] for v in stable['controls'].values()],
          'P-',color='#c45c41',label='Global POD512 + prescribed Fourier completion')
xx=[v['active_variables'] for v in cache['direct_controls'].values()];yy=[100*v['projection']['worst'] for v in cache['direct_controls'].values()]
ax.loglog(xx,yy,'o-',color='#00877c',label='Shared local POD + global parent')
for k,v in cache['direct_controls'].items():ax.annotate(f'{k}/node',(v['active_variables'],100*v['projection']['worst']),xytext=(4,7),textcoords='offset points',fontsize=8)
trained=[]
for name in ['b1global64','b1global128','b1globalwide128','b1child8','b1child16']:
    if not (H/'results'/name/'output/result.json').exists():continue
    r=read(name)
    if not r['status'].startswith('complete_'):continue
    trained.append((name,r))
    if 'global' in name:ax.scatter([r['latent']],[100*r['worst_projection']],marker='s',s=50,color='#8f54a0',label=f'Global nonlinear {r["latent"]}'+(' (wide)' if 'wide' in name else ''))
    else:ax.scatter([r['active_variables']],[100*r['worst_projection']],marker='D',s=50,label=f'Nonlinear child {r["latent"]}')
ax.scatter([growth['active_variables']],[100*growth['projection']['worst']],marker='*',s=120,color='#e36a0c',label='Same child16 head, 64 retained nodes')
ax.axhline(.5,color='#b5413c',ls='--',lw=1,label='Representation gate (every snapshot)')
ax.set(xlabel='Total adjustable variables, including retained parents',ylabel='Worst current-time relative L2 error (%)',
       title='Expanded panel: shared local corrections versus global variables')
ax.grid(alpha=.17,which='both');ax.legend(fontsize=8,loc='lower left');figure(fig,'b1-representation-capacity')

fig,axs=plt.subplots(3,3,figsize=(10,9),constrained_layout=True)
for row,name,title in [(0,'auditr1','Original learned local model'),(1,'b1directaudit','Shared local POD24'),(2,'b1childgrowth64','Nonlinear child16, 64 nodes')]:
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

fig,ax=plt.subplots(figsize=(7.5,4),constrained_layout=True)
ix=np.arange(len(headspan['models']));width=.35
ax.bar(ix-width/2,[100*r['panels']['dev']['worst'] for r in headspan['models']],width,label='Unrestricted coefficients in frozen output span')
ax.bar(ix+width/2,[100*r['head_projection']['worst'] for r in headspan['models']],width,label='Actual nonlinear head: best found fit')
ax.set(xticks=ix,xticklabels=[f"{r['latent']} variables\n{r['affine_span_rank']} field directions" for r in headspan['models']],
       ylabel='Worst development projection error (%)',title='Frozen global heads: shape-space limit plus coding/fitting gap')
ax.legend(fontsize=8);figure(fig,'b1-global-head-limits')

name='2026-11-18-tree-embedding-representation.md';old_report=OUT/name;archive=OUT/'B0-retained-representation.md'
if not archive.exists():shutil.copy2(old_report,archive)
bestold=min(old['direct_fine_projection'],key=lambda x:x['projection']['worst'])
lines=[]
for k,v in cache['direct_controls'].items():
    e=v['projection']['per_snapshot'];lines.append(f"| Direct local {k} | {v['active_variables']} | {pct(panel(e,labels,'original'))} | {pct(panel(e,labels,'new'))} | {pct(panel(e,labels,'alignment'))} | {pct(v['projection']['worst'])} |")
for n,r in trained:
    chosen=next(c for c in r['checkpoints'] if c['update']==r['selected_update']);e=chosen['projection']['per_snapshot']
    lines.append(f"| {'Global' if 'global' in n else 'Child'} nonlinear {r['latent']} {'wide' if 'wide' in n else ''} | {r.get('active_variables',r['latent'])} | {pct(panel(e,r['dev_labels'],'original'))} | {pct(panel(e,r['dev_labels'],'new'))} | {pct(panel(e,r['dev_labels'],'alignment'))} | {pct(r['worst_projection'])} |")
e=growth['projection']['per_snapshot'];gl=[v['case'] for v in growth['cases']]
lines.append(f"| Same child16 head, {growth['nodes']} nodes | {growth['active_variables']} | {pct(panel(e,gl,'original'))} | {pct(panel(e,gl,'new'))} | {pct(panel(e,gl,'alignment'))} | {pct(growth['projection']['worst'])} |")
ledger=json.loads((H/'cluster/ledger.json').read_text());terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
directaudit=read('b1directaudit')
allocation_rows='\n'.join(f"| {k} | {v['active_variables']} | {pct(v['projection']['worst'])} |" for k,v in directaudit['controls'].items())
child_allocation_text='Same-checkpoint nonlinear uniform/prescribed allocation controls are still running.'
if (H/'results/b1childallocation/output/result.json').exists():
    ac=read('b1childallocation')
    if ac['status']=='complete_frozen_child_allocation_controls':
        ar=[];fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)
        for ax,control,title in [(axs[0],directaudit['controls'],'Direct local24'),(axs[1],ac['controls'],'Frozen nonlinear child16')]:
            names=['uniform20','prescribed20','uniform84','prescribed84']
            values=[100*control[n]['projection']['worst'] for n in names]
            ax.bar(np.arange(4),values,color=['#7899ac','#00877c','#7899ac','#00877c'])
            ax.axhline(.5,color='#b5413c',ls='--',lw=1)
            ax.set(xticks=np.arange(4),xticklabels=[f"{n.replace('20',' 20').replace('84',' 84')}\n{control[n]['active_variables']} variables" for n in names],title=title,yscale='log')
            ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
        axs[0].set_ylabel('Worst all-snapshot projection error (%)')
        fig.suptitle('Uniform versus initial-field prescribed trees: same checkpoint within each panel')
        figure(fig,'b1-allocation-controls')
        for n,v in ac['controls'].items():
            ar.append(f"| {n} | {v['active_variables']} | {pct(v['projection']['worst'])} | {pct(v['actual4096_witness']['error'])} | {v['max_stationarity']:.3g} |")
        child_allocation_text='''The frozen nonlinear head is now compared on identical node budgets. These are best-found reference-assisted fits, not certified manifold minima; stationarity and actual-grid worst witnesses are explicit. No new weights are trained. A newly passing allocation still needs all-snapshot dense scoring before physics admission.

| Allocation | Variables | Worst all | Actual4096 worst witness | Maximum fit stationarity |
|---|---:|---:|---:|---:|
'''+ '\n'.join(ar)+'\n\n![Same-checkpoint allocation controls](b1-allocation-controls.png)'
global_rows='\n'.join(f"| Global saved POD prefix {k} | {v['variables']} | {pct(v['projection']['worst'])} | {v['relative_training_singular_value']:.3g} |" for k,v in fullglobal['POD'].items())
global_rows+='\n'+'\n'.join(f"| Complete sine {k} per axis | {v['variables']} | {pct(v['projection']['worst'])} | Not trained |" for k,v in fullglobal['complete_sine'].items())
global_rows+='\n'+'\n'.join(f"| POD512 + prescribed Fourier completion | {k} | {pct(v['worst'])} | Not a null-tail completion |" for k,v in stable['controls'].items())
head_rows='\n'.join(f"| {v['latent']} | {v['feature_count']} | {v['affine_span_rank']} | {pct(v['panels']['dev']['worst'])} | {pct(v['head_projection']['worst'])} |" for v in headspan['models'])
pde_runs=[];incomplete_pde=[]
for run in ['b1weakpilot1','b1weakL2048','b1weakL4096','b1weakdt125','b1weakdt625','b1cn2048','b1cn2048half','b1cnM8','b1cnM16','b1cnM32','b1cnQ512','b1cnQ1024','b1childpde','b1childpdeR1','b1childpdeR2']:
    if (H/'results'/run/'output/result.json').exists():
        r=read(run)
        if r['status'] in ['complete_fixed_tree_diagnostic_pilot','complete_coupled_child_diagnostic_pilot']:pde_runs.append((run,r))
        else:incomplete_pde.append(f"{run}: {r['status']}; retained as incomplete/failed, not a passing pilot")
pde_rows=[]
fig,axs=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True,sharey=True)
for run,r in pde_runs:
    for i,c in enumerate(r['cases']):
        for method,v in c['methods'].items():
            evolved=pct(max(v['physical_error'][1:])) if len(v['physical_error'])>1 else 'no evolved output'
            pde_rows.append(f"| {c['case']} | {method} | {r['L']} | {r.get('time_integrator','be').upper()} | {c['common_tests']} | {r.get('quadrature_intervals',r['L'])} | {r['dt']:g} | {evolved} | {v['all_steps_stationary']} | {v['physical_accuracy_pass']} |")
            if run in ['b1weakL4096','b1cn2048','b1cnM8','b1cnM16','b1childpde']:
                model='Global960' if method=='global_stable960' else ('Global768' if method.startswith('global') else ('Child16' if 'child' in method else 'Direct24'))
                axs[i].plot(np.arange(len(v['physical_error']))*.05,np.array(v['physical_error'])*100,
                    '--' if method.startswith('global') else '-',label=f"{model} {r.get('time_integrator','be').upper()}, L{r['L']}, M{c['common_tests']}")
            axs[i].set(title=c['case'],xlabel='Time');axs[i].axhline(1,color='gray',ls=':',lw=.7);axs[i].grid(alpha=.15)
axs[0].set_ylabel('Current-time physical error (%)')
handles,legend=axs[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.5,-.23),ncol=3,fontsize=7)
figure(fig,'b1-fixed-tree-physics')
diagnostics={'b1cn2048half':'Time step / 2','b1cnM8':'5504 to11008 tests','b1cnM16':'11008 to22016 tests','b1cnM32':'22016 to44032 tests','b1cnQ512':'Q512 versus dense'}
checks=[(n,r) for n,r in pde_runs if n in diagnostics];fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
for j,prefix in enumerate(['tree','global']):
    changes=[max(max(v['previous_state_relative_change']) for c in r['cases'] for method,v in c['methods'].items() if method.startswith(prefix)) for n,r in checks]
    ax.bar(np.arange(len(checks))+(j-.5)*.34,np.array(changes)*100,.34,label=prefix.capitalize())
ax.axhline(.1,color='#b5413c',ls='--',label='0.1% sensitivity threshold')
ax.set(xticks=np.arange(len(checks)),xticklabels=[diagnostics[n] for n,r in checks],yscale='log',ylabel='Largest returned-field change (%)',title='Independent numerical checks; every requested time and pilot case')
ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8);figure(fig,'b1-numerical-sensitivities')
coupled_rows=[]
for run,r in pde_runs:
    if not run.startswith('b1childpde'):continue
    for c in r['cases']:
        v=c['methods']['tree_child16_n64'];s=np.asarray(v['step_status'],dtype=float)
        coupled_rows.append(f"| {run} | {c['case']} | {pct(max(v['physical_error'][1:]))} | {int(np.sum(s[:,1]>r['stationarity_tolerance']))}/{len(s)} | {np.max(s[:,1]):.3g} | {v['initial_fit_stationarity_or_linear_residual']:.3g} |")
charged=sum(r.get('elapsed_seconds',0) for r in ledger if r.get('state') in terminal)
live=[r['name'] for r in ledger if r.get('state') not in terminal]
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
text=f'''# Burgers tree embeddings: representation diagnosis and continuation

This generated report covers the original bounded failure and the user-authorized continuation. Representation results are development evidence; nonlinear compression, dense audits and fixed-tree physics are still being investigated. No complete PDE-runtime win or final-test result is established.

## Why the original best error was still high

The original best tree had {pct(bestold['projection']['worst'])} worst projection error at 128 adjustable variables. Its actual-grid witness remains {pct(old['dense_witness']['grids']['4096']['relative_l2'])}; that failure is real, not the subsequently retracted quadrature result.

The evidence supports several contributing limitations. The original parent alone has {pct(g0['heads']['16']['worst_projection'])} worst development error. Better-scaled training with a richer frozen feature span reduces selected training error from {pct(train[0])} to {pct(train[1])}, but development error remains {pct(dev[1])}. Thus optimization and generalization are both relevant; changing scaling and the feature bank together does not isolate one cause. The old head’s relative departure from its best affine map is {pct(attr['old_parent_nonlinearity']['relative_departure_from_best_affine'])}, so describing it as exactly linear would be wrong.

A separate frozen-weight audit identifies a definite global-head restriction: a linear skip plus a finite-width final hidden layer confines every output to a fixed affine span. Giving that span unrestricted coefficients produces the bounds below. Their remaining error cannot be fixed by additional code fitting on the same frozen head. The gap above the bound can reflect both the nonlinear image and imperfect fitting; it does not isolate those two causes. The smaller normalized model uses the original cohort; the larger two use the expanded cohort.

| Global latent variables | Frozen features | Output-space rank | Relaxed output-span error | Nonlinear head error |
|---|---:|---:|---:|---:|
{head_rows}

![Frozen global head limits](b1-global-head-limits.png)

Attribution on the old fixed trees provisionally improves when parent and local coefficient restrictions are relaxed. Those sampled-integration diagnostics also increase variable counts and are not an additive causal decomposition. Stronger evidence is the resolved richer local span: its worst original-panel error is {pct(resolved['root_controls']['128']['projection']['worst'])} with {resolved['root_controls']['128']['variables']} coefficients, confirmed by actual output at {pct(resolved['dense_witness']['errors']['4096'])}. This rules out a universal error floor at the original value for the broader local representation. It does not yet establish efficient nonlinear compression.

![Training and numerical diagnosis](b1-training-and-integration.png)

## Expanded matched-data comparison

Training uses the registered prefix expansion to 128 trajectories. Development has 32 random trajectories and the same 12 alignment cases; every one of the 11 snapshot times is scored in the current-time physical norm. Final-test solution fields remain unopened. Shared local POD uses training coefficients only, fixed windows and retained parents. Its allocations use the known initial field and a sampled speed estimate, not evolved reference fields or a certified propagation bound.

| Representation | Total variables | Original random panel | Added random panel | Alignment panel | Worst all |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

The direct local entries use resolved physical Gram integration; actual-grid audits are separate. Spectral-feature nonlinear controls are architecture-inspired NM-ROMs, not faithful native coordinate-MLP Tunable reproductions.

The wider global128 repair uses960 stable features and hidden widths256,256,960. Its selected projection error remains {pct(read('b1globalwide128')['worst_projection'])}, versus {pct(read('b1global128')['worst_projection'])} before the combined feature/output-width change. The selected fits have maximum stationarity below1e-9. Widening this recipe did not solve its generalization/coding limitation; the unsuccessful checkpoint and full training history remain. This repair is not an isolated attribution test because both feature span and hidden widths change.

The extended global control reaches {pct(fullglobal['POD']['1280']['projection']['worst'])} with {fullglobal['POD']['1280']['variables']} variables, better than the admitted direct tree at its larger variable count. Retain that result, with an essential qualification: its trailing training singular value is only {fullglobal['POD']['1280']['relative_training_singular_value']:.3g} of the largest. These are valid saved orthonormal spaces extended into numerically null training directions, not reliably learned variance directions. Their stability under retraining is unqualified. The complete data-independent sine space provides a separate control without this ambiguity. Neither variable counts nor projection errors establish PDE cost.

Reordering the identical training rows in two prescribed permutations gives global1280 errors {', '.join(pct(v['controls']['1280']['projection']['worst']) for v in stable['permutations'])}. The accuracy changes modestly in this limited check; the directions remain numerically null. A separate, explicitly constructed global space avoids those arbitrary SVD directions: retain the reliable training POD512 prefix and append orthogonalized low-frequency sine directions in fixed order. It reaches {pct(stable['controls']['960']['worst'])} with960 variables, compared with frozen child16 refitting at {pct(refit16['worst_projection'])} with the same count. Its measured orthogonality defect is {stable['orthogonality_defect']:.3g}. This is a POD-plus-Fourier control, not pure POD or a neural method. It currently defeats the child's error-per-variable comparison; cost and actual-output qualification remain separate.

| Global span control | Variables | Worst all | Relative trailing training singular value |
|---|---:|---:|---:|
{global_rows}

![Error against active variables](b1-representation-capacity.png)

![Actual-grid witnesses](b1-actual-field-witnesses.png)

The image display is subsampled for readability from saved actual-grid arrays. Errors use the complete arrays. The rows show different worst cases and are not a matched-case attribution test.

Independent fixed-code scoring confirms local POD24 on every snapshot at the denser grid: worst {pct(directaudit['dense_scoring']['2048']['worst'])}, with largest change from the resolved physical metric {directaudit['max_metric_2048_change']:.3g}. The actual-output witness is {pct(directaudit['dense_witness']['relative_l2'])}. All {directaudit['zero_child_jacobian']['independent_directions']} of {directaudit['zero_child_jacobian']['child_variables']} child directions remain numerically independent after removing the parent; their squared singular-value ratio is {directaudit['zero_child_jacobian']['smallest_to_largest']:.3g}.

The following allocation controls use exactly the same local checkpoint and coefficient dimension per node. Uniform and prescribed entries at each matched node count isolate an allocation/depth-distribution effect. The random allocation is a diagnostic control, not the deployed policy.

| Allocation and retained nodes | Total variables | Worst projection |
|---|---:|---:|
{allocation_rows}

## Retraction and numerical controls

The preliminary rich-span score {pct(alias['scoring']['128']['worst'])} was invalid as a gate result: unchanged coefficients gave {pct(alias['scoring']['256']['worst'])} on denser scoring. Too few integration points per small patch missed oscillatory directions. This apparent pass is retracted; its outputs are retained. The repaired metric resolves every cell-core and halo boundary using tensor integration. The independent integration-rule difference is {resolved['root_controls']['128']['max_integration_difference']:.3g}, with actual-grid agreement recorded above. A first repair implementation failed before scientific output due to the reference’s interval/mode indexing convention; its allocation and logs are retained.

The historical attribution JSON accidentally calls a selected update number `old_dev_worst`. It is not an error metric; this report reads the actual global error from its source run. The source label is corrected for future reruns, and the original raw file remains unchanged.

## Advancement and remaining work

The direct local POD24 model passes representation and its independent dense audit, so fixed-tree physics has begun. Frozen-weight, longer dense fitting of child8 reduces its worst error to {pct(refit8['worst_projection'])}; maximum stationarity remains {refit8['max_stationarity']:.3g}. The small change weakens the explanation that its entire remaining error is a short code fit, but it is not a proof of a global minimum. Shared nonlinear children train against the assembled physical metric, with jointly refitted parent and child variables at multiple depths.

Child16 also finishes its registered training and frozen-weight fitting audit. Longer fitting reaches {pct(refit16['worst_projection'])}, with maximum stationarity {refit16['max_stationarity']:.3g}. Every random development trajectory clears the representation threshold, but alignment cases still fail. A finite same-checkpoint prescribed-tree growth diagnostic is registered to test that specific allocation/compression bottleneck while preserving parents and windows; it is not yet a qualified online adaptive method.

That frozen-head growth diagnostic now passes: {growth['nodes']} retained nodes give {growth['active_variables']} total variables, worst physical projection {pct(growth['projection']['worst'])} and actual4096 witness {pct(growth['actual4096_witness']['error'])}. Fixed-code2048 scoring covers every snapshot; its largest change from the resolved metric is {growth['max_scoring_change']:.3g}. Zero-extension preserves the previous field with maximum relative change {growth['max_zero_child_transfer']:.3g}; every audited geometry retains all {growth['nodes']*growth['latent']} independent child directions after root removal. The head weights did not change. This achieves approximately the direct24 model's reconstruction accuracy with fewer variables, while the global960 control remains more accurate at a smaller count. A coupled nonlinear PDE solve, same-checkpoint allocation controls and complete costs remain required. This is initial-callable prescribed growth, not online residual-driven adaptation.

## Fixed-tree PDE diagnosis

These pilots solve complete trajectories from the shared initial callable with the frozen prescribed tree and global POD768 control. Within each pilot, both use the same exact fine-grid upwind weak operator, time integrator, common test set and integration rule. BE is backward Euler; the registered CN repair is Crank–Nicolson. Q denotes integration intervals; Q=L is dense integration. Smaller Q retains fine stencil spacing1/L and requires independent returned-state refinement. Reference fields are opened for scoring only after both methods finish each case. The table reports all requested evolved times, not only the final output.

| Case | Method | Spatial intervals | Integrator | Tests M | Integration Q | Time step | Worst evolved error | Every step stationary | Physical accuracy pass |
|---|---|---:|---|---:|---:|---:|---:|---|---|
{chr(10).join(pde_rows)}

![Fixed-tree physics diagnosis](b1-fixed-tree-physics.png)

![Independent numerical sensitivities](b1-numerical-sensitivities.png)

Retained incomplete PDE outcomes: {'; '.join(incomplete_pde) or 'none in the collected pilot records'}. The trajectory figure emphasizes the dense BE/CN and test-enrichment comparisons; the table retains every completed pilot setting. A low physical error alone does not establish numerical convergence. The change from5504 to11008 tests is large for the tree. Its next enrichment to22016 changes the alignment-case field by0.106838%, just above the0.1% sensitivity limit;11008 is therefore still not admitted. A further independently registered enrichment is needed. The global control is much less sensitive to these added tests.

The first nonlinear-child PDE pilot reaches physical errors below1% on all three exposed cases, but160 of300 time steps miss stationarity1e-8. The initial alignment fit is also underconverged. Useful zero-child directions and accurate reconstructions do not guarantee an easy nonlinear online solve. The first bounded repair raises only the PDE LM cap from30 to120, keeping the checkpoint, initial fits, equations, integration and tolerances fixed. Its results, including any failures, are recorded below. Initial-fit stationarity is distinct from time-step stationarity; a passing step alone does not qualify initialization.

| Nonlinear PDE run | Case | Worst evolved physical error | Nonstationary steps | Worst step stationarity | Initial-fit stationarity |
|---|---|---:|---:|---:|---:|
{chr(10).join(coupled_rows)}

Pulse errors remain above the physical gate in the initial pilot for both methods despite tightly converged weak solves. Spatial, temporal, test and integration refinements distinguish numerical error from reduced dynamics. The initial pilot omitted explicit burn-in immediately before timing: those original wall times are retained as unqualified diagnostics and support no speed comparison. Follow-ups include burn-in. Pilot clocks also charge the joint comparison setup to each method; they are not valid method-specific speed ratios. Complete timing must measure each method's own initialization, assembly and output. The small pilot is not all-development qualification, deployable adaptation, or a complete-runtime benchmark.

The focused continuation changes the training allocation mixture for its nonlinear compression pilot; it is explicitly documented in B1-SHARED-CHILD.md. No truth-assisted online refinement is allowed. The verified fine-upwind weak operator remains the primary PDE formulation. Initialization, assembly, failed trials, solving, fallback and requested output must all enter later complete-query timing.

{child_allocation_text}

Original recipes, unsuccessful repairs and the original decision remain in [the retained B0 report](B0-retained-representation.md). CROM, SNF-ROM, hyper-reduced masked autoencoders and conditional equation-driven CoLoRA have not yet been reproduced in confirmation; they are not counted as defeated. The older burgers-quadtree campaign remains paused and untouched. No merge or three-dimensional work.

## Accounting and provenance

Generated {stamp}. Terminal allocations recorded in the current ledger total {charged} GPU-seconds ({charged/3600:.6f} GPU-hours); nonterminal or not-yet-reconciled names: {', '.join(live) if live else 'none'}. Outstanding reservations are enforced by the submission helper, not omitted from its admission calculation. The finite B1 ceiling, recipe definitions, source commits, GPU preflights, raw arrays, checkpoint histories and checksum-verified retention records remain in this lane. This report and all figures are generated from saved data; their source hashes are recorded beside them.

## Glossary

- Representation/projection error: best reconstruction found while the reference field is available; this is not a PDE solve.
- Parent: global field retained when local corrections are added.
- Child variables: independently adjustable numbers controlling one local correction.
- POD: a basis learned by singular-value decomposition of training fields or coefficients.
- Numerical-null directions: basis directions with training variation at rounding scale; their orientation is not reliably determined by the data.
- Complete sine space: every tensor-product sine mode up to the stated axis count, independent of training.
- Stationarity: the fitted weak residual's gradient is small; this alone does not guarantee physical accuracy.
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
