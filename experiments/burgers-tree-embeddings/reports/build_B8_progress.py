"""Publish the retained B8 primary-selection evidence without loading field arrays."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H=Path(__file__).resolve().parents[1]
REPORT=H/'reports/2026-11-19-neural-child-progress.md'
ASSETS=H/'reports/2026-11-19-neural-child-assets'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def verify_retention(name, files):
    manifest=read(H/f'manifests/retention-{name}.json')
    retained={row['file']:row for row in manifest['files']}
    for relative in files:
        path=H/'results'/name/relative
        assert relative in retained
        assert digest(path)==retained[relative]['sha256']
        assert path.stat().st_size==retained[relative]['bytes']


def main():
    verify_retention('b8childprimary',['output/result.json','output/allocation_comparison.pkl'])
    for name in ['b7parentrepair','b8trainrefs','b8devrefs']:
        verify_retention(name,['output/result.json'])
    result_path=H/'results/b8childprimary/output/result.json'
    r=read(result_path)
    assert r['status']=='complete_nonlinear_child_subset' and r['primary_only']
    assert not r['final_test_opened'] and r['parent_weights_frozen']
    comparison_path=result_path.parent/'allocation_comparison.pkl'
    assert digest(comparison_path)==r['comparison_sha256']
    # This retained comparison contains only numerical arrays/scalars, not a
    # network checkpoint. No JAX import or scientific reconstruction is needed.
    with comparison_path.open('rb') as f:rows=pickle.load(f)
    parent_path=H/'results/b7parentrepair/output/result.json';p=read(parent_path)
    assert p['status']=='complete_stable_parent' and not p['final_test_opened']
    assert r['parent_sha256']==p['selected_sha256']
    protocol=read(Path('/home/tahmid/Dev/QuadtreeNMROM/tree-embedding-protocol.json'))
    gate=float(protocol['contracts']['projection_gate'])
    layouts=['uniform','prescribed']
    bylayout={layout:sorted([row for row in rows if row['layout']==layout],key=lambda x:x['index']) for layout in layouts}
    ids=[row['index'] for row in bylayout[layouts[0]]]
    assert len(set(ids))==len(ids) and ids
    assert all([row['index'] for row in bylayout[k]]==ids for k in layouts)
    assert all(np.isfinite(row['independent_error']) and np.isfinite(row['status']).all() for row in rows)
    cases=list(dict.fromkeys(row['case'] for row in bylayout[layouts[0]]))
    times=sorted(set(row['time'] for row in bylayout[layouts[0]]))
    assert len(ids)==len(cases)*len(times)
    assert all(row['case']==r['dev_labels'][row['index']//11] and len(row['codes'])==r['active_variables'] for row in rows)
    parent_errors=np.asarray(p['independent_errors'])[ids]
    summary=[dict(label=f"Parent ({p['latent']} variables)",errors=parent_errors,
                  unfinished=0 if p['unfinished_fits']==0 else None)]
    for layout in layouts:
        summary.append(dict(label=f"{layout.capitalize()} tree ({r['active_variables']} variables)",
                            errors=np.asarray([row['independent_error'] for row in bylayout[layout]]),
                            unfinished=sum(not bool(row['status'][-1]) for row in bylayout[layout])))
    ASSETS.mkdir(parents=True,exist_ok=True)
    ncols=2;nrows=(len(cases)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(12,4.1*nrows),squeeze=False,sharex=True,sharey=True)
    colors=['#2563eb','#d97706','#16803c']
    for ax,case in zip(axes.ravel(),cases):
        selection=[i for i,row in enumerate(bylayout['uniform']) if row['case']==case]
        x=np.asarray([bylayout['uniform'][i]['time'] for i in selection])
        for s,color in zip(summary,colors):
            ax.plot(x,100*s['errors'][selection],'-o',color=color,lw=1.8,ms=5,label=s['label'])
        for layout,color in zip(layouts,colors[1:]):
            incomplete=[i for i in selection if not bool(bylayout[layout][i]['status'][-1])]
            if incomplete:ax.scatter([bylayout[layout][i]['time'] for i in incomplete],
                                     [100*bylayout[layout][i]['independent_error'] for i in incomplete],
                                     marker='x',s=95,color='black',linewidths=1.7,zorder=5)
        ax.axhline(100*gate,color='black',ls=':',lw=1.3)
        ax.set_title(case);ax.set_yscale('log');ax.set_xticks(times)
        ax.grid(alpha=.2,which='both');ax.set_xlabel('Snapshot time');ax.set_ylabel('Current-time relative L2 error (%)')
    for ax in axes.ravel()[len(cases):]:ax.set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles += [Line2D([0],[0],color='black',ls=':',label=f'{100*gate:g}% representation target'),
                Line2D([0],[0],color='black',marker='x',ls='none',label='Unfinished embedding fit')]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle(f'B8 primary selection: {len(cases)} cases × {len(times)} times\nOffline fitted reconstructions; not PDE rollouts or full-cohort qualification',fontsize=14)
    fig.tight_layout(rect=(0,.08,1,.93))
    plot=ASSETS/'B8-primary-selection.png';fig.savefig(plot,dpi=170);plt.close(fig)
    train_ref=read(H/'results/b8trainrefs/output/result.json')
    dev_ref=read(H/'results/b8devrefs/output/result.json')
    assert train_ref['cohort']=='train' and dev_ref['cohort']=='dev'
    assert train_ref['all_ladders_pass'] and dev_ref['all_ladders_pass']
    assert not train_ref['final_test_opened'] and not dev_ref['final_test_opened']
    assert not {row['case'] for row in train_ref['cases']} & {row['case'] for row in dev_ref['cases']}
    final=max(r['history'],key=lambda row:row['update']);refresh=final['training_refit']
    expected_train=len(r['train_labels'])*len(protocol['data']['snapshot_times'])
    assert refresh['count']==expected_train
    selected=next(row for row in r['history'] if row['update']==r['selected_update'])
    initial_worst=float(parent_errors.max())
    best=min(summary[1:],key=lambda row:float(row['errors'].max()))
    factor=initial_worst/float(best['errors'].max())
    fd_note='Independent-discretization checks remain pending.'
    fd_path=H/'results/b8expandedfd/output/result.json'
    if fd_path.exists():
        verify_retention('b8expandedfd',['output/result.json'])
        fd=read(fd_path);assert fd['status']=='complete'
        worst_fd=max(max(row['levels'][-1]['relative_error']) for row in fd['cases'])
        fd_note=f"The separate independent finite-difference check covers {len(fd['cases'])} cases and passed: {fd['passed']}. Its largest finest-grid discrepancy is {100*worst_fd:.6f}%; retained proof: results/b8expandedfd/output/result.json."
    text=f'''# Nonlinear child training: preliminary selection-panel results

The first B8 child model improves reconstruction on the retained primary selection panel, but does not meet the unchanged {100*gate:g}% representation target. These are offline, field-assisted fits on {len(cases)} development cases at {len(times)} times ({len(ids)} fields per layout), not all-case qualification, PDE predictions, or complete-solve timings. No actual4096 output has yet been qualified for this child checkpoint.

The frozen parent has {p['latent']} variables. Each tree adds {r['nodes']} retained nodes with {r['child_k']} independently adjustable nonlinear child variables per node, for {r['active_variables']} total variables. Each child maps its embedding into {r['features']} newly learned local features. Uniform and prescribed layouts use the same selected checkpoint at update {r['selected_update']}; prescribed allocation uses the initial field, while these offline embedding fits use the reference snapshot.

| Model on exactly the same selection fields | Worst error | Median error | Above representation target | Unfinished fits |
|---|---:|---:|---:|---:|
'''
    for row in summary:
        e=row['errors'];unfinished='not separately available' if row['unfinished'] is None else f"{row['unfinished']}/{len(e)}"
        text+=f"|{row['label']}|{100*e.max():.6f}%|{100*np.median(e):.6f}%|{int(np.sum(e>gate))}/{len(e)}|{unfinished}|\n"
    text+=f'''
The better tree's worst selection error is {factor:.3f} times smaller than the paired parent's worst selection error. This comparison adds variables and does not establish an advantage over a larger global neural decoder, nor a speed advantage. The parent values above are restricted to these same {len(ids)} fields; the separately retained full-parent evaluation contains {len(p['independent_errors'])} fields and has worst error {100*p['worst_independent_error']:.6f}%.

![Paired primary-selection reconstruction errors](2026-11-19-neural-child-assets/B8-primary-selection.png)

Each marker is an independently fitted snapshot. Lines connect the sampled times for readability; they do not show a PDE trajectory. Black crosses identify fits that missed the unchanged stopping test, even if their reconstruction error is small.

The final training-code refresh at update {final['update']} evaluated all {refresh['count']} training snapshots: worst fitted training error {100*refresh['worst']:.6f}%, with {refresh['unfinished']} unfinished fit(s). Earlier refreshes evaluated one rotating time per training trajectory; their worst errors are not a monotone or directly comparable all-training learning curve. The selected checkpoint's prescribed selection panel has {selected['unfinished']} unfinished fit(s). Full held-out scoring, trained-child tangent checks, actual4096 output, and PDE evolution remain necessary before admission.

## Expanded families

Separate reference-generation records contain {len(train_ref['cases'])} new training trajectories and {len(dev_ref['cases'])} held-out development trajectories, balanced between ridges and two-pulse fields. Their registered spectral space/time ladders pass. {fd_note} The completed child model above was trained on the original cohort only, so these figures do not measure learned performance on the expanded families. No final-test solution fields were opened.

## Evidence

- Child allocation comparison: `{comparison_path.relative_to(H)}`; SHA256 `{digest(comparison_path)}`.
- Child result: `{result_path.relative_to(H)}`; SHA256 `{digest(result_path)}`.
- Parent result: `{parent_path.relative_to(H)}`; SHA256 `{digest(parent_path)}`.
- Expanded train/dev references: `results/b8trainrefs/output/result.json` and `results/b8devrefs/output/result.json`.
- Generated by `reports/build_B8_progress.py` from retained arrays and JSON; no field reconstruction was rerun for this report.

## Glossary

**Parent:** the global nonlinear neural decoder. **Child embedding:** independently adjustable local correction variables. **Variables:** all adjustable parent and child coordinates, not neural-network weights. **Uniform/prescribed:** different retained-node layouts using the same trained child network. **Selection panel:** development snapshots used to choose a checkpoint; not a final test. **Current-time relative L2 error:** physical reconstruction error divided by the reference field norm at that same time. **Worst/median:** maximum/middle error among the paired selection fields. **Above target:** fields exceeding the assembled representation threshold. **Unfinished fit:** an embedding optimization that missed its stopping test. **Reference ladder:** comparisons across spatial resolution and time step. **PDE rollout:** advancing embeddings with the PDE without future reference snapshots; not performed by the fits shown here.
'''
    REPORT.write_text(text)
    print(json.dumps(dict(report=str(REPORT),plot=str(plot),cases=len(cases),times=times,
                          summary=[dict(model=row['label'],worst=float(row['errors'].max()),median=float(np.median(row['errors'])),unfinished=row['unfinished']) for row in summary],
                          final_training_refresh=refresh),indent=2))

if __name__=='__main__':main()
