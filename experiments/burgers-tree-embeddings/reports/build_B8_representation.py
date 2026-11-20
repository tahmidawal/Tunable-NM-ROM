"""Plot retained scalar fit metadata only; never decode models or open final fields."""
from pathlib import Path
import hashlib,json,pickle,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parents[1];CAN=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verified(job,name):
    mp=H/'manifests'/f'retention-{job}.json';m=json.loads(mp.read_text())
    assert m['local_and_remote_verified'] and m['terminal_state']=='COMPLETED'
    row=next(v for v in m['files'] if v['file']=='output/'+name);p=H/'results'/job/'output'/name
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    return p,dict(job=job,file=name,sha256=sha(p),retention_manifest_sha256=sha(mp))
proofs=[];models={};records={}
for job,filename in [('b8childgn','nonlinear_tree.pkl'),('b8parent32','stable_parent.pkl')]:
    rp,proof=verified(job,'result.json');proofs.append(proof);r=json.loads(rp.read_text())
    assert r['status'].startswith('complete_') and not r['final_test_opened']
    cp,proof=verified(job,filename);proofs.append(proof);assert sha(cp)==r['selected_sha256']
    models[job]=pickle.loads(cp.read_bytes());records[job]=r
ref,proof=verified('b8devrefs','result.json');proofs.append(proof);references=json.loads(ref.read_text())
assert references['cohort']=='dev' and not references['final_test_opened']
families={row['case']:row['family'] for row in references['cases']}
# The retained parent data record identifies original primary trajectories;
# expanded family labels come directly from their retained reference records.
for label in models['b8parent32']['data_record']['groups']['dev']['labels']:families[label]='primary'
series=[('B7-root neural tree · uniform20','b8childgn','uniform'),('B7-root neural tree · prescribed20','b8childgn','prescribed'),('Expanded global parent32 · separate model','b8parent32',None)]
rows=[]
for label,job,layout in series:
    ev=[v for v in models[job]['evaluation'] if layout is None or v['layout']==layout]
    assert len(ev)==36
    for v in ev:
        assert v['case'] in families and np.isfinite(v['independent_error']) and v['independent_error']>0
        assert np.isfinite(np.asarray(v['status'])).all()
        rows.append(dict(model=label,job=job,layout=layout,case=v['case'],family=families[v['case']],time=float(v['time']),
            independent_error=float(v['independent_error']),fit_error=float(v['status'][0]),converged=bool(v['status'][-1]),
            stationarity=float(v['status'][1]),physical_update=float(v['status'][2]),scoring_change=float(v['scoring_change'])))
name='2026-11-20-expanded-neural-representation';assets=H/'reports'/(name+'-assets');assets.mkdir(exist_ok=True)
fig,axes=plt.subplots(3,3,figsize=(15,12),sharex=True,sharey=True,layout='constrained')
groups=['primary','ridge','two_pulse'];stats=[]
for ri,(label,job,layout) in enumerate(series):
    for col,family in enumerate(groups):
        subset=[v for v in rows if v['model']==label and v['family']==family];assert len(subset)==12
        ax=axes[ri,col];cases=sorted({v['case'] for v in subset})
        for ci,case in enumerate(cases):
            values=sorted([v for v in subset if v['case']==case],key=lambda v:v['time'])
            ax.plot([v['time'] for v in values],[100*v['independent_error'] for v in values],'-o',lw=1.5,ms=4,color=plt.cm.tab10(ci),label=case)
            unfinished=[v for v in values if not v['converged']]
            ax.scatter([v['time'] for v in unfinished],[100*v['independent_error'] for v in unfinished],marker='x',s=75,c='#d62728',linewidths=2,zorder=4)
        count=sum(not v['converged'] for v in subset)
        ax.axhline(.5,color='black',ls='--',lw=1);ax.set_yscale('log');ax.set_xticks([0,.125,.25]);ax.grid(alpha=.2)
        ax.set_title(f"{family.replace('_',' ').title()} · unfinished {count}/{len(subset)}",fontsize=11)
        ax.legend(fontsize=8,loc='best')
        if col==0:ax.set_ylabel(label+'\nIndependent relative L2 error (%)',fontsize=10)
        if ri==2:ax.set_xlabel('Snapshot time (independent reconstruction fits)')
        stats.append(dict(model=label,family=family,fields=len(subset),unfinished=count,worst=max(v['independent_error'] for v in subset),
            median=float(np.median([v['independent_error'] for v in subset])),at_or_below_half_percent=sum(v['independent_error']<=.005 for v in subset)))
fig.suptitle('Expanded representation diagnostics:128² fitting /256² independent scoring\nRed × = unfinished fit; dashed line =0.5% representation threshold\nSeparate trained models — no paired child gain; not actual4096 outputs or PDE trajectories',fontsize=14)
plot=assets/'representation_errors_by_family.png';fig.savefig(plot,dpi=160);plt.close(fig)
supplement='';longfit_summary=None
if (H/'manifests/retention-b8longfits.json').exists():
    lp,proof=verified('b8longfits','result.json');proofs.append(proof);long=json.loads(lp.read_text())
    assert long['status']=='complete_offline_fit_continuation' and not long['weights_changed'] and not long['final_test_opened']
    assert long['model_sha256']==records['b8childgn']['selected_sha256'] and len(long['rows'])==24
    assert long['source_result_sha256']==next(p['sha256'] for p in proofs if p['job']=='b8childgn' and p['file']=='result.json')
    for v in long['rows']:
        original=[x for x in rows if x['job']=='b8childgn' and x['case']==v['case'] and x['time']==v['time'] and x['layout']==v['layout']]
        assert len(original)==1 and original[0]['independent_error']==v['initial_independent_error']
        assert np.isfinite(v['final']['independent_error']) and v['additional_iterations']<=1000
    before=max(v['initial_independent_error'] for v in long['rows']);after=max(v['final']['independent_error'] for v in long['rows'])
    extra_table='\n'.join(f"| {v['case']} | {v['layout']} | {100*v['initial_independent_error']:.4f}% | {100*v['final']['independent_error']:.4f}% | {'Yes' if v['converged'] else 'No'} |" for v in long['rows'] if v['time']==0)
    supplement=f'''
## Retained long-fit follow-up: separate24-field subset

The original72-fit panel and its unfinished counts above are unchanged. A separate continuation kept its selected weights fixed and jointly refitted parent/child codes for24 selected field/layout combinations, allowing up to1000additional LM iterations under the same convergence criteria. **{long['converged']}/24 converged; {long['unfinished']} remained unfinished**, with{long['failed']} numerical failures. The subset's worst independent error moved from **{100*before:.4f}% to{100*after:.4f}%**.

| t=0 case | Allocation | Before | After long fit | Converged |
|---|---|---:|---:|---|
{extra_table}

Several converged fits remain above the0.5% representation threshold, including the aligned narrow pulse, two-pulse case and one ridge fit. The original iteration limit alone is therefore not the main explanation for these remaining errors. This is not an absolute capacity proof: other local minima or a restricted nonlinear head remain possible, and five fits are still unfinished. The matched feature-span discriminator is the next test of feature capacity versus nonlinear-code restriction. Independent256 scoring remains offline; there is no actual4096 or PDE claim.

Continuation job{long['meta']['job']}; verified result hash `{proof['sha256']}`. Its complete scalar records are retained alongside this report's metadata.
'''
    longfit_summary=dict(provenance=proof,rows=long['rows'],converged=long['converged'],unfinished=long['unfinished'],failed=long['failed'],worst_before=before,worst_after=after)
metadata=dict(provenance=proofs,rows=rows,summary=stats,longfit_followup=longfit_summary,family_mapping=families,plot_sha256=sha(plot),
    fit_grid=128,independent_grid=256,current_time_reference_normalization=True,actual4096=False,pde_trajectory=False,
    paired_child_gain=False,representation_admitted=False,final_test_opened=False)
mp=assets/'representation_metadata.json';mp.write_text(json.dumps(metadata,indent=2)+'\n')
table='\n'.join(f"| {v['model']} | {v['family'].replace('_',' ')} | {100*v['worst']:.3f}% | {100*v['median']:.3f}% | {v['unfinished']}/{v['fields']} | {v['at_or_below_half_percent']}/{v['fields']} |" for v in stats)
tree=records['b8childgn'];parent=records['b8parent32']
assert tree['selected_update']==30000
tree_unfinished=sum(not v['converged'] for v in rows if v['job']=='b8childgn')
parent_unfinished=sum(not v['converged'] for v in rows if v['job']=='b8parent32')
text=f'''# Expanded neural representation: first retained comparison

The selected30k-update B7-root nonlinear tree has a worst independent reconstruction error of **{100*tree['worst_subset_error']:.3f}%**, with **{tree_unfinished}/72 fits unfinished**. The separately trained expanded global parent32 has worst error **{100*parent['worst_independent_error']:.3f}%**, with **{parent_unfinished}/36 fits unfinished**. Neither establishes representation qualification. Unfinished fits are not evidence of a decoder capacity floor.

![Representation errors by family]({assets.name}/{plot.name})

These are offline field-fitting diagnostics on12 development trajectories at0,0.125,0.25:36 fields, with two tree layouts producing72 tree fits. Fitting uses128²Gauss quadrature and independent scoring256²Gauss, normalized by each current-time reference norm. Red crosses identify unfinished fits under unchanged stationarity/update criteria. The0.5%line is the representation threshold; crossing it alone does not satisfy convergence or all-case gates. Lines connect separately fitted snapshots, not an evolved PDE trajectory. **These are not actual4096 reconstruction plots.**

| Model / allocation | Family | Worst error | Median error | Unfinished | Error ≤0.5% |
|---|---|---:|---:|---:|---:|
{table}

The first two rows of panels share one local checkpoint and compare uniform versus prescribed allocation. The bottom row is a separate expanded-data global parent, selected at update{parent['selected_update']} and scored after its training-code refresh. The tree retains the original B7 parent and independently trained local corrections. These are different trained models; their difference must not be presented as paired child improvement over the expanded parent. Family assignments come from retained reference records, not case-number parity.

{supplement}

**Decision:** retain all failures, use matched feature-span evidence to distinguish representation limits from nonlinear fitting, and evaluate fresh children trained on the expanded parent separately when that run completes. No final-test fields were opened; no CROM/SNF accuracy or runtime superiority is established here.

The earlier smooth-case actual4096 PDE fields, error plots, runtime accounting and optimized-replay postmortem remain available in [the neural-tree PDE diagnostic](2026-11-19-neural-tree-pde-diagnostic.md). Its successful single smooth case does not qualify these localized cases.

## Provenance

Tree job{tree['meta']['job']}, selected checkpoint `{tree['selected_sha256']}`; expanded parent job{parent['meta']['job']}, checkpoint `{parent['selected_sha256']}`. Source records, checkpoints and expanded-reference family mapping were checked against COMPLETED local/remote retention manifests before reading. Full scalar plot inputs and hashes are retained in [{mp.name}]({assets.name}/{mp.name}). This report only reads retained metadata; it performs no model decoding, fitting or PDE solve.
'''
report=H/'reports'/(name+'.md');report.write_text(text)
published=[]
for src in [plot,mp,report]:
    dest=CAN/src.relative_to(H);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);assert sha(src)==sha(dest)
    published.append(dict(source=str(src),published=str(dest),sha256=sha(src)))
(H/'manifests/B8-representation-publication.json').write_text(json.dumps(dict(provenance=proofs,files=published),indent=2)+'\n')
print(json.dumps(dict(report=str(CAN/report.relative_to(H)),plot=str(CAN/plot.relative_to(H)),summary=stats)))
