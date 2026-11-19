"""Retained all-development fixed-tree gates; no runtime or final-test claims."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
from audit_fixed_pilot import inspect
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reg=json.loads((H/'manifests/B4-registration.json').read_text())
labels=[f'dev{i:03d}' for i in range(32)]+[f'align{i:03d}' for i in range(12)]
names=['b4base','b4time','b4quad'];records=[];raw={};hashes={};failed_jobs=[]
for name in names:
    mp=H/'manifests'/f'retention-{name}.json';rp=H/'results'/name/'output/result.json'
    if not mp.exists():continue
    retention=json.loads(mp.read_text());assert retention['local_and_remote_verified']
    if retention['terminal_state']!='COMPLETED':
        failed_jobs.append(name);hashes[name]=sha(mp);continue
    audit=inspect(name);r=json.loads(rp.read_text())
    setting=dict(L=16384,dt=.000625 if name=='b4time' else .00125,
        quadrature_intervals=1024 if name=='b4quad' else 512,stationarity_tolerance=1e-8,
        update_tolerance=1e-10,cg_relative_tolerance=1e-6,cg_absolute_tolerance=1e-12,
        initial_relative_tolerance=1e-9,initial_absolute_tolerance=1e-13)
    matched=all(r[k]==v for k,v in setting.items())
    matched &= (r['labels']==labels and r['global_control']=='global_stable960' and r['time_integrator']=='cn'
        and r['field_implementation']=='packed full-axis separable contractions'
        and r.get('normal_matrix_implementation','formed inside each step')=='formed inside each step'
        and r['saved_grid_fields']=='prediction only; qualified reference coefficients retained by hash'
        and len(r['cases'])==44 and {c['case'] for c in r['cases']}==set(labels)
        and all(c['common_tests']==22016 for c in r['cases']))
    if name!='b4base':matched &= r['previous_result_sha256']==sha(H/'results/b4base/output/result.json')
    else:matched &= 'previous_result_sha256' not in r
    checked=0
    for case in r['cases']:
        assert set(case['methods'])=={'tree_direct24','global_stable960'}
        assert len(case['reference_coefficients_sha256'])==64
        for method in case['methods']:
            file=f"output/{case['case']}_{method}_4096.npz"
            if not any(v['file']==file for v in retention['files']):
                matched=False
                continue
            # Read only the saved prediction's header. Retention already verifies
            # every byte; the header verifies actual array shape/precision.
            with zipfile.ZipFile(H/'results'/name/file) as archive:
                with archive.open('prediction.npy') as f:
                    assert np.lib.format.read_magic(f)==(1,0)
                    shape,order,dtype=np.lib.format.read_array_header_1_0(f)
                    matched &= shape==(4096,4096) and dtype==np.dtype('float64')
            checked+=1
    audit.update(registered_setting_matches=bool(matched),actual_prediction_headers_checked=checked)
    audit['passed'] &= bool(matched)
    records.append(audit);raw[name]=r;hashes[name]=audit['source_sha256']
ledger=json.loads((H/'cluster/ledger.json').read_text())
terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
active=[r for r in ledger if (r.get('state') or 'PENDING').split()[0] not in terminal]
seconds=sum(r.get('elapsed_seconds',0) for r in ledger)
charged=sum(r.get('elapsed_seconds',0) if (r.get('state') or 'PENDING').split()[0] in terminal else r['reserved_minutes']*60 for r in ledger)
failed=[r['name'] for r in records if not r['passed']]+failed_jobs
remaining=[n for n in names if n not in {r['name'] for r in records} and n not in failed_jobs]
if failed:decision='All-development qualification fails or is incomplete for '+', '.join(failed)+'. Diagnose the retained evidence before advancement.'
elif remaining:decision='All-development qualification remains incomplete. Outstanding: '+', '.join(remaining)+'.'
else:decision='All 44 development trajectories pass physical accuracy, full solver convergence, time-step halving and doubled integration for both methods. Fixed-tree dynamics are admitted; deployable adaptation and complete-runtime confirmation remain separate stages.'
table=[]
for audit in records:
    for method in ['tree_direct24','global_stable960']:
        rows=[v for v in audit['rows'] if v['method']==method]
        e=np.array([v['max_physical_error'] for v in rows]);allow=np.array([v['reference_allowance'] for v in rows])
        change=max((v['max_field_change'] or 0) for v in rows)
        table.append(f"|{audit['name']}|{method}|{np.median(e)*100:.6f}%|{np.quantile(e,.9)*100:.6f}%|{max(e)*100:.6f}%|{max(e+allow)*100:.6f}%|{sum(v['nonconverged_steps'] for v in rows)}|{change*100:.6f}%|{audit['passed']}|")
figures=[]
if 'b4base' in raw:
    by={c['case']:c for c in raw['b4base']['cases']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'burgers-tree-embeddings'})
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for method,color,label in [('tree_direct24','#3679b6','Direct tree (1,376 variables)'),('global_stable960','#d58425','Global (960 variables)')]:
        values=[by[v]['methods'][method] for v in labels]
        error=[100*(max(v['physical_error'][1:]+[v.get('actual4096_final_error',np.inf)])+v['reference_allowance_max']) for v in values]
        axes[0].plot(np.arange(44),error,'.-',color=color,label=label)
        axes[1].plot(np.arange(44),[100*max(v['projection_error'][1:]) for v in values],'.-',color=color,label=label+' projection')
        axes[1].plot(np.arange(44),[100*max(v['distance_from_best_projection'][1:]) for v in values],'.--',color=color,label=label+' distance from projection')
    axes[0].axhline(1,color='#b5413c',ls='--',label='1% physical limit')
    for ax in axes:ax.axvline(31.5,color='#999999',lw=.7);ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
    axes[0].set(ylabel='Worst physical error + allowance (%)',title='All-development fixed-tree PDE trajectories')
    axes[1].set(ylabel='Worst evolved relative distance (%)',xlabel='Trajectory index: 32 random development, then12 alignment')
    for ext in ['png','svg']:
        p=H/'reports'/f'b4-all-development-physics.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight',**({'metadata':{'Date':None}} if ext=='svg' else {}))
        if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    plt.close(fig);figures.append('![All-development physical and projection errors](b4-all-development-physics.png)')
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
text=f'''# Burgers tree embeddings: all-development fixed-tree qualification

{decision}

The same frozen direct24/52-node tree and stable global960 POD-plus-Fourier control use the initial-field callable, fine L16384 sign-upwind weak operator, Crank–Nicolson time integration,22016 tests and independently admitted packed field evaluation. The baseline uses step0.00125 and Q512; independent refinements halve the step and double Q. References enter only scoring after both solves. Final-test solution fields remain unopened.

Every physical decision covers all five evolved output times and the actual4096 final field, normalized by the reference norm at that same time and including its qualified allowance. Complete runs must save88 actual4096 float64 predictions, whose headers and full retained hashes are checked; reference coefficients are retained by hash instead of duplicating the same reference grid. Initial-fit error is recorded separately. Missing outputs fail admission. No incomplete step or trajectory is silently omitted.

|Run|Method|Median case error|90th percentile|Worst physical error|Worst error + allowance|Nonconverged steps|Largest refinement change|All gates pass|
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table) or '|No completed retained run yet|||||||||'}

{chr(10).join(figures)}

These are physical/numerical qualification runs, not paired speed benchmarks. Query clocks still charge shared comparison setup; no method-specific speed ratio is inferred. The nonlinear PDE checkpoint remains stopped after two failed optimizer repairs; the frozen tangent simplification also failed its representation screen. The cell catalogue has only CPU fixture evidence and is not used here. Native NM-ROM confirmation, deployable adaptation, complete-runtime benchmarking, method freeze and final-test evaluation remain outstanding.

Latest scheduler snapshot: {seconds} allocated GPU-seconds ({seconds/3600:.6f}h); charge including full active reservations {charged}, cumulative cap {reg['cumulative_ceiling_gpu_seconds']}. Active owned jobs: {', '.join(v['name'] for v in active) or 'none'}. Generated {stamp}. The old campaign remains paused; no automatic merge or unrelated3D work.

## Glossary

- **Case error:** worst current-time relative field error over that trajectory's required outputs. Median and90th percentile summarize the44 case errors, not correlated snapshots.
- **Allowance:** measured reference uncertainty, added before checking the1% physical gate.
- **Projection error:** diagnostic distance from the reference to its best-fit frozen representation; tiny regularization is retained from the diagnostic fit.
- **Distance from projection:** distance between the evolved state and that fitted field. It is reported separately from total physical error, not added as an assumed error bound.
- **Refinement change:** returned-state difference after independently changing one numerical setting, normalized by the current reference norm. The limit is0.1%; baseline has no comparison.
- **Complete convergence:** finite diagnostics, stationarity within tolerance and either a small field update or100 times smaller stationarity.
- **POD/Fourier:** a training-derived linear basis plus a prescribed trigonometric completion; this global control is not a native neural NM-ROM reproduction.
- **L/Q:** fine spatial-operator intervals and integration intervals per axis, respectively. Integration retains the fine stencil spacing.
- **Allocated GPU time:** scheduler duration including setup, compilation, scoring, storage and failed jobs; it differs from online query time.
'''
name='2026-11-19-tree-embedding-development-physics.md';(H/'reports'/name).write_text(text)
main=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');assets=main/'2026-11-18-tree-embedding-representation-assets'
for p in (H/'reports').glob('b4-all-development-*'):shutil.copy2(p,assets/p.name)
(main/name).write_text(text.replace('](b4-all-development-','](2026-11-18-tree-embedding-representation-assets/b4-all-development-'))
for folder in [H/'reports',main]:
    path=folder/'2026-11-18-tree-embedding-representation.md'
    lines=path.read_text().splitlines()
    current=f'Current PDE development qualification: [all-development checks]({name}). The retained [three-case numerical checks](2026-11-18-tree-embedding-fixed-tree.md) precede this panel.'
    index=next((i for i,line in enumerate(lines) if line.startswith(('Current fixed-tree qualification:','Current PDE development qualification:'))),None)
    if index is None:lines[2:2]=[current,'']
    else:lines[index]=current
    extra='Further decoder diagnosis: [the frozen tangent comparison](2026-11-19-tree-embedding-implementation-tangent.md).'
    if extra not in lines:lines[4:4]=[extra,'']
    path.write_text('\n'.join(lines)+'\n')
status=f'''**New Burgers tree-embedding study: B4 all-development fixed-tree qualification.** {decision} Latest recorded allocation {seconds/3600:.6f}h; active jobs: {', '.join(v['name'] for v in active) or 'none'}. No runtime win, final-test opening or method freeze. [All-development evidence](reports/{name}); [representation plots](reports/2026-11-18-tree-embedding-representation.md). Old campaign remains PAUSED; nonlinear optimizer and tangent ablation remain stopped; no merge.'''
record=dict(utc=stamp,decision=decision,source_hashes=hashes,records=records,failed=failed,remaining=remaining,
            elapsed_gpu_seconds=seconds,charged_gpu_seconds=charged,active_jobs=[v['name'] for v in active],own_status=status)
key=hashlib.sha256(json.dumps(dict(sources=hashes,failed=failed,active=record['active_jobs']),sort_keys=True).encode()).hexdigest()[:12]
(H/'manifests'/f'B4-progress-{key}.json').write_text(json.dumps(record,indent=2)+'\n')
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');contents=lab.read_text();start=contents.index('**New Burgers tree-embedding study:');end=contents.index('\n\n',start)
contents=contents[:start]+status+contents[end:];title=f'### 2026-11-19 — Tree-embedding B4 qualification record {key}'
if title not in contents:contents=contents.rstrip()+f'\n\n{title}\n\n{status}\n\nReturned-step, actual-grid and sensitivity audits: `manifests/B4-progress-{key}.json` in the owned lane.\n'
lab.write_text(contents)
owner=json.loads((H/'OWNER.json').read_text());owner.update(updated_utc=stamp,status='B4_ALL_DEVELOPMENT_QUALIFICATION',decision=decision,
    allocated_gpu_seconds=seconds,allocated_gpu_hours=seconds/3600,active_jobs=len(active),terminal_jobs=len(ledger)-len(active),
    completed_jobs=sum(v.get('state')=='COMPLETED' for v in ledger),batch_gpu_seconds=reg['max_additional_gpu_seconds'],cumulative_ceiling_gpu_seconds=reg['cumulative_ceiling_gpu_seconds'])
(H/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n');claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner'];claim.write_text(json.dumps(owner,indent=2)+'\n')
print(decision)
