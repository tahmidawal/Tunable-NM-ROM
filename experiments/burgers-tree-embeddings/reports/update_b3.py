"""Publish retained B3 diagnostics without promoting them to timing confirmation."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
from audit_fixed_pilot import inspect
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def retained(name):
    rp=H/'results'/name/'output/result.json';mp=H/'manifests'/f'retention-{name}.json'
    if not rp.exists() or not mp.exists():return None
    m=json.loads(mp.read_text())
    assert m['local_and_remote_verified']
    if m['terminal_state']!='COMPLETED':return dict(failed_job=m['terminal_state'])
    assert next(v['sha256'] for v in m['files'] if v['file']=='output/result.json')==sha(rp)
    return json.loads(rp.read_text())

reg=json.loads((H/'manifests/B3-registration.json').read_text())
ledger=json.loads((H/'cluster/ledger.json').read_text())
terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
active=[r for r in ledger if (r.get('state') or 'PENDING').split()[0] not in terminal]
seconds=sum(r.get('elapsed_seconds',0) for r in ledger)
charge=sum(r.get('elapsed_seconds',0) if (r.get('state') or 'PENDING').split()[0] in terminal else r['reserved_minutes']*60 for r in ledger)
sources={};records={};parts=[];decisions=[]
packed=retained('b3packed')
if packed and 'failed_job' not in packed:
    audit=inspect('b3packed')
    expected=dict(L=16384,dt=.00125,quadrature_intervals=512,stationarity_tolerance=1e-8,
                  update_tolerance=1e-10,cg_relative_tolerance=1e-6,cg_absolute_tolerance=1e-12,
                  initial_relative_tolerance=1e-9,initial_absolute_tolerance=1e-13)
    settings=all(packed[k]==v for k,v in expected.items())
    settings &= (packed['labels']==['dev000','dev003','align002'] and packed['global_control']=='global_stable960'
                 and packed['time_integrator']=='cn' and packed['field_implementation']=='packed full-axis separable contractions'
                 and packed['comparison_kind']=='equivalent_field_implementation'
                 and packed['previous_result_sha256']==sha(H/'results/b2space/output/result.json')
                 and all(c['common_tests']==22016 for c in packed['cases']))
    change=max(v['max_field_change'] for v in audit['rows'])
    passed=audit['passed'] and settings and change<=1e-7
    audit.update(equivalent_field_gate=bool(passed),equivalence_limit=1e-7,registered_setting_matches=bool(settings))
    records['packed']=audit;sources['b3packed']=audit['source_sha256']
    decisions.append('Packed field equivalence '+('passes' if passed else 'fails')+f' (largest change{100*change:.9g}%).')
    rows=[]
    for c in packed['cases']:
        for method,m in c['methods'].items():
            rows.append(f"|{c['case']}|{method}|{100*max(m['previous_state_relative_change']):.9g}%|{m['nonconverged_steps']}|{m['packed_field_cache_bytes']/2**20:.3f}|{m['cold_start_including_compilation_seconds']:.3f}|{m['diagnostic_total_query_seconds']:.3f}|")
    parts.append('''## Equivalent field implementation

These are the unchanged decoder and fine weak PDE equations evaluated with fixed array shapes and runtime tree geometry. Padded arithmetic and storage are charged. The tolerance for field equivalence is0.00001%; physical accuracy and complete solver convergence are checked separately. These same-job component clocks **are diagnostic only**: each method is charged joint model setup, and cold initialization includes compilation. They do not establish a paired speed ratio or a cost improvement over the earlier job.

|Case|Method|Largest field change|Nonconverged steps|Padded field cache MiB|Cold initialization seconds|Diagnostic query seconds|
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows))
elif packed:
    records['packed']=packed;decisions.append('Packed diagnostic job failed; inspect retained failure evidence before any repair.')
else:decisions.append('Packed field diagnostic is outstanding.')

tangent=retained('b3tangent')
if tangent and 'failed_job' not in tangent:
    assert tangent['status']=='complete_frozen_tangent_screen'
    assert tangent['nodes']==64 and tangent['latent']==16 and tangent['active_variables']==1152
    assert not tangent['head_weights_changed'] and not tangent['PDE_solve'] and not tangent['final_test_opened']
    records['tangent']=dict(representation_gate=tangent['representation_gate'],projection=tangent['projection'],
                            actual4096_witness=tangent['actual4096_witness'],decision=tangent['decision'])
    sources['b3tangent']=sha(H/'results/b3tangent/output/result.json')
    oldpath=H/'results/b1childgrowth64/output/result.json'
    old=json.loads(oldpath.read_text());assert tangent['growth_sha256']==sha(oldpath)
    assert old['model_sha256']==tangent['model_sha256']
    labels=[v['case'] for v in tangent['cases']];assert len(labels)==44
    previous={v['case']:max(v['error'])*100 for v in old['cases']}
    now=[max(v['error'])*100 for v in tangent['cases']]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'burgers-tree-embeddings'})
    fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
    ax.plot(np.arange(44),[previous[v] for v in labels],'.-',label='Frozen nonlinear child head',color='#3679b6')
    ax.plot(np.arange(44),now,'.-',label='Frozen zero-code tangent (linear)',color='#d58425')
    ax.axhline(.5,color='#b5413c',ls='--',label='0.5% representation screen')
    ax.axvline(31.5,color='#999999',lw=.7)
    ax.set(xlabel='Development trajectory index (32 random, then12 alignment)',ylabel='Worst of11 snapshot errors (%)',
           title='Same frozen weights, tree and1,152 variables: does curvature matter?')
    ax.legend(fontsize=8)
    for ext in ['png','svg']:
        dest=H/'reports'/f'b3-frozen-tangent.{ext}'
        fig.savefig(dest,dpi=180,bbox_inches='tight',**({'metadata':{'Date':None}} if ext=='svg' else {}))
        if ext=='svg':dest.write_text('\n'.join(s.rstrip() for s in dest.read_text().splitlines())+'\n')
    plt.close(fig)
    err=max(now);station=max(max(v['stationarity']) for v in tangent['cases'])
    decisions.append(tangent['decision']+'.')
    parts.append(f'''## Frozen tangent architecture

This is a separately labelled direct-coefficient decoder: the derivative of each frozen child head at zero replaces its nonlinear output. The original nonlinear PDE optimizer remains stopped. Parent, shared weights,64 retained nodes and1152 adjustable variables are unchanged; all coefficients are jointly refitted with the complete physical overlap metric and no ridge/direction removal. Geometry is selected from the initial-field callable before reference access.

Worst physical projection error is{err:.8f}%; the actual4096 witness is{100*tangent['actual4096_witness']['error']:.8f}% at{tangent['actual4096_witness']['case']}, t={tangent['actual4096_witness']['time']}. Maximum scaled stationarity is{station:.6g}. The representation gate is **{tangent['representation_gate']}**. An unsuccessful screen is not admitted to PDE solving; a passing screen also requires the retained independent all-snapshot dense audit.

![Frozen nonlinear head and its tangent](b3-frozen-tangent.png)
''')
elif tangent:
    records['tangent']=tangent;decisions.append('Tangent diagnostic job failed; preserve its failure evidence.')
else:decisions.append('Frozen tangent representation screen is outstanding.')

stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
decision=' '.join(decisions)
text=f'''# Burgers tree embeddings: field implementation and frozen tangent diagnostics

{decision}

The preceding three-case B2 setting passes physical accuracy and independent spatial, integration, weak-test, time-step and100x-precision checks. This batch tests one equivalent implementation and one specific architecture, under a finite one-GPU-hour allowance. It does not admit all-development PDE trajectories, deployable adaptation, a runtime win, native NM-ROM confirmation or final-test access. The original16.5% failure and all later representation results remain in [the representation report](2026-11-18-tree-embedding-representation.md).

{chr(10).join(parts)}

Latest scheduler-ledger snapshot: {seconds} elapsed GPU-seconds ({seconds/3600:.6f}h); full outstanding reservations bring the charge to{charge} seconds against a cumulative ceiling of{reg['cumulative_ceiling_gpu_seconds']}. Active owned jobs: {', '.join(r['name'] for r in active) or 'none'}. Generated {stamp}. Raw outputs and logs are retained with verified hashes before admission. Final-test solution fields remain unopened and the earlier campaign remains paused.

## Glossary

- **Tangent:** the decoder's derivative at zero child codes, used here as a fixed linear correction basis.
- **Curvature:** the change in a nonlinear decoder's directions as its codes move away from zero.
- **Projection:** an offline best-fit reconstruction that may use the reference field; it is not a PDE solve.
- **Stationarity:** the remaining physical objective gradient after fitting. Small values help distinguish limited representation from a poor fit.
- **Equivalence:** the returned field agrees with the unchanged compact implementation to the registered tolerance.
- **Cold initialization:** constructing/fitting the initial state, including compilation in this diagnostic.
- **MiB:**2^20 bytes. Padded cache memory includes zeros introduced to make array shapes independent of tree geometry.
- **Allocated time:** scheduler duration, including compilation, failures and idle portions; it is distinct from online query runtime.
'''
name='2026-11-19-tree-embedding-implementation-tangent.md'
(H/'reports'/name).write_text(text)
main=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');assets=main/'2026-11-18-tree-embedding-representation-assets'
for p in (H/'reports').glob('b3-frozen-tangent.*'):shutil.copy2(p,assets/p.name)
(main/name).write_text(text.replace('](b3-frozen-tangent.png)','](2026-11-18-tree-embedding-representation-assets/b3-frozen-tangent.png)'))
status=f'''**New Burgers tree-embedding study: B3 implementation and tangent diagnostics.** {decision} B2 admits only the three-case numerical setting for direct1376/global960. Latest recorded allocation{seconds/3600:.6f}h; active jobs: {', '.join(r['name'] for r in active) or 'none'}. No all-development PDE admission, runtime win or final-test opening. [Representation plots](reports/2026-11-18-tree-embedding-representation.md); [new diagnostics](reports/{name}). Old campaign remains PAUSED; nonlinear optimizer remains stopped; no merge.'''
key=hashlib.sha256(json.dumps(dict(sources=sources,active=[r['name'] for r in active]),sort_keys=True).encode()).hexdigest()[:12]
record=dict(utc=stamp,decision=decision,source_hashes=sources,records=records,elapsed_gpu_seconds=seconds,
            charged_gpu_seconds=charge,active_jobs=[r['name'] for r in active],own_status=status)
(H/'manifests'/f'B3-progress-{key}.json').write_text(json.dumps(record,indent=2)+'\n')
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');contents=lab.read_text()
start=contents.index('**New Burgers tree-embedding study:');end=contents.index('\n\n',start)
contents=contents[:start]+status+contents[end:]
title=f'### 2026-11-19 — Tree-embedding B3 diagnostic record {key}'
if title not in contents:contents=contents.rstrip()+f'\n\n{title}\n\n{status}\n\nSource hashes and gates: `manifests/B3-progress-{key}.json` in the owned lane.\n'
lab.write_text(contents)
owner=json.loads((H/'OWNER.json').read_text())
owner.update(updated_utc=stamp,status='B3_IMPLEMENTATION_AND_TANGENT',decision=decision,
             allocated_gpu_seconds=seconds,allocated_gpu_hours=seconds/3600,active_jobs=len(active),
             terminal_jobs=len(ledger)-len(active),completed_jobs=sum(r.get('state')=='COMPLETED' for r in ledger))
(H/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner'];claim.write_text(json.dumps(owner,indent=2)+'\n')
print(decision)
