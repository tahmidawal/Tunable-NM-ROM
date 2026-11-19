"""Retained method-specific costs; this is not native NM-ROM confirmation."""
from pathlib import Path
import datetime,hashlib,itertools,json,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
registration=json.loads((H/'manifests/B5-registration.json').read_text())
names=['b5cache','b5normal','b5compact','b5block0','b5block1']
methods=['tree_direct24','global_stable960','tree_packed_control']
labels=['dev000','dev003','align002']
raw={};audits={};sources={};failed=[]
for name in names:
    mp=H/'manifests'/f'retention-{name}.json'
    if not mp.exists():continue
    retained=json.loads(mp.read_text());assert retained['local_and_remote_verified']
    sources[name]=sha(mp)
    if retained['terminal_state']!='COMPLETED':failed.append(name);continue
    p=H/'results'/name/'output/result.json';r=json.loads(p.read_text())
    assert sha(p)==next(v['sha256'] for v in retained['files'] if v['file']=='output/result.json')
    paired=name.startswith('b5block');three=name in ['b5compact','b5block0','b5block1']
    expected_methods=methods if three else methods[:2]
    repeats=5 if paired else 1;contracts=[1,5] if paired else [1]
    expected={(case,method,rep,contract) for case in labels for method in expected_methods
              for rep in range(repeats) for contract in contracts}
    keys=[(v['case'],v['method'],v['repetition'],v['output_contract']) for v in r['invocations']]
    complete=(len(keys)==len(expected) and set(keys)==expected and r['cases']==labels)
    settings=(r['config']==dict(L=16384,Q=512,tests=22016,dt=.00125)
        and r['normal_reuse']==(name!='b5cache')
        and r['field_kernel']==('compact' if three else 'packed')
        and r['include_packed_control']==three and r['final_test_opened'] is False
        and r['status']==('complete_paired_query_diagnostic' if paired else 'complete_query_equivalence_diagnostic')
        and r['meta']['x64'] and r['meta']['precision']=='highest')
    if paired:
        qualifier=raw.get('b5compact');settings &= qualifier is not None
        if qualifier:settings &= r['implementation_sha256']==qualifier['implementation_sha256']
    rows=[];headers={}
    for v in r['invocations']:
        rec=v['record'];states=np.asarray(rec.get('step_status',[]))
        converged=(rec.get('complete',False) and states.shape==(200,6) and np.all(np.isfinite(states))
            and np.all(states[:,1]<=1e-8) and np.all((states[:,2]<=1e-10)|(states[:,1]<=1e-10))
            and rec.get('initial_linear_residual',np.inf)<=1e-9)
        errors=rec.get('physical_error',[]);actual=rec.get('actual4096_errors',[])
        physical=(len(errors)==6 and len(actual)==v['output_contract']
            and max(errors[1:]+actual)+rec['reference_allowance_max']<=.01)
        changes=rec.get('previous_state_relative_change',[])
        equivalent=len(changes)==6 and max(changes)<=1e-7
        fields=rec.get('actual_fields',[]);output_ok=len(fields)==v['output_contract']
        for field in fields:
            rel='output/'+field['file'];f=H/'results'/name/rel
            evidence=next((x for x in retained['files'] if x['file']==rel),None)
            output_ok &= (evidence is not None and evidence['sha256']==field['stored_file_sha256']
                and field['shape']==[4096,4096] and field['dtype']=='float64')
            if evidence and rel not in headers:
                with zipfile.ZipFile(f) as archive:
                    with archive.open('prediction.npy') as stream:
                        assert np.lib.format.read_magic(stream)==(1,0)
                        shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                        headers[rel]=shape==(4096,4096) and dtype==np.dtype('float64')
            output_ok &= headers.get(rel,False)
        seconds=rec.get('total_query_seconds',np.nan)
        timed=np.isfinite(seconds) and seconds>0 and rec.get('external_total_timer',False)
        if paired:timed &= (rec.get('component_instrumentation') is False and
                           rec.get('internal_instrumented_timer_seconds') is None)
        rows.append(dict(case=v['case'],method=v['method'],repetition=v['repetition'],outputs=v['output_contract'],
            seconds=seconds,passed=bool(converged and physical and equivalent and output_ok and timed),
            solver_pass=bool(converged),physical_pass=bool(physical),equivalence_pass=bool(equivalent),
            actual_output_pass=bool(output_ok),timing_boundary_pass=bool(timed),
            max_error_with_allowance=max(errors[1:]+actual)+rec['reference_allowance_max'] if len(errors)>1 and actual else None,
            max_equivalence_change=max(changes) if changes else None))
    passed=bool(complete and settings and all(v['passed'] for v in rows) and r['all_passed'])
    audits[name]=dict(passed=passed,registered_setting_matches=bool(settings),complete=complete,rows=rows,
        prediction_headers_checked=len(headers),result_sha256=sha(p))
    raw[name]=r
    if not passed:failed.append(name)

pair_rows=[];order_pass=None
if all(name in raw for name in ['b5block0','b5block1']):
    # Validate the exact paired order reversal, not only the intended seed.
    def orders(r):return {(v['case'],v['output_contract'],v['repetition']):v['method_order'] for v in r['invocations']}
    o0,o1=orders(raw['b5block0']),orders(raw['b5block1'])
    order_pass=set(o0)==set(o1) and all(o0[k]==list(reversed(o1[k])) for k in o0)
    if not order_pass:failed.append('paired_order')
    for block in ['b5block0','b5block1']:
        rows=audits[block]['rows'];lookup={(v['case'],v['method'],v['repetition'],v['outputs']):v for v in rows}
        for case in labels:
            for outputs in [1,5]:
                for method in ['tree_direct24','tree_packed_control']:
                    ratios=[]
                    for repetition in range(5):
                        tree=lookup[(case,method,repetition,outputs)];glob=lookup[(case,'global_stable960',repetition,outputs)]
                        ratios.append(tree['seconds']/glob['seconds'])
                    pair_rows.append(dict(block=block,case=case,outputs=outputs,method=method,
                        paired_ratios=ratios,median=float(np.median(ratios))))

summary=[]
if pair_rows and not failed:
    for outputs in [1,5]:
        for method in ['tree_direct24','tree_packed_control']:
            case_ratios=[]
            for case in labels:
                ratios=[x for v in pair_rows if (v['case'],v['outputs'],v['method'])==(case,outputs,method) for x in v['paired_ratios']]
                case_ratios.append(float(np.median(ratios)))
            # Exact small-panel trajectory bootstrap:27 ordered resamples.
            bootstrap=[np.median([case_ratios[i] for i in indices]) for indices in itertools.product(range(3),repeat=3)]
            summary.append(dict(method=method,outputs=outputs,case_ratios=case_ratios,
                median=float(np.median(case_ratios)),bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
                worst=max(case_ratios),faster_cases=sum(v<1 for v in case_ratios)))

remaining=[v for v in names if v not in audits and v not in failed]
if failed:decision='A query qualification or retained timing check failed: '+', '.join(failed)+'. Preserve the failure and diagnose it before advancement.'
elif remaining:decision='Complete-query diagnostic remains in progress. Outstanding: '+', '.join(remaining)+'.'
else:
    primary=next(v for v in summary if v['method']=='tree_direct24' and v['outputs']==1)
    faster=primary['median']<1 and primary['bootstrap95'][1]<1
    decision=('The compact tree passes the three-case diagnostic cost criterion.' if faster else
              'The compact tree does not pass the three-case diagnostic cost criterion.')
    decision+=' This does not establish native NM-ROM superiority or a final-test result; review measured component costs before any finite next batch.'

figures=[]
if pair_rows:
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'burgers-tree-embeddings'})
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)
    for ax,outputs in zip(axes,[1,5]):
        for method,color,offset in [('tree_direct24','#247e83',-.12),('tree_packed_control','#ac703b',.12)]:
            for block,marker in [('b5block0','o'),('b5block1','s')]:
                rows=[next(v for v in pair_rows if (v['case'],v['outputs'],v['method'],v['block'])==(case,outputs,method,block)) for case in labels]
                x=np.arange(3)+offset+(-.025 if block=='b5block0' else .025)
                ax.scatter(x,[v['median'] for v in rows],marker=marker,color=color,
                    label=('Compact' if method=='tree_direct24' else 'Padded')+' / global, block'+block[-1])
        ax.axhline(1,color='#b5413c',ls='--');ax.set(xticks=np.arange(3),xticklabels=labels,title=f'{outputs} actual4096 device field'+('s' if outputs>1 else ''),yscale='log')
    axes[0].set_ylabel('Median paired complete-query ratio (tree/global)');axes[1].legend(fontsize=8)
    for ext in ['png','svg']:
        p=H/'reports'/f'b5-complete-query-cost.{ext}';fig.savefig(p,dpi=180,**({'metadata':{'Date':None}} if ext=='svg' else {}))
        if ext=='svg':p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n')
    plt.close(fig);figures=['![Complete-query cost ratios](b5-complete-query-cost.png)']

table=['|Run|Invocations|All gates pass|Distinct actual fields checked|','|---|---:|---|---:|']
for name,a in audits.items():table.append(f"|{name}|{len(a['rows'])}|{a['passed']}|{a['prediction_headers_checked']}|")
cost=['|Tree evaluation|Requested fields|Median ratio|Trajectory-bootstrap95% interval|Worst case ratio|Cases faster|',
      '|---|---:|---:|---|---:|---:|']
for s in summary:cost.append(f"|{s['method']}|{s['outputs']}|{s['median']:.6f}|[{s['bootstrap95'][0]:.6f}, {s['bootstrap95'][1]:.6f}]|{s['worst']:.6f}|{s['faster_cases']}/{len(labels)}|")
components=[]
instrumented=next((n for n in ['b5compact','b5normal','b5cache'] if n in raw),None)
if instrumented:
    components=[f'Instrumented diagnostic components from{instrumented}, in seconds. These single invocations locate work; the repeated uninstrumented measurements above determine cost comparisons.',
        '', '|Case|Method|Selection/assembly|Initial fit|PDE stepping|Requested output|Whole query|',
        '|---|---|---:|---:|---:|---:|---:|']
    for v in raw[instrumented]['invocations']:
        rec=v['record']
        if not rec.get('complete'):continue
        values=[rec[k] for k in ['setup_seconds','initial_fit_seconds','solving_seconds','requested_reconstruction_seconds','total_query_seconds']]
        components.append('|'+v['case']+'|'+v['method']+'|'+'|'.join(f'{x:.6f}' for x in values)+'|')
ledger=json.loads((H/'cluster/ledger.json').read_text());terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
active=[v for v in ledger if (v.get('state') or 'PENDING').split()[0] not in terminal]
seconds=sum(v.get('elapsed_seconds',0) for v in ledger)
charge=sum(v.get('elapsed_seconds',0) if v not in active else v['reserved_minutes']*60 for v in ledger)
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
text=f'''# Burgers tree embeddings: complete fixed-query cost diagnostic

{decision}

The direct24/52-node tree and stable global960 control use the admitted fine weak PDE, numerical tolerances and training data. Reusable tables cover all340 cells through depth4. Source verification, cache construction/storage and training-only prewarming are recorded separately. Every query charges initial sampling/selection, gathering, visibility, initial fitting, solving and actual requested reconstruction. Repeated timed queries disable internal stopwatches and synchronize all returned work. These fixed trees have no refinement trials or fallback; their zero counts do not predict adaptive cost.

{chr(10).join(table)}

{chr(10).join(cost) if summary else 'Paired cost results are not yet admitted.'}

{chr(10).join(figures)}

{chr(10).join(components)}

Ratios always compare invocations on the same GPU/allocation. The two allocations use opposite randomized method orders, with five repetitions per case and output contract. Accuracy and cost use the same returned states. One final4096-square device field is the primary contract; five4096-square fields use separate complete invocations. Host transfer is recorded separately. Each distinct float64 output is retained by raw-byte hash; exact repeats map to that retained array. No coefficient-only output substitutes for a requested field.

This panel contains only three development trajectories. The exact small-panel bootstrap resamples trajectories, not repetitions; its interval is a diagnostic with limited coverage, not final confirmation. The global control is POD plus prescribed Fourier completion, not a native Tunable/CROM/SNF/masked-autoencoder reproduction. Native competitors, deployable adaptation, method freeze and unopened final-test solutions remain separate requirements. An unqualified timing result is retained but cannot support a speed claim.

Latest scheduler snapshot: {seconds} allocated GPU-seconds ({seconds/3600:.6f}h), with full outstanding reservations charged to{charge} seconds against the cumulative cap{registration['cumulative_ceiling_gpu_seconds']}. Active owned jobs: {', '.join(v['name'] for v in active) or 'none'}. Generated {stamp}. The old campaign remains paused; no merge or unrelated3D work.

## Glossary

- **Complete query:** the known initial callable enters, and actual requested device fields leave; all query-specific work is timed.
- **Reusable setup:** fixed-decoder tables and training-only compilation that can serve previously unseen inputs; construction and storage remain reported.
- **Compact/padded:** local patch arithmetic versus full-axis contractions containing support zeros; both implement the same decoder.
- **Paired ratio:** tree query duration divided by the global query duration from the same case, repetition and allocation. Below1 is faster.
- **Trajectory bootstrap:** resampling whole cases to summarize uncertainty, without treating repeated measurements as independent cases.
- **Equivalence gate:** field change at most1e-7 of the current reference norm, alongside physical and full solver-convergence checks.
- **Allocated GPU time:** complete scheduler duration, including compilation, failed jobs, scoring and output storage.
'''
name='2026-11-19-tree-embedding-complete-query.md';(H/'reports'/name).write_text(text)
main=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');assets=main/'2026-11-18-tree-embedding-representation-assets'
for p in (H/'reports').glob('b5-complete-query-cost.*'):shutil.copy2(p,assets/p.name)
(main/name).write_text(text.replace('](b5-complete-query-cost.png)','](2026-11-18-tree-embedding-representation-assets/b5-complete-query-cost.png)'))
status=f'**New Burgers tree-embedding study: B5 complete-query diagnostic.** {decision} Latest recorded allocation{seconds/3600:.6f}h; active jobs: {", ".join(v["name"] for v in active) or "none"}. [Complete costs](reports/{name}); [representation plots](reports/2026-11-18-tree-embedding-representation.md). Final tests remain unopened and the old campaign paused; no merge.'
record=dict(utc=stamp,decision=decision,source_hashes=sources,audits=audits,paired_rows=pair_rows,summary=summary,
    order_reversal_pass=order_pass,failed=failed,remaining=remaining,elapsed_gpu_seconds=seconds,charged_gpu_seconds=charge,own_status=status)
key=hashlib.sha256(json.dumps(dict(sources=sources,failed=failed,active=[v['name'] for v in active]),sort_keys=True).encode()).hexdigest()[:12]
(H/'manifests'/f'B5-progress-{key}.json').write_text(json.dumps(record,indent=2)+'\n')
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');contents=lab.read_text()
start=contents.index('**New Burgers tree-embedding study:');end=contents.index('\n\n',start)
contents=contents[:start]+status+contents[end:];title=f'### 2026-11-19 — Tree-embedding B5 query record {key}'
if title not in contents:contents=contents.rstrip()+f'\n\n{title}\n\n{status}\n\nRetained gate and paired arrays: `manifests/B5-progress-{key}.json` in the owned lane.\n'
lab.write_text(contents)
owner=json.loads((H/'OWNER.json').read_text());owner.update(updated_utc=stamp,status='B5_COMPLETE_QUERY_DIAGNOSTIC',decision=decision,
    allocated_gpu_seconds=seconds,allocated_gpu_hours=seconds/3600,active_jobs=len(active),terminal_jobs=len(ledger)-len(active),
    completed_jobs=sum(v.get('state')=='COMPLETED' for v in ledger),batch_gpu_seconds=registration['max_additional_gpu_seconds'],
    cumulative_ceiling_gpu_seconds=registration['cumulative_ceiling_gpu_seconds'])
(H/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner'];claim.write_text(json.dumps(owner,indent=2)+'\n')
print(decision)
