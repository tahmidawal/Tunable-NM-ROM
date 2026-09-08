"""Generate complete-query findings and stopping diagnostics from native rows."""
import argparse,json,hashlib,statistics as st
from pathlib import Path
from collections import defaultdict,Counter

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run
payload=(run/'result.json').read_bytes();d=json.loads(payload);cfg=d['config'];audit=json.loads((run/'audit.json').read_text())
assert d['complete'] and audit['passed'];groups=defaultdict(list)
for r in d['rows']+d['stationary_rows']:
    for cohort in ('all_development',r['group']):
        groups[r['intervals'],cohort,r['panel'],r['model'],r['arm'],r['projection'],r['initialization']].append(r)
summaries=[]
for (n,cohort,panel,model,arm,projection,initialization),rows in groups.items():
    cases=[]
    for case in sorted({r['case'] for r in rows}):
        rs=sorted((r for r in rows if r['case']==case),key=lambda r:r['repetition'])
        assert len(rs)==(cfg['repetitions'] if panel=='primary' else 1)
        item=dict(case=case,raw_seconds=[r['total_seconds'] for r in rs],median_seconds=st.median(r['total_seconds'] for r in rs),
            physical_error=max(r['physical_error'] for r in rs),same_grid_error=max(r['same_grid_error'] for r in rs),
            adjusted_error=max(r['conservative_physical_error'] for r in rs),reference_delta=rs[0]['reference_delta'],
            solver_valid=all(r['solver_valid'] for r in rs),stationary=all(r['stationary'] for r in rs),
            parity_passed=all(r['parity_passed'] for r in rs),attempts=st.median(r['attempts'] for r in rs),
            reasons=[r['reason'] for r in rs],distinct_fields=len({r['field_sha256'] for r in rs}))
        if model:item.update(initial_residual=st.median(r['initial_residual'] for r in rs),final_residual=st.median(r['residual'] for r in rs),
            absolute_tau_threshold=st.median(r['absolute_tau_threshold'] for r in rs),stationarity=max(r['stationarity'] for r in rs),
            selected_indices=sorted({r['selected_training_code_index'] for r in rs}),near_tie=any(r['cache_near_tie'] for r in rs))
        cases.append(item)
    total=st.median(c['median_seconds'] for c in cases)
    parts=['input_seconds','fused_device_seconds','output_seconds'] if model else ['input_seconds','projection_init_seconds','solver_seconds','output_seconds']
    s=dict(intervals=n,cohort=cohort,panel=panel,model=model,arm=arm,projection=projection,initialization=initialization,
        case_count=len(cases),cases=cases,latency_seconds=total,physical_median=st.median(c['physical_error'] for c in cases),
        physical_max=max(c['physical_error'] for c in cases),same_grid_max=max(c['same_grid_error'] for c in cases),
        adjusted_max=max(c['adjusted_error'] for c in cases),invalid_count=sum(not c['solver_valid'] for c in cases),
        nonstationary_count=sum(not c['stationary'] for c in cases),parity_failure_count=sum(not c['parity_passed'] for c in cases),
        attempt_median=st.median(c['attempts'] for c in cases),reason_counts=dict(Counter(str(c['reasons'][0]) for c in cases)),
        timing_outliers=sum(r['total_seconds']>3*total for r in rows),fallback_count=sum(r.get('fallback_count',0) for r in rows),
        components_seconds={name:st.median(st.median(r[name] for r in rows if r['case']==c['case']) for c in cases) for name in parts},targets={})
    for eps in cfg['targets']:
        failed=[c['case'] for c in cases if not c['solver_valid'] or c['adjusted_error']>eps or c['reference_delta']>cfg['reference_fraction']*eps]
        s['targets'][str(eps)]=dict(qualified=not failed,failed_cases=failed)
    if model:s.update(initial_residual_median=st.median(c['initial_residual'] for c in cases),
        final_residual_median=st.median(c['final_residual'] for c in cases),absolute_tau_threshold_median=st.median(c['absolute_tau_threshold'] for c in cases))
    summaries.append(s)
ratios=lambda control,candidate:[c['median_seconds']/r['median_seconds'] for c,r in zip(control['cases'],candidate['cases'])]
for s in summaries:
    if s['panel']=='primary' and s['model']:
        dst=next(r for r in summaries if r['intervals']==s['intervals'] and r['cohort']==s['cohort'] and r['arm']=='dst')
        values=ratios(dst,s);s.update(same_grid_dst_case_cost_ratios=values,same_grid_dst_median_case_cost_ratio=st.median(values))
envelopes=[]
for n in cfg['intervals']:
    for cohort in ('all_development','existing_development','fresh_development'):
        for eps in cfg['targets']:
            valid=[s for s in summaries if s['intervals']==n and s['cohort']==cohort and s['panel']=='primary' and s['targets'][str(eps)]['qualified']]
            rom=min((s for s in valid if s['model']),key=lambda s:s['latency_seconds'],default=None)
            fom=min((s for s in valid if not s['model']),key=lambda s:s['latency_seconds'],default=None)
            compact=lambda s:None if s is None else {k:s[k] for k in ('model','arm','projection','initialization','latency_seconds','physical_max','adjusted_max')}
            pairs=ratios(fom,rom) if fom and rom else None
            envelopes.append(dict(intervals=n,cohort=cohort,target=eps,rom=compact(rom),fom=compact(fom),
                per_case_cost_ratios=pairs,median_case_cost_ratio=st.median(pairs) if pairs else None))
effects=[]
for n in cfg['intervals']:
    for model in cfg['models']:
        find=lambda p,i:next(s for s in summaries if s['intervals']==n and s['cohort']=='all_development' and s['panel']=='primary' and s['model']==model and s['projection']==p and s['initialization']==i)
        for init in cfg['initialization']:
            control,candidate=[find(p,init) for p in cfg['projection']];values=ratios(control,candidate)
            effects.append(dict(intervals=n,model=model,contrast='thin_over_dst_projection',held_fixed=init,
                per_case_cost_ratios=values,median_case_cost_ratio=st.median(values),both_solver_valid=not control['invalid_count'] and not candidate['invalid_count']))
        for projection in cfg['projection']:
            control,candidate=[find(projection,i) for i in cfg['initialization']];values=ratios(control,candidate)
            effects.append(dict(intervals=n,model=model,contrast='mean_over_nearest_initialization',held_fixed=projection,
                per_case_cost_ratios=values,median_case_cost_ratio=st.median(values),both_solver_valid=not control['invalid_count'] and not candidate['invalid_count'],
                paired_physical_error_changes=[b['physical_error']-a['physical_error'] for a,b in zip(control['cases'],candidate['cases'])]))
summary=dict(source_sha256=hashlib.sha256(payload).hexdigest(),provenance=d['provenance'],config=cfg,audit=audit,contract=d['contract'],
    checkpoints=d['checkpoints'],setup=d['setup'],references=d['references'],summaries=summaries,envelope=envelopes,factorial_cost_contrasts=effects)
(run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shortp={'skinny_sine_products':'sine products','forward_dst_and_gather':'DST gather'}
shorti={'mean_training_code':'mean code','nearest_cached_scaled_weak_prediction':'nearest code'}
nearest_ratios=[r['median_case_cost_ratio'] for r in effects if r['contrast']=='mean_over_nearest_initialization']
projection_ratios=[r['median_case_cost_ratio'] for r in effects if r['contrast']=='thin_over_dst_projection']
lines=['# Poisson source projection and initialization findings','',
    'These generated results are provisional development evidence from the approved frozen-checkpoint factorial. They compare complete query costs and errors under the existing per-start residual stopping rule; the final cohort stays closed.','',
    f"Job `{d['provenance']['job_id']}`, source `{d['provenance']['commit']}`, GPU `{d['provenance']['gpu']}`. The primary panel contains {audit['primary_invocations']} calls with {cfg['repetitions']} repetitions; separate stationary controls contain {audit['stationary_invocations']} single calls excluded from speed selection. All source draws and both frozen checkpoint hashes are checked against the preceding audited training study.",'',
    f"Query meshes use {cfg['intervals']} intervals, while the inherited training grid uses {d['checkpoints'][0]['config']['N']} nodes. Assemblies retain bank ranks {sorted({x['retained_bank_rank'] for x in d['setup']})} and smooth-test counts {sorted({x['retained_modes'] for x in d['setup']})}. The original relative-loss continuation was selected by worst adjusted error over every declared development case and both meshes, before this run.",'',
    '## Primary complete-query panel','',
    '| Intervals | Checkpoint | Projection | Initialization | Query ms | Median / worst physical error | Invalid | Nonstationary | Median same-grid DST/ROM |',
    '|---:|---|---|---|---:|---:|---:|---:|---:|']
for s in summaries:
    if s['cohort']!='all_development' or s['panel']!='primary' or not s['model']:continue
    lines.append(f"| {s['intervals']} | {s['model']} | {shortp[s['projection']]} | {shorti[s['initialization']]} | {1000*s['latency_seconds']:.7g} | {s['physical_median']:.8g} / {s['physical_max']:.8g} | {s['invalid_count']} | {s['nonstationary_count']} | {s['same_grid_dst_median_case_cost_ratio']:.7g} |")
lines+=['',f"Across the matched checkpoint/mesh/projection contrasts, median case ratios of mean-start to nearest-start cost range from {min(nearest_ratios):.7g} to {max(nearest_ratios):.7g}. "
    f"Median sine-product/DST-projection ratios range from {min(projection_ratios):.7g} to {max(projection_ratios):.7g}. "
    'The paired projection comparisons do not establish a consistent DST-projection speed gain. Aggregate latency and the median paired ratio summarize different aspects of the case distribution and can order configurations differently; medians and ratios do not commute. The envelope selects by aggregate case-median latency, while every reported paired ratio uses case-wise ratios.','',
    'Errors include failed outputs and use common observation nodes divided by the fine reference norm. Full requested-mesh same-grid discrepancies, both cohort summaries, raw repetition times and outliers remain in JSON. A nonstationary primary call may meet its intentional residual-reduction stop; it is not labeled stationary. Same-grid ratios above unity favor ROM and are raw cost comparisons, not a substitute for target qualification.','',
    '## Initialization and unchanged stopping procedure','',
    '| Intervals | Checkpoint | Projection | Initialization | Median initial residual | Median absolute tau threshold | Median final residual | Median attempts | Case stop reasons |',
    '|---:|---|---|---|---:|---:|---:|---:|---|']
for s in summaries:
    if s['cohort']!='all_development' or s['panel']!='primary' or not s['model']:continue
    lines.append(f"| {s['intervals']} | {s['model']} | {shortp[s['projection']]} | {shorti[s['initialization']]} | {s['initial_residual_median']:.7g} | {s['absolute_tau_threshold_median']:.7g} | {s['final_residual_median']:.7g} | {s['attempt_median']:.6g} | `{json.dumps(s['reason_counts'],sort_keys=True)}` |")
lines+=['',r"The threshold is $\tau\|r(z_0)\|_2$ for each call's own start. A nearer start can impose a tighter absolute target and require a stationary exit. Neither target nor residual is re-anchored. Cost differences therefore describe the complete initialization and existing stopping procedure. Every selected cache index, nearest distance, runner-up gap and near-tie flag is retained; the cache uses training-code predictions only.",'',
    '## Separate stationary accuracy controls','',
    '| Intervals | Checkpoint | Projection | Initialization | Median / worst physical error | Invalid | Nonstationary | Median attempts |',
    '|---:|---|---|---|---:|---:|---:|---:|']
for s in summaries:
    if s['cohort']!='all_development' or s['panel']!='stationary':continue
    lines.append(f"| {s['intervals']} | {s['model']} | {shortp[s['projection']]} | {shorti[s['initialization']]} | {s['physical_median']:.8g} / {s['physical_max']:.8g} | {s['invalid_count']} | {s['nonstationary_count']} | {s['attempt_median']:.6g} |")
lines+=['','These single-call controls retain timing components and every output, but never enter the speed envelope. Projection equality checks pair the same initialization choice. Different mean-versus-nearest local minima remain valid scientific outcomes and are assessed by error and stationarity, without an equality requirement. A small residual alone does not rescue a failed stationary control.','',
    '## Development cost-to-accuracy envelope','',
    '| Intervals | Target | ROM checkpoint / projection / start | Eligible FOM | ROM / FOM ms | Median case FOM/ROM |',
    '|---:|---:|---|---|---:|---:|']
for e in envelopes:
    if e['cohort']!='all_development':continue
    r,f=e['rom'],e['fom'];label='unattained' if not r else f"{r['model']} / {shortp[r['projection']]} / {shorti[r['initialization']]}"
    rt='—' if not r else f"{1000*r['latency_seconds']:.7g}";ft='—' if not f else f"{1000*f['latency_seconds']:.7g}"
    ratio='—' if e['median_case_cost_ratio'] is None else f"{e['median_case_cost_ratio']:.7g}"
    lines.append(f"| {e['intervals']} | {e['target']} | {label} | {f['arm'] if f else 'unattained'} | {rt} / {ft} | {ratio} |")
lines+=['',r'Eligibility requires every case to satisfy the solver and parity gates, $(e+\delta)/(1-\delta)\leq\epsilon$, and the declared reference allowance. Reference differences are empirical estimates, not rigorous continuum bounds. The FOM envelope includes same-grid DST and the charged coarse-grid solve with interpolation. Latencies are medians of case-median repetitions; paired ratios are medians of ratios of case medians. Selection remains development-only.','',
    '## Recorded primary cost components','',
    '| Intervals | Checkpoint / solver | Projection | Initialization | Input ms | Fused device ms | FOM solve/interpolation ms | Output ms |',
    '|---:|---|---|---|---:|---:|---:|---:|']
for s in summaries:
    if s['cohort']!='all_development' or s['panel']!='primary':continue
    parts=s['components_seconds'];fmt=lambda key:'—' if key not in parts else f'{1000*parts[key]:.7g}'
    lines.append(f"| {s['intervals']} | {s['model'] or s['arm']} | {shortp.get(s['projection'],'—')} | {shorti.get(s['initialization'],'—')} | {fmt('input_seconds')} | {fmt('fused_device_seconds')} | {fmt('solver_seconds')} | {fmt('output_seconds')} |")
lines+=['','Each component is a median of case-median values from the same primary invocations. Component medians need not sum to the total median.','',
    '| Intervals | Checkpoint | Bank build s | Weak assembly s | Cache build including compile s | Cache MiB |',
    '|---:|---|---:|---:|---:|---:|']
for setup in d['setup']:
    cache=setup['cache'];size=(cache['prediction_bytes']+cache['code_bytes'])/(1024**2)
    lines.append(f"| {setup['intervals']} | {setup['model']} | {setup['bank_build_seconds']:.7g} | {setup['weak_assembly_seconds']:.7g} | {cache['setup_seconds_including_compile']:.7g} | {size:.7g} |")
lines+=['','Setup values are actual observed offline durations including applicable compilation; they are not separately warmed speed comparisons. Query warm-up durations remain in native JSON.','',
    '## Timing scope and artifact audit','',audit['timing_boundary'],'',
    'Input, fused device and host-output intervals belong to each exact measured invocation. The fused device interval includes projection, optional lookup, guarded LM and decoding; those operations have no separately substituted component measurements. Offline cache/operator setup and compilation/warm-up records remain separate. Physical error, stationarity and numerical parity diagnostics use the actual output but are not included in query cost.','',
    f"The independent CPU audit checks {audit['distinct_preserved_field_hashes']} preserved full fields. Maximum physical-metric disagreement is {audit['maximum_independent_cpu_metric_difference']:.9g}, decoded-field relative disagreement {audit['maximum_decoder_field_relative_difference']:.9g}, cached-prediction disagreement {audit['maximum_cpu_cache_prediction_relative_difference']:.9g}, and stationarity difference {audit['maximum_stationarity_absolute_difference']:.9g}. Projection gate failures: {audit['projection_gate_failures']}; nearest-index mismatches: {audit['lookup_index_mismatches']}; near-tie calls: {audit['near_tie_invocations']}; primary/stationary fallback totals: {audit['timed_fallbacks']}/{audit['stationary_fallbacks']}. Source/checkpoint/archive checks and exact remote deletion passed.",'',
    '## Generated figures','',
    '![Complete-query development accuracy and cost](speed-cost-error.png)','',
    '![Paired projection and initialization effects](speed-paired-effects.png)','',
    'Standalone PDF versions accompany these data-generated figures.','',
    '## Plain-language glossary','',
    '- **Projection / sine products / DST gather:** smooth source coefficients / skinny sine matrix products / forward sine transform followed by selecting the same coefficients.',
    '- **Checkpoint / code / nearest cache:** frozen bank and head / compact neural coordinates / closest decoder-predicted weak source among training codes.',
    '- **Physical / same-grid / adjusted error:** common-observation refined discrepancy / complete requested-grid discrepancy / physical discrepancy enlarged using empirical reference refinement.',
    '- **Tau / initial / final / threshold / attempt:** requested residual reduction / residual at the chosen start / residual at the returned code / tau times that initial norm / attempted LM update.',
    '- **Stop reasons:** 0 is the iteration budget, 1 is small accepted relative decrease or step, 2 is the tau threshold, 3 is the damping cap, and 5 is a nonfinite initial residual. Stationarity is diagnosed separately.',
    '- **Stationary / invalid / parity:** small normalized objective gradient / failure of a declared solver or numerical gate / agreement of the two projection implementations at the same initialization.',
    '- **FOM / ROM / LM / guarded / fallback:** full solver / reduced model / damped nonlinear least squares / residual-checked small linear solve / charged generic linear solve if that check fails.',
    '- **Primary / stationary panel / envelope:** repeated speed measurements / single-call tighter accuracy controls / cheapest eligible tested configuration.',
    '- **Median case ratio / outlier / near tie / ms:** median of ratios of case-median costs / repetition above three aggregate latencies / small relative runner-up squared-distance gap / milliseconds.','']
(run/'FINDINGS.md').write_text('\n'.join(lines))
print(json.dumps(dict(envelope=[e for e in envelopes if e['cohort']=='all_development'],audit=audit),indent=2))
