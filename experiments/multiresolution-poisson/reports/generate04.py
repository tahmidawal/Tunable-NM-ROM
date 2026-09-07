"""Generate the approved training factorial findings without hand-entered data."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,json,hashlib,statistics as st


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();p=a.run
    payload=(p/'result.json').read_bytes();d=json.loads(payload);cfg=d['config']
    audit=json.loads((p/'audit.json').read_text());assert audit['passed'] and d['complete']
    endpoints={x['model']:x for x in d['checkpoints']};groups=defaultdict(list)
    for r in d['rows']:
        for cohort in ('all_development',r['group']):groups[(r['intervals'],cohort,r['model'],r['arm'],r['tau'])].append(r)
    summaries=[]
    for (n,cohort,model,arm,tau),rows in groups.items():
        cases=[]
        for case in sorted({r['case'] for r in rows}):
            rs=sorted((r for r in rows if r['case']==case),key=lambda r:r['repetition']);assert len(rs)==cfg['repetitions']
            r=rs[0];times=[x['total_seconds'] for x in rs]
            cases.append(dict(case=case,median_seconds=st.median(times),raw_seconds=times,
                physical_error=max(x['physical_error'] for x in rs),same_grid_error=max(x['same_grid_error'] for x in rs),
                adjusted_error=max(x['conservative_physical_error'] for x in rs),reference_delta=r['reference_delta'],
                solver_valid=all(x['solver_valid'] for x in rs),stationary=all(x['stationary'] for x in rs),
                stationarity=max(x['stationarity'] for x in rs),reason=r['reason'],attempts=r['attempts'],
                parity_passed=all(x['parity_passed'] for x in rs),distinct_repetition_fields=len({x['field_sha256'] for x in rs})))
        latency=st.median(x['median_seconds'] for x in cases)
        components=('input_seconds','fused_device_seconds','output_seconds') if arm in ('rom_fused','rom_gj') else ('input_seconds','projection_init_seconds','solver_seconds','output_seconds')
        item=dict(intervals=n,cohort=cohort,model=model,arm=arm,tau=tau,requested_modes=rows[0]['requested_modes'],retained_modes=rows[0]['retained_modes'],
            cases=cases,case_count=len(cases),latency_seconds=latency,physical_median=st.median(x['physical_error'] for x in cases),
            physical_max=max(x['physical_error'] for x in cases),same_grid_max=max(x['same_grid_error'] for x in cases),
            invalid_count=sum(not x['solver_valid'] for x in cases),nonstationary_count=sum(not x['stationary'] for x in cases),
            numerical_gate_failures=sum(not x['parity_passed'] for x in cases),fallbacks=sum(x.get('fallback_count',0) for x in rows),
            timing_outliers=sum(x['total_seconds']>3*latency for x in rows),reason_counts=dict(Counter(str(x['reason']) for x in cases)),
            valid_endpoint=True if model is None else endpoints[model]['valid_endpoint'],
            components_seconds={k:st.median(st.median(x[k] for x in rows if x['case']==case['case']) for case in cases) for k in components},targets={})
        for eps in cfg['targets']:
            failed=[x['case'] for x in cases if not item['valid_endpoint'] or not x['solver_valid'] or x['adjusted_error']>eps or x['reference_delta']>cfg['reference_fraction']*eps]
            item['targets'][str(eps)]=dict(qualified=not failed,failed_cases=failed,failure_count=len(failed))
        summaries.append(item)
    cohorts=('existing_development','fresh_development','all_development');envelopes=[]
    for n in cfg['intervals']:
        for cohort in cohorts:
            for eps in cfg['targets']:
                valid=[s for s in summaries if s['intervals']==n and s['cohort']==cohort and s['targets'][str(eps)]['qualified']]
                rom=min((s for s in valid if s['model'] is not None),key=lambda s:s['latency_seconds'],default=None)
                fom=min((s for s in valid if s['model'] is None),key=lambda s:s['latency_seconds'],default=None)
                compact=lambda s:None if s is None else {k:s[k] for k in ('model','arm','tau','requested_modes','retained_modes','latency_seconds','physical_max')}
                row=dict(intervals=n,cohort=cohort,target=eps,rom=compact(rom),fom=compact(fom),median_case_cost_ratio=None)
                if rom and fom:
                    ratios=[f['median_seconds']/r['median_seconds'] for f,r in zip(fom['cases'],rom['cases'])]
                    row.update(per_case_cost_ratios=ratios,median_case_cost_ratio=st.median(ratios),ratio_of_aggregate_latencies=fom['latency_seconds']/rom['latency_seconds'])
                envelopes.append(row)
    diagnostic=[]
    for n in cfg['intervals']:
        for cohort in cohorts:
            for model in cfg['arms']:
                rr=[r for r in d['oracles'] if r['intervals']==n and r['model']==model and (cohort=='all_development' or r['group']==cohort)]
                diagnostic.append(dict(intervals=n,cohort=cohort,model=model,case_count=len(rr),
                    bank_physical_median=st.median(r['bank_metrics']['physical_error'] for r in rr),bank_physical_max=max(r['bank_metrics']['physical_error'] for r in rr),
                    bank_same_grid_max=max(r['full_bank_same_grid_error'] for r in rr),
                    head_physical_median=st.median(r['best_head_metrics']['physical_error'] for r in rr),head_physical_max=max(r['best_head_metrics']['physical_error'] for r in rr),
                    head_nonstationary_count=sum(r['best_stationarity']>cfg['stationarity_tolerance'] for r in rr),
                    head_stationarity_max=max(r['best_stationarity'] for r in rr)))
    effects=[]
    for n in cfg['intervals']:
        for cohort in cohorts:
            find=lambda model:next(s for s in summaries if s['intervals']==n and s['cohort']==cohort and s['model']==model and s['arm']=='rom_modular' and s['tau']==0)
            for objective in ('global','relative'):
                aa,bb=find('original_'+objective),find('expanded_'+objective)
                changes=[b['physical_error']-a['physical_error'] for a,b in zip(aa['cases'],bb['cases'])]
                effects.append(dict(intervals=n,cohort=cohort,contrast='expanded_minus_original_coverage',held_fixed=objective,
                    paired_error_changes=changes,median_paired_error_change=st.median(changes),improved_count=sum(v<0 for v in changes),both_solver_valid=all(x['solver_valid'] for s in (aa,bb) for x in s['cases'])))
            for coverage in ('original','expanded'):
                aa,bb=find(coverage+'_global'),find(coverage+'_relative')
                changes=[b['physical_error']-a['physical_error'] for a,b in zip(aa['cases'],bb['cases'])]
                effects.append(dict(intervals=n,cohort=cohort,contrast='relative_minus_global_loss',held_fixed=coverage,
                    paired_error_changes=changes,median_paired_error_change=st.median(changes),improved_count=sum(v<0 for v in changes),both_solver_valid=all(x['solver_valid'] for s in (aa,bb) for x in s['cases'])))
    result=dict(source_sha256=hashlib.sha256(payload).hexdigest(),provenance=d['provenance'],config=cfg,audit=audit,
        training=d['training'],checkpoints=d['checkpoints'],summaries=summaries,envelope=envelopes,
        representation=diagnostic,factorial_contrasts=effects,references=d['references'],setup=d['setup'])
    (p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Poisson training coverage and loss-normalization findings','',
        'These generated results are provisional development evidence from a fixed-compute continuation factorial, retaining the unchanged compact checkpoint. They separate training coverage from loss normalization and evaluate the resulting bank, head and deployed weak solve on original and fresh development sources.','',
        f"Job `{d['provenance']['job_id']}`, source `{d['provenance']['commit']}`, GPU `{d['provenance']['gpu']}`. Evaluation contains {cfg['existing_development_count']} existing and {cfg['additional_development_count']} fresh development sources, with {cfg['repetitions']} full-query repetitions. Sealed final cohorts remain closed.",'',
        '## Fixed training endpoints','',
        '| Model | Training sources | Updates | Complete endpoint | Compile s | Optimizer loop s | Actual train elapsed s | Training median relative error | Training worst relative error |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in d['training']['arms']:
        lines.append(f"| {r['tag']} | {r['coverage_count']} | {r['steps_done']} | {r['valid_endpoint']} | {r['compile_seconds']:.6g} | {r['optimizer_loop_seconds']:.6g} | {r['elapsed_seconds_including_compile']:.6g} | {r['training_metrics']['median_relative_l2']:.9g} | {r['training_metrics']['worst_relative_l2']:.9g} |")
    lines+=['',f"Total offline elapsed including source generation, training-code fitting, compilation and training diagnostics is {d['training']['offline_seconds_including_data_codefit_compilation_and_diagnostics']:.6g} s. "
        f"Training truth DST/CG discrepancy is {d['training']['cg_check']['relative_dst_discrepancy']:.9g} at CG tolerance {d['training']['cg_check']['tolerance']:.6g}.",'',
        'Global-loss arms share the original training set\'s full-field mean-square denominator. Relative-loss arms use each training field\'s fixed full-grid mean square. Coverage pairs share exact initial weights, training codes and random-key schedule across objectives. The larger set receives fewer passes per source at this matched update/batch budget. In-sample errors cover different source sets across coverage arms and are not held-out accuracy measurements. Fixed scheduled endpoints are reported; none was selected by intermediate validation. Offline durations are actual observed costs including compilation, without a separate warm training-speed measurement.','',
        '## Deployed stationary-control accuracy','',
        '| Intervals | Development cohort | Model | Median physical error | Worst physical error | Invalid | Nonstationary | Generic modular latency ms |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['cohort']=='all_development' or s['arm']!='rom_modular' or s['tau']!=0:continue
        lines.append(f"| {s['intervals']} | {s['cohort']} | {s['model']} | {s['physical_median']:.9g} | {s['physical_max']:.9g} | {s['invalid_count']} | {s['nonstationary_count']} | {s['latency_seconds']*1000:.6g} |")
    lines+=['','Physical errors above include every output, including failures. They use common nested observation nodes and the fine reference\'s norm; full-mesh same-grid errors remain separately available in JSON. The tighter generic solver is an accuracy diagnostic and is not automatically the selected cost configuration. The initialized code is each checkpoint\'s training-code mean, with no evaluation answer or source descriptor input.','',
        '## Bank and head diagnosis','',
        '| Intervals | Development cohort | Model | Worst bank same-grid error | Worst bank physical error | Worst best-head physical error | Nonstationary selected head fits |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in diagnostic:
        if r['cohort']=='all_development':continue
        lines.append(f"| {r['intervals']} | {r['cohort']} | {r['model']} | {r['bank_same_grid_max']:.9g} | {r['bank_physical_max']:.9g} | {r['head_physical_max']:.9g} | {r['head_nonstationary_count']} |")
    lines+=['','Bank projection and head fits see the discrete reference only for diagnosis and never initialize online queries. Head fitting minimizes the exact full-field objective in QR coordinates from fixed deterministic starts. A stationary local fit is not proof of a global optimum. All starts and terminal diagnostics remain recorded. Actual retained bank ranks and operator hashes are in setup records.','',
        '## Factorial contrasts','',
        '| Intervals | Development cohort | Contrast | Held fixed | Median paired physical-error change | Improved cases | Both solver-valid throughout |',
        '|---:|---|---|---|---:|---:|---|']
    for r in effects:
        if r['cohort']=='all_development':continue
        lines.append(f"| {r['intervals']} | {r['cohort']} | {r['contrast']} | {r['held_fixed']} | {r['median_paired_error_change']:.9g} | {r['improved_count']} | {r['both_solver_valid']} |")
    lines+=['','Contrasts use the tighter generic modular solve. Negative error changes favor expanded coverage or relative normalization, respectively. A contrast containing failed solves is descriptive, and does not establish a controlled accuracy gain. These results test the normalization/coverage hypothesis within this fixed architecture and compute budget; they do not establish a general model-family limitation or optimal training recipe.','',
        '## Qualifying complete-query development envelope','',
        '| Intervals | Development cohort | Target | ROM model / arm / tau | FOM | ROM ms | FOM ms | Median case FOM/ROM ratio |',
        '|---:|---|---:|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'unattained' if x is None else f'{x:.6g}'
    for e in envelopes:
        if e['cohort']!='all_development':continue
        r,f=e['rom'],e['fom'];label='unattained' if r is None else f"{r['model']} / {r['arm']} / {r['tau']}"
        lines.append(f"| {e['intervals']} | {e['cohort']} | {e['target']} | {label} | {f['arm'] if f else 'unattained'} | {fmt(r['latency_seconds']*1000 if r else None)} | {fmt(f['latency_seconds']*1000 if f else None)} | {fmt(e['median_case_cost_ratio'])} |")
    lines+=['',r'Every case must pass endpoint, solver and numerical gates plus $(e+\delta)/(1-\delta)\leq\epsilon$ and the reference allowance. This uses empirical refinement evidence, not rigorous continuum bounds. '
        'Latencies are medians of case-median repetition times; cost ratios are medians of ratios of case-median FOM and ROM times. A ratio above unity favors ROM. The displayed envelope selects among tested development configurations and provides no independent final confirmation. Separate existing/fresh envelopes, all raw repetitions, component costs, outlier counts and failures remain in JSON.','',
        f"The artifact audit checks {audit['row_count']} timed invocations and {audit['distinct_preserved_field_hashes']} distinct saved fields. Maximum CPU discrepancy in invocation metrics is {audit['maximum_independent_cpu_metric_difference']:.9g}; maximum oracle-metric discrepancy is {audit['maximum_oracle_cpu_error_difference']:.9g}. "
        f"Specialized agreement failures: {audit['specialized_agreement_failures']}; timed fallbacks: {audit['specialized_timed_fallbacks']}. "
        f"Maximum final reference refinement difference is {max(r['reference_delta'] for r in d['references']):.9g}. "
        'Original generic controls and the guarded specialized query share each job\'s GPU and return a full host field. The lookup/projection speed proposal is separate and absent from these endpoints.','',
        '## Selected-query component costs','',
        '| Intervals | Model | Arm | Tau | Input ms | Projection/init ms | Solver ms | Fused device ms | Output ms |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    selected=set()
    for e in envelopes:
        if e['cohort']!='all_development' or e['rom'] is None:continue
        for choice in (e['rom'],e['fom']):
            selected.add((e['intervals'],choice['model'],choice['arm'],choice['tau']))
        selected.add((e['intervals'],e['rom']['model'],'rom_modular',e['rom']['tau']))
    for row in summaries:
        key=(row['intervals'],row['model'],row['arm'],row['tau'])
        if row['cohort']!='all_development' or key not in selected:continue
        cc=row['components_seconds'];values=' | '.join('—' if k not in cc else f'{1000*cc[k]:.6g}' for k in ('input_seconds','projection_init_seconds','solver_seconds','fused_device_seconds','output_seconds'))
        lines.append(f"| {row['intervals']} | {row['model'] or '—'} | {row['arm']} | {row['tau'] if row['tau'] is not None else '—'} | {values} |")
    lines+=['','The selected full-query implementation and its generic modular control each supply their own component medians. Fused device time includes projection, solve, charged guards/fallback and decoding; separated control times are never substituted into a fused invocation. Component medians need not sum to the total median.','',
        '## Generated figures','',
        '![Common-observation accuracy by fixed endpoint](accuracy-factorial.png)','',
        '![Training objectives on the fixed schedules](training-objectives.png)','',
        '![Complete-query cost and error](cost-error-factorial.png)','',
        'The corresponding PDF files are standalone export artifacts. Every plotted value comes from the collected native JSON.','',
        '## Plain-language glossary','',
        '- **Model / original / expanded:** checkpoint evaluated / inherited training coverage / that coverage plus new sources from the same family.',
        '- **Global / relative loss:** squared field error divided by one shared scale / each field\'s squared error divided by its own full-field scale.',
        '- **Update / endpoint / compile / elapsed:** optimizer step / final scheduled checkpoint / preparation of device executable / actual wall time including compilation.',
        '- **Existing / fresh development:** previously inspected diagnostic sources / independently seeded sources held outside all training; neither is a sealed final cohort.',
        '- **Intervals / modes / bank / head:** grid cells per axis / smooth PDE tests / learned spatial features / nonlinear map from compact coordinates to feature coefficients.',
        '- **Physical / same-grid / paired error change:** common-observation discrepancy from refined reference / full-mesh discrepancy from discrete full solver / treatment error minus control error on the same source.',
        '- **Stationary / invalid / oracle / QR:** small normalized objective gradient / failed gate / reference-only diagnostic fit / orthonormal coordinates preserving the field least-squares objective.',
        '- **ROM / FOM / DST / modular / fused / GJ:** reduced model / full discrete model / direct sine-transform solver / separated query stages / combined device pipeline / guarded Gauss–Jordan linear solve.',
        '- **Tau / target / envelope / cost ratio:** requested initial-residual reduction / error ceiling / cheapest qualifying tested configuration / median of case-median FOM cost divided by ROM cost.',
        '- **Fallback / outlier / delta / e / epsilon:** charged generic linear solve / repetition above three aggregate latencies / final refinement discrepancy relative to fine-reference norm / observed error / requested ceiling.',
        '- **Coverage / factorial / held fixed / matched compute:** sampled training source range / independently crossed treatments / unchanged treatment dimension / same updates and batch sizes.',
        '- **s / ms / source hash / checkpoint hash:** seconds / milliseconds / code identity / saved model identity.','']
    (p/'FINDINGS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(training_endpoints=audit['training_endpoints'],envelope=[e for e in envelopes if e['cohort']=='all_development']),indent=2))

if __name__=='__main__':main()
