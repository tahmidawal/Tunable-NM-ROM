"""Generate follow-up findings and preserve explicit timing-statistic definitions."""
from collections import defaultdict,Counter
from pathlib import Path
import argparse
import hashlib
import json
import statistics as st


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    p=args.run;src=p/'result.json'
    d=json.loads(src.read_text());cfg=d['config'];rows=d['rows'];assert d['complete']
    assert json.loads((p/'cleanup.json').read_text())['remote_deleted_and_absence_checked']
    groups=defaultdict(list)
    for r in rows:groups[(r['intervals'],r['arm'],r['requested_modes'],r['tau'])].append(r)
    summary=[]
    for (n,arm,m,tau),rs in groups.items():
        per=[]
        for case in range(cfg['cohort_count']):
            cr=[r for r in rs if r['case']==case];assert len(cr)==cfg['repetitions']
            assert len({r['field_sha256'] for r in cr})==1
            r=cr[0];times=[x['total_seconds'] for x in cr]
            per.append(dict(case=case,median_seconds=st.median(times),raw_seconds=times,
                physical_error=r['physical_error'],same_grid_error=r['same_grid_error'],
                conservative_physical_error=r['conservative_physical_error'],delta=r['reference_delta'],
                stationary=r['stationary'],stationarity=r['stationarity'],solver_valid=r['solver_valid'],
                reason=r['reason'],attempts=r['attempts'],parity_passed=r['parity_passed']))
        s=dict(intervals=n,arm=arm,requested_modes=m,retained_modes=rs[0]['retained_modes'],tau=tau,cases=per,
            latency_seconds=st.median(x['median_seconds'] for x in per),
            aggregate_median_seconds=st.median(r['total_seconds'] for r in rs),
            physical_median=st.median(x['physical_error'] for x in per),physical_max=max(x['physical_error'] for x in per),
            same_grid_max=max(x['same_grid_error'] for x in per),
            nonstationary_count=sum(not x['stationary'] for x in per),invalid_count=sum(not x['solver_valid'] for x in per),
            reason_counts=dict(Counter(str(x['reason']) for x in per)),
            median_attempts=st.median(x['attempts'] for x in per),targets={})
        s['timing_outliers']=sum(r['total_seconds']>3*s['latency_seconds'] for r in rs)
        component_names=('input_seconds','fused_device_seconds','output_seconds') if arm=='rom_fused' else ('input_seconds','projection_init_seconds','solver_seconds','output_seconds')
        s['components_seconds']={k:st.median(st.median(r[k] for r in rs if r['case']==i) for i in range(cfg['cohort_count'])) for k in component_names}
        for eps in cfg['targets']:
            failed=[x['case'] for x in per if not x['solver_valid'] or x['conservative_physical_error']>eps or x['delta']>cfg['reference_fraction']*eps]
            s['targets'][str(eps)]=dict(qualified=not failed,failure_count=len(failed),failed_cases=failed)
        summary.append(s)
    envelopes=[]
    for n in cfg['intervals']:
        for eps in cfg['targets']:
            valid=[s for s in summary if s['intervals']==n and s['targets'][str(eps)]['qualified']]
            rom=sorted((s for s in valid if s['arm'].startswith('rom')),key=lambda s:s['latency_seconds'])
            fom=sorted((s for s in valid if not s['arm'].startswith('rom')),key=lambda s:s['latency_seconds'])
            e=dict(intervals=n,target=eps,rom=None,fom=None,median_case_cost_ratio=None)
            for label,found in [('rom',rom),('fom',fom)]:
                if found:e[label]={k:found[0][k] for k in ('arm','requested_modes','retained_modes','tau','latency_seconds','physical_max')}
            if rom and fom:
                ratios=[f['median_seconds']/r['median_seconds'] for f,r in zip(fom[0]['cases'],rom[0]['cases'])]
                e.update(per_case_cost_ratios=ratios,median_case_cost_ratio=st.median(ratios),
                    ratio_of_reported_latencies=fom[0]['latency_seconds']/rom[0]['latency_seconds'])
            envelopes.append(e)
    fusion=[]
    for s in summary:
        if s['arm']!='rom_fused':continue
        mod=next(x for x in summary if x['intervals']==s['intervals'] and x['arm']=='rom_modular' and x['requested_modes']==s['requested_modes'] and x['tau']==s['tau'])
        ratios=[a['median_seconds']/b['median_seconds'] for a,b in zip(mod['cases'],s['cases'])]
        fusion.append(dict(intervals=s['intervals'],requested_modes=s['requested_modes'],tau=s['tau'],
            median_case_modular_over_fused=st.median(ratios),per_case_ratios=ratios,
            all_parity_passed=all(x['parity_passed'] for x in s['cases'])))
    # Reconciliation uses the same old raw observations; no cross-job timing comparison.
    oldpath=p.parent/'pilot01/result.json';reconciliation=[]
    if oldpath.exists():
        old=json.loads(oldpath.read_text())
        for n in old['config']['intervals']:
            rom=[r for r in old['rows'] if r['intervals']==n and r['arm']=='rom' and r['tau']==.01]
            fom=[r for r in old['rows'] if r['intervals']==n and r['arm']=='dst']
            medratio=[];repsratio=[]
            for case in range(old['config']['cohort_count']):
                rr=sorted((r for r in rom if r['case']==case),key=lambda r:r['repetition'])
                ff=sorted((r for r in fom if r['case']==case),key=lambda r:r['repetition'])
                medratio.append(st.median(r['total_seconds'] for r in ff)/st.median(r['total_seconds'] for r in rr))
                repsratio.append(st.median(f['total_seconds']/r['total_seconds'] for f,r in zip(ff,rr)))
            reconciliation.append(dict(intervals=n,earlier_median_of_per_rep_ratios=st.median(repsratio),
                campaign_median_of_case_median_ratios=st.median(medratio)))
    diagnostics=[]
    for n in cfg['intervals']:
        oo=[o for o in d['oracles'] if o['intervals']==n]
        weak=[x for x in summary if x['intervals']==n and x['arm']=='rom_modular' and x['tau']==0.]
        low=min(weak,key=lambda x:x['requested_modes']);high=max(weak,key=lambda x:x['requested_modes'])
        drifts=[abs(min(r['same_grid_error'] for r in o['rows'] if r['budget']==min(cfg['oracle_budgets']))-min(r['same_grid_error'] for r in o['rows'] if r['budget']==max(cfg['oracle_budgets']))) for o in oo]
        diagnostics.append(dict(intervals=n,weak_low_modes=low['requested_modes'],weak_high_modes=high['requested_modes'],
            low_mode_worst_physical=low['physical_max'],high_mode_worst_physical=high['physical_max'],
            bank_worst_same_grid=max(o['full_bank_same_grid_error'] for o in oo),
            best_head_worst_physical=max(o['best_head_metrics']['physical_error'] for o in oo),
            selected_head_stationarity_max=max(o['best_stationarity'] for o in oo),
            oracle_budget_best_error_drift_max=max(drifts),
            high_modes_to_best_head_physical_difference_max=max(abs(x['physical_error']-o['best_head_metrics']['physical_error']) for x,o in zip(high['cases'],oo))))
    result=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),config=cfg,provenance=d['provenance'],
        summaries=summary,envelope=envelopes,fusion=fusion,oracles=d['oracles'],reference=d['references'],
        parity=d['parity'],diagnostics_summary=diagnostics,first_pilot_statistic_reconciliation=reconciliation)
    (p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Frozen Poisson test-space and query-fusion findings','',
        'These generated results are provisional development evidence using the same frozen checkpoint and source cohort as the first pilot. '
        'They diagnose weak-test truncation and representation error, and measure a parity-checked query fusion against efficient direct solvers.','',
        f"Job `{d['provenance']['job_id']}`, source `{d['provenance']['commit']}`, GPU `{d['provenance']['gpu']}`. "
        f"No network training or sealed-final-cohort evaluation ran. The cohort has {cfg['cohort_count']} sources and {cfg['repetitions']} timing repetitions per configuration.",'',
        '## Accuracy and complete-query cost','',
        '| Intervals | Arm | Requested/retained modes | Tau | Latency ms | Median physical error | Worst physical error | Invalid | Nonstationary | Time outliers |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        modes='—' if s['requested_modes'] is None else f"{s['requested_modes']}/{s['retained_modes']}"
        lines.append(f"| {s['intervals']} | {s['arm']} | {modes} | {s['tau'] if s['tau'] is not None else '—'} | {s['latency_seconds']*1000:.5f} | {s['physical_median']:.6g} | {s['physical_max']:.6g} | {s['invalid_count']} | {s['nonstationary_count']} | {s['timing_outliers']} |")
    lines+=['','Latency is the median over sources of each source\'s median repetition time. '
        'All cases and timing outliers remain included. Physical error is measured on common nested observation nodes against independently refined reference; it is not a rigorous continuum norm bound.','',
        '## Cheapest qualifying development configurations','',
        '| Requested intervals | Target | ROM arm / modes / tau | FOM | ROM ms | FOM ms | Median case cost ratio |',
        '|---:|---:|---|---|---:|---:|---:|']
    fmt=lambda x:'unattained' if x is None else f'{x:.6g}'
    for e in envelopes:
        r=e['rom'];f=e['fom'];rlabel='unattained' if not r else f"{r['arm']} / {r['requested_modes']} / {r['tau']}"
        lines.append(f"| {e['intervals']} | {e['target']} | {rlabel} | {f['arm'] if f else 'unattained'} | {fmt(r['latency_seconds']*1000 if r else None)} | {fmt(f['latency_seconds']*1000 if f else None)} | {fmt(e['median_case_cost_ratio'])} |")
    lines+=['','For each source, the cost ratio is its median FOM time divided by its median ROM time; the table reports the median of those source ratios. '
        'A ratio above unity favors the ROM. The ratio of the two displayed aggregate latencies is a different statistic and is retained separately in JSON. '
        r'Every source must meet solver/parity conditions and the target after empirical reference adjustment $(e+\delta)/(1-\delta)$, with $\delta$ also within the reference budget. '
        'This is development selection, not independent confirmation.','',
        '## Representation diagnostics','',
        '| Intervals | Case | Full-bank same-grid error | Best head same-grid error | Best head physical error | Best head stationarity | Budget / start |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for o in d['oracles']:
        lines.append(f"| {o['intervals']} | {o['case']} | {o['full_bank_same_grid_error']:.6g} | {o['best_same_grid_error']:.6g} | {o['best_head_metrics']['physical_error']:.6g} | {o['best_stationarity']:.6g} | {o['best_budget']} / {o['best_start_index']} |")
    for diag in diagnostics:
        lines += ['',f"At {diag['intervals']} intervals, increasing the tight-solve test request from {diag['weak_low_modes']} to {diag['weak_high_modes']} changes worst physical error from {diag['low_mode_worst_physical']:.9g} to {diag['high_mode_worst_physical']:.9g}. The enlarged-test solutions differ in physical error from the best recorded full-field head fits by at most {diag['high_modes_to_best_head_physical_difference_max']:.9g} across sources. The maximum change in best head-fit error after doubling the oracle budget is {diag['oracle_budget_best_error_drift_max']:.9g}."]
    lines+=['','The full-bank projection is a higher-dimensional least-square diagnostic. The head oracle minimizes the full-field objective in exact QR coordinates from fixed deterministic starts. '
        'Both see the reference answer only for diagnosis; neither initializes or selects deployed queries. '
        'All starts, budgets, gradients, attempts and latent outputs are retained. A stationary local fit is not proof of a global optimum.','',
        '## Fusion parity and component cost','',
        '| Intervals | Modes | Tau | Median modular/fused case ratio | All field/latent/counter parity gates |',
        '|---:|---:|---:|---:|---|']
    for f in fusion:lines.append(f"| {f['intervals']} | {f['requested_modes']} | {f['tau']} | {f['median_case_modular_over_fused']:.6g} | {f['all_parity_passed']} |")
    lines+=['','| Intervals | Arm | Modes | Tau | Input ms | Projection/init ms | Solver ms | Fused device ms | Output ms |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        c=s['components_seconds'];vals=' | '.join('—' if k not in c else f'{1000*c[k]:.5f}' for k in ('input_seconds','projection_init_seconds','solver_seconds','fused_device_seconds','output_seconds'))
        lines.append(f"| {s['intervals']} | {s['arm']} | {s['requested_modes'] if s['requested_modes'] else '—'} | {s['tau'] if s['tau'] is not None else '—'} | {vals} |")
    lines+=['','A fused invocation has input, combined device pipeline and output timestamps. Its internal projection/solve/decode times cannot be separated without reintroducing synchronization. '
        'The segmented control supplies actual same-invocation component timings; they are not substituted into fused results. Component medians need not sum to the total median. '
        'The original generic small linear-system kernel remains unchanged, so this run does not exhaust kernel optimization.','',
        '## Reference and statistic limits','',
        f"Reference intervals are {cfg['reference_intervals']}; observations use {cfg['observation_intervals']} intervals. "
        f"Maximum empirical final refinement difference is {max(x['reference_delta'] for x in d['references']):.9g}. "
        'No Richardson reduction or rigorous error-bound claim is made. Larger test spaces are deployment changes on frozen networks; no per-resolution training was tested.','',
        '| First-pilot intervals | Earlier median of paired-repetition ratios | Campaign median of case-median ratios |',
        '|---:|---:|---:|']
    for r in reconciliation:lines.append(f"| {r['intervals']} | {r['earlier_median_of_per_rep_ratios']:.9g} | {r['campaign_median_of_case_median_ratios']:.9g} |")
    lines+=['','The two first-pilot values use the same raw observations and differ only in aggregation order. '
        'Earlier values are preserved explicitly; no data or scientific conclusion was silently replaced. Raw times are never compared across the first and second jobs.','',
        '## Plain-language glossary','',
        '- **Intervals / modes / retained:** cells per grid axis / smooth functions averaging the PDE / actual complete sine shells used after the requested cutoff.',
        '- **Arm / tau / latency:** measured implementation / requested initial-residual reduction / median of each source\'s median complete-query time.',
        '- **ROM / FOM / DST:** reduced model / full discrete model / direct sine-transform solver.',
        '- **Physical / same-grid error:** relative discrepancy from refined reference at common observation nodes / from the full solver on the same mesh.',
        '- **Invalid / nonstationary / time outlier:** failed solver or parity gate / gradient above stationary threshold / repeated time above three configuration latencies; all are retained.',
        '- **Full bank / head oracle / QR:** unrestricted learned spatial span / reference-only fit of nonlinear coefficients / exact orthonormal coordinates preserving field least squares.',
        '- **Budget / start / stationarity:** maximum LM attempts / fixed initial latent choice / normalized gradient measuring local first-order optimality.',
        '- **Modular / fused / parity:** separately dispatched stages / one compiled pipeline / field, latent and exact counter agreement.',
        '- **Input / projection / solver / output:** host field upload / source weak-mode preparation / latent or direct equation solution / complete host field and metadata return.',
        '- **Cost ratio / target / qualification:** FOM median cost divided by ROM median cost per source, then median over sources / requested error ceiling / all-source validity and uncertainty-adjusted accuracy.',
        '- **Reference difference / delta / e:** last nested-refinement discrepancy / that discrepancy normalized by fine-reference norm / measured error against fine reference.',
        '- **ms / s / checkpoint / cohort:** milliseconds / seconds / frozen trained weights / fixed development source group.','']
    (p/'FINDINGS.md').write_text('\n'.join(lines))
    print(json.dumps(envelopes,indent=2))

if __name__=='__main__':main()
