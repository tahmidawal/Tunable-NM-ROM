"""Generate complete-query kernel findings exclusively from collected JSON."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,hashlib,json,statistics as st

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();p=a.run
    raw=(p/'result.json').read_bytes();d=json.loads(raw);cfg=d['config']
    audit=json.loads((p/'audit.json').read_text());assert d['complete'] and audit['passed']
    groups=defaultdict(list)
    for r in d['rows']:groups[(r['intervals'],r['arm'],r['requested_modes'],r['tau'])].append(r)
    summaries=[]
    for (n,arm,m,tau),rows in groups.items():
        cases=[]
        for case in range(cfg['cohort_count']):
            rs=sorted((r for r in rows if r['case']==case),key=lambda r:r['repetition'])
            assert len(rs)==cfg['repetitions'] and len({r['field_sha256'] for r in rs})==1
            times=[r['total_seconds'] for r in rs];r=rs[0]
            cases.append(dict(case=case,median_seconds=st.median(times),raw_seconds=times,
                physical_error=r['physical_error'],same_grid_error=r['same_grid_error'],
                conservative_physical_error=r['conservative_physical_error'],delta=r['reference_delta'],
                stationary=r['stationary'],stationarity=r['stationarity'],solver_valid=all(x['solver_valid'] for x in rs),
                reason=r['reason'],attempts=r['attempts'],accepted=r.get('accepted'),jacobians=r.get('jacobians'),
                parity_passed=all(x['parity_passed'] for x in rs),fallbacks=sum(x.get('fallback_count',0) for x in rs)))
        latency=st.median(x['median_seconds'] for x in cases)
        components=('input_seconds','fused_device_seconds','output_seconds') if arm in ('rom_fused','rom_gj') else ('input_seconds','projection_init_seconds','solver_seconds','output_seconds')
        item=dict(intervals=n,arm=arm,requested_modes=m,retained_modes=rows[0]['retained_modes'],tau=tau,cases=cases,
            latency_seconds=latency,physical_median=st.median(x['physical_error'] for x in cases),
            physical_max=max(x['physical_error'] for x in cases),same_grid_max=max(x['same_grid_error'] for x in cases),
            invalid_count=sum(not x['solver_valid'] for x in cases),nonstationary_count=sum(not x['stationary'] for x in cases),
            reason_counts=dict(Counter(str(x['reason']) for x in cases)),median_attempts=st.median(x['attempts'] for x in cases),
            fallbacks=sum(x['fallbacks'] for x in cases),timing_outliers=sum(r['total_seconds']>3*latency for r in rows),
            components_seconds={k:st.median(st.median(r[k] for r in rows if r['case']==case) for case in range(cfg['cohort_count'])) for k in components},targets={})
        for eps in cfg['targets']:
            failed=[x['case'] for x in cases if not x['solver_valid'] or x['conservative_physical_error']>eps or x['delta']>cfg['reference_fraction']*eps]
            item['targets'][str(eps)]=dict(qualified=not failed,failed_cases=failed,failure_count=len(failed))
        summaries.append(item)
    ratios=[]
    for s in summaries:
        if s['arm']!='rom_gj':continue
        x=dict(intervals=s['intervals'],requested_modes=s['requested_modes'],retained_modes=s['retained_modes'],tau=s['tau'])
        for arm in ('rom_modular','rom_fused','dst'):
            control=next(c for c in summaries if c['intervals']==s['intervals'] and c['arm']==arm and (arm=='dst' or (c['requested_modes']==s['requested_modes'] and c['tau']==s['tau'])))
            values=[c['median_seconds']/q['median_seconds'] for c,q in zip(control['cases'],s['cases'])]
            x[arm+'_over_gj']=st.median(values);x[arm+'_over_gj_per_case']=values
        ratios.append(x)
    envelopes=[]
    for n in cfg['intervals']:
        for eps in cfg['targets']:
            valid=[s for s in summaries if s['intervals']==n and s['targets'][str(eps)]['qualified']]
            rom=min((s for s in valid if s['arm'].startswith('rom')),key=lambda s:s['latency_seconds'],default=None)
            fom=min((s for s in valid if not s['arm'].startswith('rom')),key=lambda s:s['latency_seconds'],default=None)
            e=dict(intervals=n,target=eps,rom=rom,fom=fom,median_case_cost_ratio=None)
            if rom and fom:
                rr=[f['median_seconds']/r['median_seconds'] for f,r in zip(fom['cases'],rom['cases'])]
                e.update(per_case_cost_ratios=rr,median_case_cost_ratio=st.median(rr),ratio_of_reported_latencies=fom['latency_seconds']/rom['latency_seconds'])
            envelopes.append(e)
    agreement=dict(field_relative_max=max(x['field_relative'] for x in d['parity']),
        latent_relative_max=max(x['latent_relative'] for x in d['parity']),
        initial_scaled_residual_norm_difference_max=max(x['objective_scaled_difference'] for x in d['parity']),
        failed=sum(not x['passed'] for x in d['parity']),
        changed_counters=sum(any(x[k]!=0 for k in ('attempt_difference','accepted_difference','jacobian_difference')) for x in d['parity']),
        changed_reasons=sum(x['candidate_reason']!=x['control_reason'] for x in d['parity']),
        replay_counter_mismatches=sum(not x['replay_counter_agreement'] for x in d['trajectories']))
    result=dict(source_sha256=hashlib.sha256(raw).hexdigest(),provenance=d['provenance'],config=cfg,
        summaries=summaries,envelope=envelopes,paired_ratios=ratios,agreement=agreement,audit=audit,
        reference=d['references'],synthetic_controls=d['synthetic_controls'],trajectories=d['trajectories'])
    (p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Guarded Poisson small-system kernel findings','',
        'These generated results are provisional development evidence on the unchanged separable checkpoint and source cohort. They compare complete host-input/host-output queries while changing only the damped-normal-system solve, with explicit numerical-agreement and fallback checks.','',
        f"Job `{d['provenance']['job_id']}`, source `{d['provenance']['commit']}`, GPU `{d['provenance']['gpu']}`. The archive audit covers {audit['row_count']} invocations and {audit['normal_systems_checked']} replayed linear systems. No training or sealed-final evaluation ran.",'',
        '## Full-query accuracy and latency','',
        '| Intervals | Arm | Requested/retained modes | Tau | Latency ms | Worst physical error | Invalid | Nonstationary | Time outliers | Fallbacks |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        m='—' if s['requested_modes'] is None else f"{s['requested_modes']}/{s['retained_modes']}"
        lines.append(f"| {s['intervals']} | {s['arm']} | {m} | {s['tau'] if s['tau'] is not None else '—'} | {s['latency_seconds']*1000:.6g} | {s['physical_max']:.9g} | {s['invalid_count']} | {s['nonstationary_count']} | {s['timing_outliers']} | {s['fallbacks']} |")
    lines+=['','Latency is the median over cases of each case\'s median repetition time. Nonstationary tau-controlled outputs remain eligible only when their explicit residual-reduction stopping condition was reached and independently checked physical accuracy qualifies. Tighter nonstationary outputs remain invalid. Every outlier and unsuccessful output stays in the data.','',
        '## Paired complete-query kernel effect','',
        '| Intervals | Modes | Tau | Original modular / specialized | Original fused / specialized | Same-grid DST / specialized |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in ratios:lines.append(f"| {r['intervals']} | {r['requested_modes']} | {r['tau']} | {r['rom_modular_over_gj']:.6g} | {r['rom_fused_over_gj']:.6g} | {r['dst_over_gj']:.6g} |")
    lines+=['','Every ratio first divides the two case-median times, then takes the median over cases. A ratio above unity favors the specialized query. These same-mesh ratios do not imply target qualification and do not replace the cheaper qualifying FOM envelope below. No times from earlier jobs are used.','',
        '| Intervals | Target | Qualifying ROM / modes / tau | Qualifying FOM | ROM ms | FOM ms | FOM / ROM case ratio |',
        '|---:|---:|---|---|---:|---:|---:|']
    fmt=lambda v:'unattained' if v is None else f'{v:.6g}'
    for e in envelopes:
        r,f=e['rom'],e['fom'];label='unattained' if r is None else f"{r['arm']} / {r['requested_modes']} / {r['tau']}"
        lines.append(f"| {e['intervals']} | {e['target']} | {label} | {f['arm'] if f else 'unattained'} | {fmt(r['latency_seconds']*1000 if r else None)} | {fmt(f['latency_seconds']*1000 if f else None)} | {fmt(e['median_case_cost_ratio'])} |")
    lines+=['',r'Qualification requires every case to satisfy solver and numerical-agreement gates, $(e+\delta)/(1-\delta)\leq\epsilon$, and the configured reference-difference allowance. The reference adjustment is empirical, not a rigorous continuum bound. This envelope is selected on development data and is not independent final confirmation.','',
        '## Numerical agreement and linear systems','',
        f"Maximum specialized-versus-original field discrepancy is {agreement['field_relative_max']:.9g}, latent discrepancy {agreement['latent_relative_max']:.9g}, and initial-residual-scaled terminal residual-norm change {agreement['initial_scaled_residual_norm_difference_max']:.9g}. "
        f"Failed agreement gates: {agreement['failed']}; changed counters: {agreement['changed_counters']}; changed stopping reasons: {agreement['changed_reasons']}; replay counter mismatches: {agreement['replay_counter_mismatches']}.",'',
        f"Independent CPU replay checks cover {audit['normal_systems_checked']} SPD normal systems. Maximum condition number is {audit['normal_system_condition_max']:.9g}; maximum linear backward error is {audit['normal_system_backward_max']:.9g}; maximum relative step discrepancy against CPU generic solve is {audit['normal_system_cpu_solve_difference_max']:.9g}. "
        f"Timed specialized fallback count is {audit['specialized_timed_fallbacks']}. Online guards and any fallback execute within the measured fused query.",'',
        '| Requested condition | Maximum known-solution relative error | Maximum backward error | Forward-error limit |',
        '|---:|---:|---:|---:|']
    for cond in cfg['synthetic_condition_numbers']:
        rs=[r for r in d['synthetic_controls'] if r['requested_condition']==cond]
        lines.append(f"| {cond:.6g} | {max(r['known_solution_relative_error'] for r in rs):.9g} | {max(r['backward_error'] for r in rs):.9g} | {rs[0]['forward_limit']:.6g} |")
    lines+=['','The separate non-SPD smoke fixture asserts that fallback is taken. Small backward error alone does not guarantee a small forward error for ill-conditioned systems; known-solution synthetic checks and actual final-field agreement provide complementary evidence. The original modular/fused control uses untouched `ctol_tol` source. Replays archive the actual matrices and steps but supply no timing data.','',
        '## Cost components and reference scope','',
        '| Intervals | Arm | Modes | Tau | Input ms | Projection/init ms | Solver ms | Fused device ms | Output ms |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        c=s['components_seconds'];v=' | '.join('—' if k not in c else f'{1000*c[k]:.6g}' for k in ('input_seconds','projection_init_seconds','solver_seconds','fused_device_seconds','output_seconds'))
        lines.append(f"| {s['intervals']} | {s['arm']} | {s['requested_modes'] or '—'} | {s['tau'] if s['tau'] is not None else '—'} | {v} |")
    lines+=['','Fused projection, solve, residual checks and field decoding share one device interval. The original modular query provides separately synchronized components; these values are not substituted into fused invocations. Post-query physical and final-gradient validation is excluded from all query timings. Setup, operator construction and compilation/warmup remain separately recorded.','',
        f"References use {cfg['reference_intervals']} intervals with common observation at {cfg['observation_intervals']} intervals. Maximum final empirical refinement difference is {max(r['reference_delta'] for r in d['references']):.9g}. The artifact audit independently recomputes every recorded field metric, with maximum discrepancy {audit['maximum_independent_cpu_metric_difference']:.9g}. "
        'Physical error is relative discrepancy on common nested observation nodes; same-grid discrepancy is retained separately in JSON. No new representation oracle is run here and no global approximation-floor claim follows from the kernel study.','',
        '## Plain-language glossary','',
        '- **Intervals / requested / retained modes:** cells per grid axis / desired number of smooth PDE tests / actual complete sine shells retained.',
        '- **Arm / tau / latency:** implementation / initial-residual reduction target / median of case-median complete-query times.',
        '- **ROM / FOM / DST / GJ:** reduced model / full discrete model / direct sine-transform solver / diagonally scaled Gauss–Jordan elimination.',
        '- **Physical / same-grid error:** common-observation relative discrepancy from refined reference / full-mesh relative discrepancy from the same discrete full solver.',
        '- **Stationary / invalid / outlier / fallback:** normalized objective gradient below threshold / failed numerical or solver gate / repetition above three aggregate latencies / charged use of generic linear solve.',
        '- **Modular / fused / specialized:** separately dispatched query stages / one compiled device pipeline / fused query using guarded small-system elimination.',
        '- **Condition / backward / forward error:** sensitivity to perturbations / normalized linear equation residual / relative solution discrepancy.',
        '- **SPD / normal system / damping:** symmetric positive definite / equations formed from the residual Jacobian / regularization of the proposed step.',
        '- **Replay / counter / numerical agreement:** untimed diagnostic rerun / recorded solver work / scale-aware output comparison under frozen limits.',
        '- **Input / projection / solver / output:** host source upload / weak-source construction and initialization / latent or direct solve / full host field and metadata return.',
        '- **Target / reference allowance / envelope:** requested error ceiling / permitted empirical refinement difference / cheapest qualifying tested development configuration.',
        '- **Delta / e / epsilon / ms:** last refinement discrepancy relative to fine-reference norm / observed field error / requested ceiling / milliseconds.','']
    (p/'FINDINGS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(agreement=agreement,paired_ratios=ratios,envelope=[{k:v for k,v in e.items() if k not in ('rom','fom')} for e in envelopes]),indent=2))

if __name__=='__main__':main()
