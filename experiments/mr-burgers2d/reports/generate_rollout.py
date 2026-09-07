"""Generate the corrected full-rollout report from audited paired invocations."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np


def summarize(d):
    groups=collections.defaultdict(list)
    for r in d['invocations']:groups[r['name']].append(r)
    assert set(groups)=={r['name'] for r in d['declared_subjects']}
    cases=range(d['config']['cases']);reps=range(d['config']['reps']);summary=[]
    for name,rows in groups.items():
        assert len(rows)==len(cases)*len(reps) and {(r['case'],r['rep']) for r in rows}=={(c,p) for c in cases for p in reps}
        cfg=rows[0];times={c:float(np.median([r['seconds'] for r in rows if r['case']==c])) for c in cases}
        valid=all(r['finite'] and (r['nonlinear_tolerance_satisfied'] if r['method']=='fom' else r['ic_reason']!=3 and 3 not in r['stop_reasons']) for r in rows)
        q=dict(name=name,method=cfg['method'],intervals=cfg['output_intervals'],solver_intervals=cfg['solver_intervals'],times=times,
            ms=1000*float(np.median(list(times.values()))),valid=valid,
            timing_outliers_gt2x_case_median=sum(r['seconds']>2*times[r['case']] for r in rows),
            cold_rule=cfg.get('cold_rule'),ic_budget=cfg.get('ic_budget'),dt=cfg['dt'],stall=cfg.get('stall'),
            initial_budget_stops=sum(r.get('ic_reason')==0 for r in rows),initial_improvement_stops=sum(r.get('ic_reason')==2 for r in rows),
            evolution_budget_stops=sum(r.get('stop_reasons',[]).count(0) for r in rows),
            first_step_budget_stops=sum(bool(r.get('stop_reasons')) and r['stop_reasons'][0]==0 for r in rows),evolution_improvement_stops=sum(r.get('stop_reasons',[]).count(2) for r in rows),
            failed_invocations=sum(not r['finite'] or (not r['nonlinear_tolerance_satisfied'] if r['method']=='fom' else r['ic_reason']==3 or 3 in r['stop_reasons']) for r in rows))
        for norm,key in [('common','physical_error'),('dense','dense_physical_error')]:
            case_errors={c:max(r[key]['fixed_initial_max'] if r[key] else np.inf for r in rows if r['case']==c) for c in cases}
            q[norm]=dict(case_errors=case_errors,worst=max(case_errors.values()),median=float(np.median(list(case_errors.values()))),outliers_0p01=sum(e>.01 for e in case_errors.values()))
        summary.append(q)
    return summary,groups


def select(d,summary):
    rows=[]
    for norm in ['common','dense']:
        for L in sorted({r['intervals'] for r in summary}):
            met=d['reference_metrics_by_output'][str(d['observation_intervals'] if norm=='common' else L)]
            margin={r['case']:r['conservative_difference_sum'] for r in met['uncertainty']}
            order_ok=all(r['asymptotic_decrease_observed'] and r['empirical_richardson_estimate'] is not None for r in met['order_audit'])
            for r in met['order_audit']:
                if r['empirical_richardson_estimate'] is not None:margin[r['case']]=max(margin[r['case']],r['empirical_richardson_estimate'])
            for target in [.1,.05,.01]:
                qualified=order_ok and max(margin.values())<=target/10
                eligible=[r for r in summary if r['intervals']==L and r['valid'] and all(e+margin[c]<=target for c,e in r[norm]['case_errors'].items())]
                rom=min((r for r in eligible if r['method']=='rom'),key=lambda r:(r['ms'],r['name']),default=None)
                fom=min((r for r in eligible if r['method']=='fom'),key=lambda r:(r['ms'],r['name']),default=None)
                ratio=float(np.median([fom['times'][c]/rom['times'][c] for c in margin])) if rom and fom and qualified else None
                rows.append(dict(norm=norm,intervals=L,target=target,reference_margin=max(margin.values()),empirical_reference_budget_passed=qualified,
                    rom=rom,fom=fom,paired_ratio=ratio,rigorous_reference_bound=None))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('output');a=p.parse_args();run=Path(a.run)
    d=json.loads((run/'out/pilot.json').read_text());audit=json.loads((run/'AUDIT.json').read_text())
    assert d['complete'] and d['experiment'] in ['fixed_gauss_rollout','gauss_timestep_study']
    assert audit['source_sha256']==hashlib.sha256((run/'out/pilot.json').read_bytes()).hexdigest()
    summary,groups=summarize(d);selected=select(d,summary)
    (run/'SUMMARY.json').write_text(json.dumps(dict(configurations=summary,selections=selected),indent=2)+'\n')
    wins=[r for r in selected if r['paired_ratio'] is not None and r['paired_ratio']>1]
    result='A measured development crossover appears in the selections below; it requires independent confirmation.' if wins else 'No tested configuration establishes a complete-query ROM advantage over the eligible efficient FOM envelope.'
    largest=max(r['intervals'] for r in summary)
    timestep=d['experiment']=='gauss_timestep_study'
    edge=next((r for r in summary if r['name']==f'rom_L{largest}_edge_ic60_dt0.005_stall0.01_starts1'),None)
    gauss=next(r for r in summary if r['name']==f'rom_L{largest}_fixed_gauss_ic180_dt0.005_stall0.01_starts1')
    fastest=min((r for r in summary if r['method']=='rom' and r['intervals']==largest),key=lambda r:r['ms'])
    choice=next(r for r in selected if r['norm']=='dense' and r['intervals']==largest and r['target']==.05)
    lead=(f"At {largest} intervals, the least-cost ROM meeting the empirical {100*choice['target']:g}% target uses timestep {choice['rom']['dt']:g}: complete-grid error {choice['rom']['dense']['worst']:.7g} at {choice['rom']['ms']:.4f} ms, compared with {choice['fom']['ms']:.4f} ms for the eligible FOM. The original Gauss step size costs {gauss['ms']:.4f} ms in this same job." if timestep and choice['rom'] and choice['fom'] else f"The timestep study retains every measured configuration below, including target misses." if timestep else f"At {largest} intervals, the primary fixed-Gauss rollout lowers worst complete-grid error from {edge['dense']['worst']:.7g} to {gauss['dense']['worst']:.7g}, with complete-query costs {edge['ms']:.4f} and {gauss['ms']:.4f} ms respectively.")
    lineage=''
    if timestep:
        previous=json.loads((run.parent/'rollout04/out/pilot.json').read_text())
        assert previous['physical_cases']==d['physical_cases'] and previous['checkpoint_sha256']==d['checkpoint_sha256']
        lineage=f"This timestep study follows the corrected-initializer comparison from job `{previous['job_id']}`. [Its audited paired configuration summary](../runs/rollout04/SUMMARY.json) retains the original edge/Gauss comparison. The present study remeasures its Gauss control in this job; no cross-job timing ratio is used."
    lines=['# Burgers 2D: complete-query accuracy and time-step cost' if timestep else '# Burgers 2D: fixed physical initialization in the complete query','',
        'Audited development results for unchanged coordinate-separable network weights across the requested meshes. '+result,'',
        lead,'',lineage,'',
        f"Source `{d['commit']}`, job `{d['job_id']}`, GPU `{d['gpu']}`; backend `{d['backend']}`, f64 `{d['x64']}`, precision `{d['matmul_precision']}`. Checkpoint SHA-256 `{d['checkpoint_sha256']}`.",'',
        f"The clipped Gaussian family and seed {d['config']['seed']} retain all {d['config']['cases']} physical cases. Each configuration has {d['config']['reps']} timing repetitions; final cases remain unopened. The inherited checkpoint was trained on {d['checkpoint_training_intervals']} intervals and is evaluated at {d['config']['meshes']} intervals.",'',
        r'The scalar equation is $u_t+u(u_x+u_y)=\nu\Delta u$ on the unit square, with zero Dirichlet walls. Both methods use backward Euler and sign-dependent upwinding. The FOM uses adaptive Newton/BiCGStab with exact Helmholtz preconditioning through FFT sine transforms.','',
        'The existing linear latent extrapolation predictor is retained and used only when it lowers the weak residual. The query starts with the supplied dense host field and viscosity, and ends with all requested dense host fields. Fixed Gauss sampling charges bilinear interpolation from that input. Weak evolution, sign-dependent upwinding and nonnegative advection quadrature remain unchanged. Initial fit, evolution, input/output transfers and requested dense reconstruction are included in query cost. Setup and compilation are separate.','',
        f"Output times are {d['output_times']}. The common-grid metric observes {d['observation_intervals']} intervals; the complete-grid metric scores every node on the requested mesh. Both divide by the initial reference-field norm. Configuration cost is the median of per-case repetition medians. Each paired ratio is the median of per-case FOM/ROM median-time ratios from this job. GPU burn-in precedes every timed call, and the recorded configuration-order seed is {d['timing_order_seed']}.",'',
        '## Initial fitting and subsequent evolution','',
        'The table scores actual returned fields. Initial and later errors use the same initial-field normalization. A later error increase does not by itself isolate spatial representation, the nonlinear head, quadrature or optimization.','',
        '| Intervals | ROM setting | Query ms | Initial complete-grid worst | Later complete-grid worst | Complete-grid same-mesh worst | Current-relative complete-grid worst | IC budget / improvement stops | Evolution budget / improvement stops |',
        '|---|---|---:|---:|---:|---:|---:|---|---|']
    for r in sorted((r for r in summary if r['method']=='rom'),key=lambda r:(r['intervals'],r['name'])):
        raw=groups[r['name']];initial=max((x['dense_physical_error']['fixed_initial_per_time'][0] for x in raw if x['dense_physical_error']),default=np.inf)
        later=max((max(x['dense_physical_error']['fixed_initial_per_time'][1:]) for x in raw if x['dense_physical_error']),default=np.inf)
        same=max((x['dense_same_grid_error']['fixed_initial_max'] for x in raw if x['dense_same_grid_error']),default=np.inf)
        current=max((x['dense_physical_error']['current_relative_max'] for x in raw if x['dense_physical_error']),default=np.inf)
        lines.append(f"| {r['intervals']} | `{r['name']}` | {r['ms']:.4f} | {initial:.8g} | {later:.8g} | {same:.8g} | {current:.8g} | {r['initial_budget_stops']} / {r['initial_improvement_stops']} | {r['evolution_budget_stops']} / {r['evolution_improvement_stops']} |")
    if timestep:
        lines+=['','All timestep arms use the same fixed physical fitting points and budget. Interpolation of the supplied dense field is charged and can still depend on mesh resolution. The following work and exit counts are from the measured complete invocations; budget exits are retained even when their returned field passes an accuracy target.','',
            '| Intervals | dt | Steps per query | Median LM attempts per query | Largest final weak residual | Evolution exits: budget / residual / improvement / failed | First-step budget exits |', '|---|---:|---:|---:|---:|---|---:|']
        for r in sorted((r for r in summary if r['method']=='rom'),key=lambda r:(r['intervals'],r['dt'])):
            raw=groups[r['name']];counts=collections.Counter(s for x in raw for s in x['stop_reasons'])
            residual=max((v for x in raw for v in x['residuals'] if v is not None),default=np.inf)
            lines.append(f"| {r['intervals']} | {r['dt']:g} | {len(raw[0]['iterations'])} | {np.median([sum(x['iterations']) for x in raw]):.1f} | {residual:.7g} | "+' / '.join(str(counts[i]) for i in range(4))+f" | {r['first_step_budget_stops']} |")
    lines+=['','## Reference refinement and empirical eligibility','',
        f"The finest reference has {d['config']['reference_mesh']} intervals and timestep {d['config']['reference_dt']:g}. The largest reference nonlinear relative residual is {max(r['max_relative_residual'] for r in d['reference']):.7g}. Spatial and temporal orders are observed from three levels; the margin is the larger of their raw-difference sum and the Richardson estimate for each case. Qualification also requires this margin to be at most one tenth of the target. These estimates do not supply a rigorous reference bound.",'',
        '| Scoring intervals | Worst empirical margin | Minimum / maximum spatial order | Minimum / maximum time order |', '|---|---:|---|---|']
    for key,met in d['reference_metrics_by_output'].items():
        margin={r['case']:r['conservative_difference_sum'] for r in met['uncertainty']}
        for r in met['order_audit']:
            if r['empirical_richardson_estimate'] is not None:margin[r['case']]=max(margin[r['case']],r['empirical_richardson_estimate'])
        ps=[r['observed_spatial_order'] for r in met['order_audit']];pt=[r['observed_temporal_order'] for r in met['order_audit']]
        lines.append(f"| {key} | {max(margin.values()):.8g} | {min(ps):.5f} / {max(ps):.5f} | {min(pt):.5f} / {max(pt):.5f} |")
    lines+=['','Every eligible case must satisfy measured error plus its reference margin at the target. The FOM envelope retains all declared coarse solves and the same-mesh solve, charging output interpolation. Configurations may be selected separately for each metric and requested resolution.','',
        '| Norm | Output intervals | Target | Reference budget | Selected ROM | Selected FOM | ROM / FOM ms | Paired FOM/ROM |', '|---|---:|---:|---|---|---|---|---:|']
    for r in selected:
        rom,fom=r['rom'],r['fom'];rn=rom['name'] if rom else 'unattained';fn=fom['name'] if fom else 'unattained'
        rt=f"{rom['ms']:.4f}" if rom else '—';ft=f"{fom['ms']:.4f}" if fom else '—';ratio=f"{r['paired_ratio']:.5f}" if r['paired_ratio'] is not None else '—'
        lines.append(f"| {r['norm']} | {r['intervals']} | {r['target']:g} | {'empirical pass' if r['empirical_reference_budget_passed'] else 'unresolved'} | `{rn}` | `{fn}` | {rt} / {ft} | {ratio} |")
    lines+=['','![Complete-query accuracy and cost](accuracy-cost.svg)','',
        '## Where the primary Gauss error appears','',
        f"The primary corrected setting uses timestep {gauss['dt']:g}, the predeclared larger fitting budget with one start and the original loose evolution threshold. This table distinguishes an initial target miss from a miss that appears later. A configured improvement stop is not a stationary or globally optimal fit.",'',
        '| Intervals | Case | Initial complete-grid error | Later complete-grid maximum | Time of maximum |', '|---|---:|---:|---:|---:|']
    for L in sorted({r['intervals'] for r in summary}):
        name=f'rom_L{L}_fixed_gauss_ic180_dt0.005_stall0.01_starts1'
        for case in range(d['config']['cases']):
            rows=[r for r in groups[name] if r['case']==case];err=np.max([r['dense_physical_error']['fixed_initial_per_time'] for r in rows],axis=0)
            lines.append(f"| {L} | {case} | {err[0]:.8g} | {max(err[1:]):.8g} | {d['output_times'][int(np.argmax(err))]:g} |")
    diagnostics_path=run/'DIAGNOSTICS.json'
    if diagnostics_path.exists():
        diagnostics=json.loads(diagnostics_path.read_text())
        if timestep:lines+=['',f"Across the saved timestep controls, the largest absolute change in the initial field is {max(r['initial_field_max_difference'] for r in diagnostics['controls']):.6g}. The increased later error therefore occurs after this unchanged initialization.",'']
        lines+=['','The next diagnostic uses the same saved full-query fields. The near-wall band contains nodes closer to a wall than the training mesh first interior node. Its squared-error fraction locates the error; it does not independently measure a bank projection floor.','',
            '| Intervals | Case | Cold rule / dt | Initial near-wall squared-error fraction | Target | Target diagnosis |', '|---|---:|---|---:|---:|---|']
        for r in diagnostics['cases']:
            lines.append(f"| {r['intervals']} | {r['case']} | {r['cold_rule']} / {r.get('dt',.005):g} | {r['boundary_band_squared_error_fraction_per_time'][0]:.7g} | {r['target']:g} | {r['target_diagnostic']} |")
    lines+=['','## Complete measured configuration set' ,'',
        '| Configuration | Query ms | Common-grid median / worst | Complete-grid median / worst | Error outliers common / complete | Timing outliers | Failed calls |', '|---|---:|---|---|---|---:|---:|']
    for r in sorted(summary,key=lambda r:(r['intervals'],r['method'],r['ms'])):
        lines.append(f"| `{r['name']}` | {r['ms']:.4f} | {r['common']['median']:.7g} / {r['common']['worst']:.7g} | {r['dense']['median']:.7g} / {r['dense']['worst']:.7g} | {r['common']['outliers_0p01']} / {r['dense']['outliers_0p01']} | {r['timing_outliers_gt2x_case_median']} | {r['failed_invocations']} |")
    lines+=['','## Runtime components and setup','',
        'Component timings are separately synchronized diagnostic invocations with field parity against the fused query. They do not replace complete-query timing or establish kernel-launch counts.','',
        '| Intervals | Cold rule / budget / dt | Input ms | Initial fit ms | Evolution ms | Decode ms | Output ms | Staged total ms | LM attempts |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for L,rule,icb,dt,stall in sorted({(r['intervals'],r['cold_rule'],r['ic_budget'],r.get('dt',.005),r.get('stall',.01)) for r in d['component_profiles']}):
        raw=[r for r in d['component_profiles'] if (r['intervals'],r['cold_rule'],r['ic_budget'],r.get('dt',.005),r.get('stall',.01))==(L,rule,icb,dt,stall)]
        vals=[1000*np.median([r[k] for r in raw]) for k in ['input_transfer_s','cold_fit_s','evolution_s','dense_decode_s','output_transfer_s','staged_total_s']]
        lines.append(f"| {L} | {rule} / {icb} / {dt:g} | "+' | '.join(f'{v:.4f}' for v in vals)+f" | {np.median([r['lm_attempts'] for r in raw]):.1f} |")
    lines+=['','| Intervals | Sampled rank | K / R / M / m | Bank/operator setup s | Gauss setup s | Compilation/warmup s |', '|---|---:|---|---:|---:|---:|']
    for r in d['mesh_setup']:
        lines.append(f"| {r['intervals']} | {r['sampled_bank_rank_relative_1e12']} | {r['K']} / {r['R']} / {r['M']} / {r['m']} | {r['setup_seconds']:.4f} | {r['gauss_cold_setup']['setup_seconds']:.4f} | {r['compilation_warmup_seconds']:.4f} |")
    interpretation=('The timestep sweep is development selection on the same cases. Larger steps can reduce evolution work while increasing temporal error or hitting the unchanged trial budget. Consequently, larger-step field loss cannot be attributed solely to time discretization. The fitted initial field is shared across these controls and remains an accuracy limit; this sweep does not change bank representation or certify stationary optimization. The first step has no extrapolation history, so later predictor scaling alone cannot directly remove a startup cap. A separate bounded startup-step allocation or different time integration formula would target this limitation; no such change or additional job is included here.' if timestep else 'The dominant measured reduced phase is evolution. A bounded larger-timestep study is the next controlled cost test, keeping the fitted initializer, checkpoint, field family and FOM envelope fixed. Halving the timestep and tightening the evolution stopping threshold are already measured controls above; neither proves global or stationary optimization. Tighter accuracy also remains sensitive to the initial-fit error, especially on the finest mesh.')
    lines+=['',interpretation,'', '## Audit and scope','',
        f"The independent NumPy audit checked {audit['invocations_verified']} invocations across {audit['declared_configurations_verified']} configurations and {audit['full_grid_artifacts_verified']} saved complete-grid artifacts. Maximum error-recomputation discrepancies are {audit['common_grid_error_recompute_max_difference']:.6g} on the common grid and {audit['dense_error_recompute_max_difference']:.6g} on complete grids. It found {audit['fom_nonlinear_tolerance_failures']} FOM tolerance failures and {audit['nonfinite_invocations']} nonfinite invocations. {audit['full_output_repetitions_matching_first']} of {audit['invocations_verified']} recorded dense-output hashes match their corresponding first repetitions.",'',
        'Source, checkpoint and closed-output checksums are retained. Every repeated output is checked against an actual complete-grid artifact. Initial clipping and physical cases are preserved. Independent confirmation, rigorous reference bounds and per-resolution weight retraining remain open. The historical cold-only bank-floor diagnostics are not substituted for the returned rollout errors.','',
        '## Plain-language glossary','',
        '- **Intervals / complete grid / common grid:** cells per axis / every requested output node / fixed observation nodes shared by all resolutions.',
        '- **FOM / ROM / frozen checkpoint:** full spatial model / nonlinear-manifold reduced model / saved neural weights kept unchanged.',
        '- **Bank / head / IC:** learned coordinate-dependent spatial features / nonlinear map from latent coordinates to feature coefficients / initial condition.',
        '- **Newton / BiCGStab / Helmholtz preconditioning / FFT:** nonlinear iteration / iterative linear solution / inversion of the diffusion-plus-identity part to help that solve / fast Fourier transform.',
        '- **Gauss / edge / QR / cold:** fixed physical Gaussian quadrature / original mesh-dependent initial samples / orthogonal triangular factorization / initial reduced-state fitting.',
        '- **K / R / M / m / sampled rank:** latent coordinates / learned spatial features / smooth weak tests / advection quadrature nodes / independently resolved bank directions on deterministic sampled rows.',
        '- **Query ms / case median / paired FOM/ROM:** complete host-input-to-host-output latency in milliseconds / middle timing repetition for one physical case / casewise full-model time divided by reduced-model time, then a cohort median.',
        '- **Initial / later / median / worst error:** first returned field / remaining output times / middle case error / largest case error; each is divided by the corresponding initial reference norm.',
        '- **Current-relative error:** discrepancy divided by the current reference field norm, which exposes relative error as the field decays; it is a diagnostic separate from the initial-norm eligibility target.',
        '- **Same-mesh error:** discrepancy from a tightly converged full solve using the same mesh and timestep.',
        '- **dt / steps / final weak residual:** time-step size / autonomous steps between the first and last requested output / final weighted weak-equation discrepancy for each such solve; it is not a physical field-error norm.',
        '- **Budget / improvement / failed stops:** configured trial cap / small-step or relative-improvement exit / rejected or nonfinite exit. The first two do not prove stationarity.',
        '- **Empirical margin / Richardson / observed order / target / reference budget:** estimated reference error / extrapolation assuming the observed convergence rate continues / that rate inferred from three levels / requested accuracy / allowed fraction of target consumed by reference uncertainty.',
        '- **Envelope / selected / unattained:** least-cost eligible candidate including coarse FOM interpolation / chosen development configuration / no tested configuration meets the stated error condition.',
        '- **Weak evolution / sign-upwind / nonnegative quadrature:** PDE equations averaged against smooth functions / local-sign-selected differences / positive weighted sampling of advection.',
        '- **Input / initial fit / evolution / decode / output / staged total / LM attempts:** device transfer / initial latent optimization / reduced timestepping / dense reconstruction / host transfer / sum of diagnostic phases / actual Levenberg–Marquardt trial steps.',
        '- **Error / timing outliers:** physical cases above relative error 0.01 / calls taking more than twice that case’s timing median. All outliers remain in the results.',
        '- **Near-wall squared-error fraction:** share of total squared field discrepancy at nodes closer to the boundary than the first interior node of the training mesh.',
        '- **Setup / compilation / artifact / hash / parity:** reusable numerical preparation / executable preparation / saved numerical file / content fingerprint / agreement of two implementations or outputs.',
        '- **Validation / final / f64 / highest / backend:** development cases / reserved independent cases / double precision / required matrix arithmetic precision / execution device type.','']
    Path(a.output).write_text('\n'.join(lines));print(a.output)


if __name__=='__main__':main()
