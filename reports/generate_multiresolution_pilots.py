"""Generate the four-PDE pilot comparison from immutable run artifacts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPAIR = ROOT/'worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign'
FILES = {
    'heat': ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/pilot01/archive/outputs/results.json',
    'heat_review': REPAIR/'heat_pilot01_review.json',
    'burgers': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/pilot01/out/pilot.json',
    'burgers_review': REPAIR/'burgers_pilot01_review.json',
    'burgers_followup': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/pilot02/out/pilot.json',
    'burgers_followup_audit': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/pilot02/AUDIT.json',
    'burgers_cold': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/cold03/out/cold_fit.json',
    'burgers_cold_audit': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/cold03/AUDIT.json',
    'poisson': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot01/result.json',
    'poisson_review': REPAIR/'poisson_pilot01_review.json',
    'wave': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/pilot01/analysis/summary.json',
    'wave_native': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/pilot01/cluster/out/pilot/result.json',
    'wave_review': REPAIR/'wave_pilot01_accounting_review.json',
    'heat_runtime': ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/pilot02/archive/outputs/results.json',
    'heat_runtime_audit': ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/pilot02/audit.json',
    'poisson_followup': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot02/result.json',
    'poisson_followup_summary': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot02/summary.json',
    'poisson_followup_audit': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot02/audit.json',
    'heat_head': ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/pilot03/archive/outputs/results.json',
    'heat_head_review': REPAIR/'heat_pilot03_review.json',
    'heat_transfer': ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/archive/outputs/results.json',
    'heat_transfer_audit': ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/audit.json',
    'heat_transfer_review': REPAIR/'heat_transfer04_review.json',
    'poisson_kernel': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot03/result.json',
    'poisson_kernel_summary': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot03/summary.json',
    'poisson_kernel_review': REPAIR/'poisson_pilot03_review.json',
    'poisson_training': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot04/result.json',
    'poisson_training_summary': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot04/summary.json',
    'poisson_training_audit': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot04/audit.json',
    'poisson_training_review': REPAIR/'poisson_pilot04_review.json',
    'poisson_speed': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot05/result.json',
    'poisson_speed_summary': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot05/summary.json',
    'poisson_speed_audit': ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot05/audit.json',
    'poisson_speed_review': REPAIR/'poisson_pilot05_review.json',
    'burgers_rollout': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/rollout04/out/pilot.json',
    'burgers_rollout_summary': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/rollout04/SUMMARY.json',
    'burgers_rollout_audit': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/rollout04/AUDIT.json',
    'burgers_rollout_review': REPAIR/'burgers_rollout04_review.json',
    'burgers_steps': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/steps05/out/pilot.json',
    'burgers_steps_summary': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/steps05/SUMMARY.json',
    'burgers_steps_audit': ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/steps05/AUDIT.json',
    'burgers_steps_review': REPAIR/'burgers_steps05_review.json',
    'wave_dynamics': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/cluster/out/pilot/result.json',
    'wave_dynamics_summary': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/summary.json',
    'wave_dynamics_audit': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/audit.json',
    'wave_dynamics_review': REPAIR/'wave_dynamics02_review.json',
    'wave_moment': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/absorbing-moment.json',
    'wave_heads': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/k32heads03/cluster/out/pilot/result.json',
    'wave_heads_summary': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/k32heads03/analysis/summary.json',
    'wave_heads_audit': ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/k32heads03/analysis/audit.json',
    'wave_heads_review': REPAIR/'wave_k32heads03_review.json',
    'wave_heads_figures': REPAIR/'wave-k32-evolution-provenance.json',
}


def link(path, label):
    return f'[{label}](../{path.relative_to(ROOT)})'


def main():
    data = {key: json.loads(path.read_text()) for key, path in FILES.items()}
    manifest = {key: {'path': str(path.relative_to(ROOT)),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for key, path in FILES.items()}
    assert data['heat_review']['source_sha256'][str(FILES['heat'])] == manifest['heat']['sha256']
    assert data['burgers_review']['source_json_sha256'] == manifest['burgers']['sha256']
    assert data['poisson_review']['source_json_sha256'] == manifest['poisson']['sha256']
    assert data['wave']['result_sha256'] == manifest['wave_native']['sha256']
    comparisons = []

    def add(pde, n, setting, metric, rom, fom, error_key, time_key):
        rc, fc = {r['case']: r for r in rom}, {r['case']: r for r in fom}
        assert rc.keys() == fc.keys()
        comparisons.append(dict(pde=pde, intervals=n, setting=setting, metric=metric,
            cases=len(rc), rom_error_median=statistics.median(r[error_key] for r in rom),
            rom_error_worst=max(r[error_key] for r in rom),
            fom_error_worst=max(r[error_key] for r in fom),
            rom_ms=1000*statistics.median(r[time_key] for r in rom),
            fom_ms=1000*statistics.median(r[time_key] for r in fom),
            raw_paired_ratio=statistics.median(fc[c][time_key]/rc[c][time_key] for c in rc)))

    h = data['heat_review']['rows']
    for n in data['heat']['config']['evaluation_intervals']:
        rows = [{**r, 'error': r['common_grid_time_max_errors']['relative_current']} for r in h if r['intervals'] == n]
        add('Heat', n, 'CN dt=0.025', 'Current L2',
            [r for r in rows if r['method'] == 'rom_cn_dt0.025'],
            [r for r in rows if r['method'] == 'fom_dst_exact_time'], 'error', 'query_median_seconds')

    b = {r['configuration']: r['cases'] for r in data['burgers_review']['configurations']}
    for setup in data['burgers']['mesh_setup']:
        n = setup['intervals']
        add('Burgers', n, 'dt=0.005; stall=0.001', 'Initial L2',
            b[f'rom_L{n}_dt0.005_stall0.001'], b[f'fom_L{n}_out{n}_dt0.005_ntol0.003'],
            'initial_normalized_max', 'query_median_seconds')

    p = data['poisson']
    for n in p['config']['intervals']:
        result = {}
        for arm, tau in [('rom', .01), ('dst', None)]:
            groups = defaultdict(list)
            for row in p['rows']:
                if row['intervals'] == n and row['arm'] == arm and row['tau'] == tau:
                    groups[row['case']].append(row)
            result[arm] = [dict(case=case, error=max(r['physical_error'] for r in rows),
                                seconds=statistics.median(r['total_seconds'] for r in rows))
                           for case, rows in groups.items()]
        add('Poisson', n, 'tau=0.01', 'Steady L2', result['rom'], result['dst'], 'error', 'seconds')

    w = data['wave']
    for boundary in w['config']['boundaries']:
        for n in w['config']['meshes']:
            pick = lambda method, setting: next(r['cases'] for r in w['groups']
                if r['boundary'] == boundary and r['intervals'] == n and
                r['method'] == method and r['setting'] == setting)
            add('Reflective wave' if boundary == 'dirichlet' else 'Absorbing wave', n,
                'dt=0.0025', 'Initial wave state', pick('rom', .0025),
                pick('dst', 0.) if boundary == 'dirichlet' else pick('rk4', .45),
                'worst_required_physical_error', 'query_median')

    lines = ['# Multiresolution development pilots: accuracy and complete-query cost', '',
        'This report covers frozen-network mesh-transfer pilots, bounded solver changes and controlled training refinements of the current separable NM-ROM against efficient FOM solvers. '
        'The numbers are provisional development evidence; independent confirmation and the full resolution study remain open.', '',
        'The tested frozen decoders produce solutions on new meshes, but the completed nonlinear-ROM studies have not established a complete-query advantage over efficient FOMs. '
        'Increasing resolution does not reliably reduce ROM error in these pilots. Further work must address representation, initialization or reduced-solver cost, according to the PDE.', '',
        'The older ViT + CP architecture is excluded. Waves use only the fresh verified lineage. '
        'The final cohorts remain unopened. Different rows use different physical error definitions, stated below; they must not be ranked as a common cross-PDE accuracy score.', '',
        '## Primary configurations from the first pilots', '',
        'These are explicit representative settings, not a claim that every row is the cheapest configuration at a qualified accuracy target. '
        'Burgers uses a same-mesh FOM here; its native report also includes a coarse-mesh output envelope. '
        'Poisson uses the development-selected stopping setting shown. Raw cost ratios are separate from physical-accuracy qualification.', '',
        '| PDE | Intervals | ROM setting | Error metric | Cases | ROM median / worst error (%) | FOM worst error (%) | ROM / FOM query (ms) | Raw paired FOM/ROM |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['pde']} | {row['intervals']} | {row['setting']} | {row['metric']} | {row['cases']} | "
            f"{100*row['rom_error_median']:.4g} / {100*row['rom_error_worst']:.4g} | {100*row['fom_error_worst']:.4g} | "
            f"{row['rom_ms']:.5g} / {row['fom_ms']:.5g} | {row['raw_paired_ratio']:.4g} |")
    lines += ['', 'All timings include the supplied host field, initialization/source projection, solve or evolution, '
        'and requested host field outputs. Each cost is the cohort median of per-case repetition medians. '
        'The paired ratio is the median of per-case FOM/ROM cost ratios. Each pair was measured in one job on one GPU; '
        'raw wall times must not be compared across PDE jobs as a hardware-normalized ranking.', '',
        '**Heat:** errors are maxima over output times relative to the current reference norm, evaluated on the common observation grid. '
        'This is a newly verified, restricted single-bump development family. It does not yet cover the broader heat use case.', '',
        '**Burgers:** errors use the initial reference norm and the common observation grid. The first reference refinement estimate leaves target qualification unresolved. '
        'Cold-fit budget exits are retained in the native records and independent audit; an observed small error is not proof of a stationary initial fit. '
        'The follow-ups below test more starting guesses, a finer reference and the initial fitting objective.', '',
        '**Poisson:** errors are relative solution norms for each steady source on the common observation grid. Tighter stationary solves still leave a worst-case error floor. '
        'The reference has empirical refinement evidence; development qualification is not an independent final-cohort result.', '',
        '**Waves:** the reported metric is the maximum of displacement, velocity and energy-state errors on the common observation grid. '
        'Displacement is divided by the initial displacement L2 norm; velocity and energy-state error use the initial phase-energy scale $\\sqrt{2E(0)}$. '
        'Here $E(0)$ is the initial wave energy, including displacement gradients and velocity; the initial velocity alone can be zero. '
        'Absorbing errors relative to the small remaining field are substantially larger and must also be reported. '
        'The time-step pair is resolved for these cases, so a smaller step does not remedy the observed error.', '',
        '## Evidence and limitations', '',
        '| PDE | Numerical source commit | GPU job | Native findings |', '|---|---|---|---|']
    findings = {
        'Heat': (data['heat']['source_manifest']['source_commit'], data['heat']['metadata']['job_id'],
                 ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/HEAT-PILOT-NOTES.md'),
        'Burgers': (data['burgers']['commit'], data['burgers']['job_id'],
                    FILES['burgers']),
        'Poisson': (p['provenance']['commit'], p['provenance']['job_id'],
                    ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot01/FINDINGS.md'),
        'Waves': (w['provenance']['source_commit'], w['provenance']['job_id'],
                  ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/pilot01/analysis/FINDINGS.md'),
    }
    for pde, (commit, job, path) in findings.items():
        lines.append(f'| {pde} | `{commit}` | {job} | {link(path, "Findings / source records")} |')
    lines += ['', 'All first-pilot outputs were checksum-collected and their exact remote job directories removed. '
        'Numerical checks cover GPU execution, precision, relevant operators, reference refinement and state advancement. '
        'Root independently recomputed the preserved heat, Burgers and Poisson field errors. '
        'The wave root review checks source design, paired accounting and artifact hashes; it is not an independent full-grid regeneration of every wave metric.', '',
        'The rigorous reference-bound field remains unspecified where only empirical refinement is available. '
        'No paper-wide speedup, optimal capacity, optimized training cost, broad-family robustness, or independent confirmation is established.', '',
        '## Next experiments justified by the diagnostics', '',
        '- **Burgers:** larger steps partly reduce cost but also expose iteration caps, especially at startup. A later bounded startup-solve or time-formula control should separate those effects before further architectural changes. The existing latent extrapolator is already present; adding it again is not a new intervention.',
        '- **Heat:** both frozen expanded-coverage endpoints now pass the wider mesh ladder and fresh inputs in the restricted family. Complete output and initialization costs still matter. Broader inputs, tighter accuracy and separately measured per-resolution training are the remaining scientific controls; any runtime change must keep the charged coarse-FOM envelope.',
        '- **Poisson:** relative loss improves the worst development sources, and field-based initialization gives a modest cost reduction. Equivalent source projections give no consistent gain. The remaining accuracy floor and nonlinear solve cost require further representation or solver controls; expanded coverage at matched updates is not uniformly beneficial.',
        '- **Waves:** the larger nonlinear heads greatly improve reflective accuracy but remain costly and leave late absorbing error. Separate constant-test and initial-moment interventions are justified as later controls by the conservation diagnostic, without assuming they solve the full problem. A coarse-FOM resolution envelope is still needed before promoting any raw linear-control timing ratio.', '',
        'These are directions for a later bounded round; every GPU job in this report is complete. The complete study still needs wider mesh ladders for the remaining PDEs, separately labeled per-resolution training, '
        'independent data/training repeats, validation-selected settings and sealed final evaluation.', '',
        '## Visual artifacts', '',
        link(REPAIR/'heat-pilot01-accuracy-cost.png', 'Heat accuracy and complete-query cost')+'. '+
        link(REPAIR/'dirichlet-wave-evolution-case1.png', 'Reflective wave evolving')+'. '+
        link(REPAIR/'absorbing-wave-evolution-case1.png', 'Absorbing wave evolving')+'.', '',
        'The wave still sequences show reference displacement, predicted displacement and absolute difference with fixed scales. '
        'They use saved fields from an actual finer-mesh ROM solve; the display resolution is labeled. '
        'Displacement pictures do not replace the velocity, energy or vanishing-field diagnostics. '
        'PDF exports and figure source/provenance are beside the images.', '',
        link(REPAIR/'dirichlet-wave-k32-evolution-case1.png', 'Larger-head reflective wave evolving')+'. '+
        link(REPAIR/'absorbing-wave-k32-evolution-case1.png', 'Larger-head absorbing wave evolving')+'. '+
        'These updated sequences compare the reference, the earlier smaller head and the first declared larger-head seed within the same audited run. '
        'The same development-case index is used as in the original still sequences. Small displacement discrepancies can coexist with larger velocity or gradient-based energy errors; fixed scales can also hide late absorbing relative error.', '',
        '## Reproduction', '',
        'Run `/home/tahmid/Dev/.venv/bin/python reports/generate_multiresolution_pilots.py` from a checkout with the recorded experiment worktrees. '
        'The adjacent JSON manifest identifies every source artifact by content hash. All numerical table values are generated; none are hand-entered.', '',
        '## Plain-language glossary', '',
        '- **PDE / FOM / NM-ROM / ROM:** partial differential equation / full spatial solver / nonlinear-manifold reduced solver / reduced solver.',
        '- **Frozen / intervals / cases:** unchanged network weights / grid cells along one axis / distinct physical inputs. Timing repetitions are not additional cases.',
        '- **ROM setting / dt / CN / stall / tau:** chosen solver configuration / time step / Crank–Nicolson time formula / relative improvement stopping rule / requested weak-residual reduction.',
        '- **Current L2 / initial L2 / steady L2:** field error divided by the current reference norm / initial reference norm / steady reference solution norm.',
        '- **Initial wave state / energy-state error:** displacement normalized by its initial L2 norm and velocity/energy error normalized by the initial phase-energy scale / error combining displacement gradients with velocity.',
        '- **Common observation grid / same-grid norm:** fixed physical sampling locations shared by different query resolutions / error measured over the full grid at the stated resolution. These measures need not coincide.',
        '- **Median / worst / query ms / paired ratio:** middle case result / largest case error / complete input-to-output milliseconds / per-case FOM time divided by ROM time, then a cohort median.',
        '- **Bank / head / latent / weak test mode:** learned spatial features / their nonlinear coefficient map / compressed state coordinates / smooth function averaging the PDE equation.',
        '- **Cold start / stationary / budget exit:** initial reduced-state fitting / meeting a local derivative convergence check / exhausting allowed iterations.',
        '- **DST / coarse-grid envelope:** direct sine-transform solution / least-cost qualifying full solve allowing fewer cells and charging interpolation to the requested output.',
        '- **Reference refinement / uncertainty / qualification:** comparing finer trusted solves / remaining reference error / meeting accuracy and numerical-validity requirements with that uncertainty included.',
        '- **Development / validation / sealed final / provisional:** preliminary experiment data / data used to select settings / untouched independent confirmation data / evidence with the stated limitations.',
        '- **Compiled / segmented / projection / compression:** one prepared executable / separately launched stages / representing a field in a spatial span / constraining that representation through fewer latent coordinates.',
        '- **Strict / relaxed / gradient tolerance / field drift:** more demanding stopping control / less demanding stopping setting / required smallness of the objective derivative / change from the strict-control trajectory divided by the current reference norm.',
        '- **Paired improvement / requested modes / retained modes:** strict-control time divided by changed-method time, summarized across cases / desired minimum number of weak tests / actual number when tied sine eigenmodes are retained together.',
        '- **Edge Gram / edge QR / midpoint / Gauss / full-grid QR:** fitting on grid-dependent edge-inclusive samples using normal equations / a stable factorization on those same samples / fixed equally spaced physical midpoints / fixed weighted Gaussian quadrature points / least-squares fitting using every grid value.',
        '- **Bank projection floor / best-recorded fit / Richardson estimate:** smallest error possible in the unrestricted linear feature span / best fit found by the tested nonlinear searches, without proving optimality / remaining reference error estimated by assuming observed refinement rates continue.',
        '- **Envelope / additive empirical margin / passed reference budget:** least-cost tested method meeting the development target / estimated reference error added to measured ROM or FOM error / that estimate also lies below the predeclared fraction of the target. Empirical passage is not a rigorous certificate.',
        '- **Commit / manifest / checksum:** saved source revision / inventory of source artifacts / content fingerprint checking exact file bytes.', '']
    lines[-1:-1] = [
        '- **Endpoint / minibatch seed / per-snapshot relative squared-error loss:** saved weights after the declared training budget / random seed selecting training examples per update / squared reconstruction error divided by that snapshot\'s squared field norm.',
        '- **Initialization library / gate / factorial:** stored training codes used to start a solve / a predeclared requirement for continuing an experiment / crossing independently varied choices to distinguish their effects.',
        '- **Gauss–Jordan / backward error / fallback / solver counter / replay:** elimination for the small latent linear system / residual of the computed linear solution relative to its data / guarded use of the original solver / count of optimization steps or evaluations / instrumented recomputation of a saved solve.',
        '- **Affine / phase dimension / tangent velocity / normal force / curvature:** linear map plus a constant offset / displacement and velocity coordinate count / velocity representable by local decoder derivatives / weak acceleration outside those derivative directions / acceleration contributed by the bending decoder map.',
        '- **Moment / invariant / constant test / drift:** weighted global combination of displacement and velocity / quantity the discrete equations preserve / spatially constant function used to test those equations / change from a method\'s own initial value.',
        '- **First-step budget exit / predictor:** iteration limit reached at the first evolution step / proposed next latent state extrapolated from earlier states and checked by the weak residual.',
        '- **Timing outlier:** a repetition longer than twice its own case/configuration median in the latest-results table. It remains in all reported calculations.',
    ]
    followup_lines, followup_values = followups(data, manifest)
    development_lines, development_values = continued_development(data, manifest)
    followup_lines += development_lines
    followup_values.update(development_values)
    overview_lines, overview_values = latest_overview(data)
    where = lines.index('## Primary configurations from the first pilots')
    lines[where:where] = overview_lines
    where = lines.index('## Next experiments justified by the diagnostics')
    lines[where:where] = followup_lines
    (HERE/'2026-09-07-multiresolution-pilots.md').write_text('\n'.join(lines))
    (HERE/'2026-09-07-multiresolution-pilots.json').write_text(json.dumps({'artifacts': manifest, 'latest_comparisons': overview_values, 'comparisons': comparisons, 'followups': followup_values}, indent=2)+'\n')


def followups(data, manifest):
    heat, ha = data['heat_runtime'], data['heat_runtime_audit']
    poisson, ps, pa = data['poisson_followup'], data['poisson_followup_summary'], data['poisson_followup_audit']
    assert heat['complete'] and ha['passed'] and poisson['complete'] and pa['passed']
    assert heat['checkpoint_sha256'] == data['heat']['checkpoint_sha256']
    assert poisson['checkpoint_sha256'] == data['poisson']['checkpoint_sha256']
    assert ps['source_sha256'] == manifest['poisson_followup']['sha256']
    values = {'heat_runtime': [], 'poisson_modes': []}
    lines = ['## Bounded improvements with unchanged network weights', '',
        'These follow-ups retain the first-pilot physical cases and checkpoints. They provide development evidence about specific numerical changes; no new training or final-cohort evaluation is included.', '',
        '### Heat: less stringent stopping reduces cost', '',
        f"Source `{ha['source_commit']}`, job `{heat['metadata']['job_id']}`. The native audit verifies {ha['invocations']} timed invocations; compiled and modular fields, latent states and iteration counters agree at each matched tolerance. "
        'The original strict control is measured again in the same job, so the improvement does not compare clocks across jobs.', '',
        '| Intervals | Strict modular ms | Relaxed compiled ms | Paired improvement | FOM ms | ROM worst current error (%) | Maximum field drift |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n in sorted({g['intervals'] for g in ha['groups']}):
        select = lambda method: next(g for g in ha['groups'] if g['intervals'] == n and g['method'] == method)
        strict = select('rom_modular_gtol1e-09')
        relaxed = select('rom_compiled_gtol1e-05')
        fom = select('fom_dst_exact_time')
        assert relaxed['all_valid'] and relaxed['all_parity'] and relaxed['drift_within_predeclared_ceiling']
        case_time = lambda g: {c['case']: statistics.median(r['query_seconds'] for r in c['repetitions']) for c in g['cases']}
        st, rt = case_time(strict), case_time(relaxed)
        value = dict(intervals=n, strict_ms=1000*strict['median_query_seconds'],
                     relaxed_ms=1000*relaxed['median_query_seconds'],
                     paired_case_median_improvement=statistics.median(st[c]/rt[c] for c in st),
                     fom_ms=1000*fom['median_query_seconds'], worst_current_error=relaxed['error_worst'],
                     maximum_field_drift=max(r['field_drift_from_strict'] for c in relaxed['cases'] for r in c['repetitions']))
        values['heat_runtime'].append(value)
        lines.append(f"| {n} | {value['strict_ms']:.5g} | {value['relaxed_ms']:.5g} | {value['paired_case_median_improvement']:.4g} | {value['fom_ms']:.5g} | {100*value['worst_current_error']:.5g} | {value['maximum_field_drift']:.6g} |")
    lines += ['', f"The relaxed normalized-gradient tolerance is `{max(heat['settings']['gradient_tolerances'])}`; the strict tolerance is `{min(heat['settings']['gradient_tolerances'])}`. "
        f"Field drift is relative to the current reference norm and stays below the predeclared `{ha['drift_ceiling']}` ceiling. "
        'Fewer nonlinear iterations account for most of the gain; compiling the query alone gives a smaller improvement. The FOM remains faster, and the head reconstruction error remains.', '',
        link(ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/HEAT-RUNTIME-NOTES.md', 'Complete heat runtime findings')+'.', '',
        '### Poisson: more weak modes do not remove the bank limitation', '',
        f"Source `{pa['source_commit']}`, job `{pa['job_id']}`. The native audit recomputes {pa['row_count']} timed errors and verifies the compiled-query parity controls. "
        'The following table uses stationary solves with the early residual-reduction stop disabled.', '',
        '| Intervals | Requested / retained weak modes | Worst physical error (%) | Compiled query ms | Direct FOM ms |',
        '|---|---:|---:|---:|---:|']
    for row in ps['summaries']:
        if row['arm'] != 'rom_fused' or row['tau'] != 0:
            continue
        assert row['nonstationary_count'] == 0 and row['invalid_count'] == 0
        baseline = next(g for g in ps['summaries'] if g['intervals'] == row['intervals'] and g['arm'] == 'dst')
        value = dict(intervals=row['intervals'], requested_modes=row['requested_modes'], retained_modes=row['retained_modes'],
                     worst_physical_error=row['physical_max'], rom_ms=1000*row['latency_seconds'], fom_ms=1000*baseline['latency_seconds'])
        values['poisson_modes'].append(value)
        lines.append(f"| {value['intervals']} | {value['requested_modes']} / {value['retained_modes']} | {100*value['worst_physical_error']:.5g} | {value['rom_ms']:.5g} | {value['fom_ms']:.5g} |")
    lines += ['', '| Intervals | Worst full-bank projection error (%) | Worst best-recorded head-fit error (%) |', '|---|---:|---:|']
    for n in sorted({r['intervals'] for r in ps['oracles']}):
        rows = [r for r in ps['oracles'] if r['intervals'] == n]
        lines.append(f"| {n} | {100*max(r['full_bank_same_grid_error'] for r in rows):.5g} | {100*max(r['best_same_grid_error'] for r in rows):.5g} |")
    lines += ['', 'These reference-only reconstruction diagnostics use the same-grid field norm and cannot initialize a deployed query. '
        'The unrestricted bank already has a difficult-source error larger than the next all-case target; more weak modes cannot repair missing spatial directions. '
        'Compiling the unchanged solver gives only small timing changes. A stronger bank/training study and a specialized small-system linear solve remain distinct next tests.', '',
        link(ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot02/FINDINGS.md', 'Complete Poisson follow-up findings')+'.', '']
    burgers_lines, burgers_values = burgers_followups(data, manifest)
    lines += burgers_lines
    values.update(burgers_values)
    steps_lines, steps_values = burgers_timesteps(data, manifest)
    lines += steps_lines
    values.update(steps_values)
    lines += ['All cost ratios in this report use ratios of per-case timing medians before the cohort median. '
        'Some native exploratory reports also retain medians of per-repetition ratios or select configurations by the pooled median across all repetitions under an explicit different label; those statistics are not interchangeable.', '']
    return lines, values


def burgers_followups(data, manifest):
    pilot, audit = data['burgers_followup'], data['burgers_followup_audit']
    cold, ca = data['burgers_cold'], data['burgers_cold_audit']
    assert pilot['complete'] and cold['complete']
    assert audit['source_sha256'] == manifest['burgers_followup']['sha256']
    assert ca['source_sha256'] == manifest['burgers_cold']['sha256']
    assert pilot['checkpoint_sha256'] == cold['checkpoint_sha256'] == data['burgers']['checkpoint_sha256']
    assert pilot['physical_cases'] == cold['physical_cases'] == data['burgers']['physical_cases']
    assert audit['output_checksums_verified'] and ca['output_checksums_verified']
    assert audit['reference_solvers_converged'] and ca['checkpoint_and_physical_cases_match_pilot']
    cases = set(range(len(pilot['physical_cases'])))
    reference = {r['case']: r['conservative_difference_sum'] for r in pilot['reference_uncertainty']}
    for row in pilot['reference_order_audit']:
        assert row['asymptotic_decrease_observed']
        reference[row['case']] = max(reference[row['case']], row['empirical_richardson_estimate'])
    grouped = defaultdict(list)
    for row in pilot['invocations']:
        grouped[row['name']].append(row)
    # Declared pilot.py search at the recorded source commit, independent of which rows survived.
    expected_names = set()
    for n in map(int, pilot['config']['meshes'].split(',')):
        for dt in (.005, .0025):
            for stall in (.001, .01):
                for starts in map(int, pilot['config']['ic_starts'].split(',')):
                    expected_names.add(f'rom_L{n}_dt{dt}_stall{stall}_starts{starts}')
        for grid in (g for g in (128, 256, 512, 1024) if g <= n):
            for dt, ntol in ((.01, .01), (.01, .003), (.005, .01), (.005, .003), (.0025, .003)):
                expected_names.add(f'fom_L{grid}_out{n}_dt{dt}_ntol{ntol}')
    assert set(grouped) == expected_names
    assert len(pilot['invocations']) == audit['invocations_verified'] == len(expected_names)*len(cases)*pilot['config']['reps']
    groups = []
    for name, rows in grouped.items():
        assert {(r['case'], r['rep']) for r in rows} == {(c, r) for c in cases for r in range(pilot['config']['reps'])}
        assert len(rows) == len(cases)*pilot['config']['reps']
        times = {c: statistics.median(r['seconds'] for r in rows if r['case'] == c) for c in cases}
        valid = all(r['finite'] and (r['nonlinear_tolerance_satisfied'] if r['method'] == 'fom'
                    else r['ic_reason'] != 3 and 3 not in r['stop_reasons']) for r in rows)
        groups.append(dict(name=name, method=rows[0]['method'], intervals=rows[0]['output_intervals'],
                           valid=valid, times=times, ms=1000*statistics.median(times.values()),
                           error=max(r['physical_error']['fixed_initial_max'] for r in rows),
                           empirical_error_margin=max(r['physical_error']['fixed_initial_max']+reference[r['case']] for r in rows)))
    values = {'burgers_envelope': [], 'burgers_initial_fit': [], 'burgers_empirical_reference_max': max(reference.values())}
    lines = ['### Burgers: a finer reference and a resolution-dependent initial-fit defect', '',
        f"Full-query source `{pilot['commit']}`, job `{pilot['job_id']}`. The native audit independently recomputes {audit['invocations_verified']} invocation errors from preserved observation fields. "
        f"The largest empirical reference estimate is `{max(reference.values()):.7g}`, allowing development checks at the targets shown; stricter targets remain unresolved. "
        'The additive margin uses the larger of the spatial-plus-time difference and the Richardson estimate for each case. The initial analytic norm is fixed. '
        'These are empirical development checks, not rigorous reference bounds.', '',
        '| Output intervals | Target (%) | Selected ROM | Selected FOM envelope | ROM / FOM ms | Paired FOM/ROM |',
        '|---|---:|---|---|---:|---:|']
    for n in sorted({r['intervals'] for r in pilot['mesh_setup']}):
        for target in (.1, .05):
            assert max(reference.values()) <= .1*target
            selected = {}
            for method in ('rom', 'fom'):
                eligible = [g for g in groups if g['method'] == method and g['intervals'] == n and g['valid']
                            and g['empirical_error_margin'] <= target]
                selected[method] = min(eligible, key=lambda g: (g['ms'], g['name'])) if eligible else None
            rom, fom = selected['rom'], selected['fom']
            ratio = statistics.median(fom['times'][c]/rom['times'][c] for c in cases) if rom and fom else None
            values['burgers_envelope'].append(dict(intervals=n, target=target, rom=rom, fom=fom, paired_ratio=ratio,
                qualification='empirical reference budget passed; independent confirmation absent',
                rigorous_reference_bound=None))
            rom_name = f"`{rom['name']}`" if rom else 'Target unattained'
            fom_name = f"`{fom['name']}`" if fom else 'Target unattained'
            rom_time = f"{rom['ms']:.5g}" if rom else '—'
            fom_time = f"{fom['ms']:.5g}" if fom else '—'
            ratio_text = f'{ratio:.4g}' if ratio is not None else '—'
            lines.append(f'| {n} | {100*target:g} | {rom_name} | {fom_name} | {rom_time} / {fom_time} | {ratio_text} |')
    lines += ['', 'Selections minimize the cohort median of case-median query times, including coarse-FOM interpolation to the requested output. '
        'The native Burgers report also supplies selections using pooled repetition medians; closely timed FOM choices can differ. '
        f"All {audit['ic_budget_exits_across_all_rom_invocations']} cold-fit budget exits remain visible across the full pilot search. "
        'Configured budget or small-improvement exits are explicit early stops, not proof of stationary fitting. '
        'More initial guesses did not remove the error increase with resolution, and no tested envelope establishes a ROM advantage.', '',
        f"Cold-fit-only source `{cold['commit']}`, job `{cold['job_id']}`. The audit checks {ca['declared_fits_verified']} declared fits, retained fields, unchanged cases and checkpoint bytes. "
        'The following full-grid initial-error diagnostics are provisional: finer-grid full norms and bank floors were computed in the GPU job, '
        'but only the common-grid fields were retained for independent reconstruction. These fits do not include a PDE rollout.', '',
        '| Intervals | Edge Gram (%) | Edge QR (%) | Fixed midpoint (%) | Fixed Gauss (%) | Full-grid QR (%) | Unrestricted bank floor (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    budget, starts = max(r['budget'] for r in cold['rows']), max(r['starts'] for r in cold['rows'])
    rules = ('edge_gram', 'edge_qr', 'fixed_midpoint', 'fixed_gauss', 'full_qr')
    for n in sorted({r['intervals'] for r in cold['rows']}):
        rows = [r for r in cold['rows'] if r['intervals'] == n and r['budget'] == budget and r['starts'] == starts]
        errors = {}
        for rule in rules:
            chosen = [r for r in rows if r['rule'] == rule]
            assert {r['case'] for r in chosen} == cases
            errors[rule] = max(r['relative_full_grid_error'] for r in chosen)
        floor = max(r['unrestricted_bank_floor'] for r in rows)
        values['burgers_initial_fit'].append(dict(intervals=n, budget=budget, starts=starts,
            worst_full_grid_errors=errors, worst_unrestricted_bank_floor=floor))
        lines.append(f'| {n} | '+ ' | '.join(f'{100*errors[rule]:.5g}' for rule in rules)+f' | {100*floor:.5g} |')
    lines += ['', f"The table uses a budget of `{budget}` iterations and `{starts}` training-code starting guesses, reporting the worst physical case. "
        'The same edge samples give essentially the same result under Gram and QR fitting; fixed physical midpoint or Gauss sampling greatly reduces the finer-grid fitting error. '
        'The remaining bank floor also grows with resolution. Thus both the changing sampled objective and a real frozen-bank representation loss matter. '
        'The earlier explanation based only on optimizer starting guesses is withdrawn. '
        'The clipped initial-condition family has not changed. These are best-recorded nonlinear fits, not stationary or globally optimal certificates. '
        'Fixed physical field sampling with charged interpolation is an initialization method, not strong-form PDE collocation. '
        'Corrected-initializer rollouts were unmeasured at this stage; the continued-development section below now reports them.', '',
        link(ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md', 'Complete Burgers follow-up findings')+'.', '']
    return lines, values


def continued_development(data, manifest):
    heat, hr = data['heat_head'], data['heat_head_review']
    ps, pr = data['poisson_kernel_summary'], data['poisson_kernel_review']
    assert heat['complete'] and ps['audit']['passed']
    assert hr['source_json_sha256'] == manifest['heat_head']['sha256']
    assert ps['source_sha256'] == pr['source_json_sha256'] == manifest['poisson_kernel']['sha256']
    assert hr['frozen_spatial_parameters_match'] and hr['declared_endpoint_and_repetition_counts_match']
    settings = heat['settings']
    models = {r['name']: r for r in heat['models']}
    values = {'heat_head_reconstruction': [], 'heat_head_rollout': [], 'poisson_kernel': []}
    lines = ['## Continued development: controlled training and solver changes', '',
        '### Heat: broader training coverage improves accuracy with the unchanged spatial bank', '',
        f"Source `{hr['source_commit']}`, job `{hr['job_id']}`. The spatial bank is exactly unchanged. "
        f"Each refined head receives {settings['updates']} updates, with the same architecture and per-snapshot relative squared-error loss. "
        'The original and expanded training cohorts are crossed with the recorded minibatch seeds. All final checkpoints are retained; none was selected by a validation training loss.', '',
        '| Head endpoint | Training trajectories | Initial median / worst relative error (%) | Later worst reconstruction error (%) | Initial-fit gate |',
        '|---|---:|---:|---:|---|']
    for row in hr['reconstruction']:
        assert row['nonstationary_snapshot_fits'] == 0
        value = dict(model=row['model'], training_trajectories=models[row['model']]['details']['trajectories'],
            initial_median=statistics.median(c['initial_error'] for c in row['cases']),
            initial_worst=max(c['initial_error'] for c in row['cases']),
            later_worst=max(c['later_worst_error'] for c in row['cases']),
            gate_passed=row['rollout_gate_passed'])
        values['heat_head_reconstruction'].append(value)
        lines.append(f"| `{value['model']}` | {value['training_trajectories']} | {100*value['initial_median']:.5g} / {100*value['initial_worst']:.5g} | {100*value['later_worst']:.5g} | {'Pass' if value['gate_passed'] else 'Fail'} |")
    lines += ['', f"The predeclared gate requires every initial fit to be stationary and below {100*settings['rollout_gate']['initial_worst_max']:g}% error. "
        'Both expanded-cohort endpoints pass; continuing optimization on only the original cohort does not. '
        'Each endpoint uses its own training-code initialization library. This demonstrates an improvement in the combined training-and-initialization procedure; '
        'it does not isolate better weights from better starting codes. A stationary fit is a local convergence result, not a proof of global optimality.', '',
        'Full rollouts below use the already tested relaxed solver tolerance. Errors are relative to the current reference on the common observation grid; the same restricted physical family and development cases are retained.', '',
        '| Intervals | Head endpoint | Worst rollout error (%) | ROM / direct FOM query (ms) | Paired FOM/ROM | Nonstationary initial fits / steps |',
        '|---|---|---:|---:|---:|---:|']
    gtol = max(settings['rollout_gradient_tolerances'])
    for n in settings['rollout_intervals']:
        baseline = {r['case']: r for r in hr['rows'] if r['intervals'] == n and r['model'] == 'fom'}
        for model in ['frozen'] + heat['qualified_refinements']:
            rows = [r for r in hr['rows'] if r['intervals'] == n and r['method'] == f'{model}_gtol{gtol}']
            assert {r['case'] for r in rows} == baseline.keys()
            value = dict(intervals=n, model=model, gradient_tolerance=gtol,
                worst_current_error=max(r['worst_current_error'] for r in rows),
                rom_ms=1000*statistics.median(r['query_median_seconds'] for r in rows),
                fom_ms=1000*statistics.median(r['query_median_seconds'] for r in baseline.values()),
                paired_fom_over_rom=statistics.median(baseline[r['case']]['query_median_seconds']/r['query_median_seconds'] for r in rows),
                initial_nonstationary=sum(r['initial_nonstationary_repetitions'] for r in rows),
                steps_nonstationary=sum(r['nonstationary_steps_all_repetitions'] for r in rows))
            values['heat_head_rollout'].append(value)
            lines.append(f"| {n} | `{model}` | {100*value['worst_current_error']:.5g} | {value['rom_ms']:.5g} / {value['fom_ms']:.5g} | {value['paired_fom_over_rom']:.4g} | {value['initial_nonstationary']} / {value['steps_nonstationary']} |")
    lines += ['', 'The accuracy gain survives evolution on both query meshes, but the efficient direct FOM remains faster. '
        f"The root independently recomputed all {hr['verified_timed_invocations']} timed outputs; the largest metric disagreement was `{hr['maximum_metric_disagreement']:.3g}`. "
        f"The empirical spectral-reference difference is `{heat['reference_evidence']['empirical_relative_delta']:.7g}`; no rigorous bound is supplied. "
        'Saved offline refinement durations include compilation and host work without a dedicated warm-timing protocol; they are observed costs, not a training-speed comparison.', '',
        link(ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/HEAT-HEAD-NOTES.md', 'Complete heat training findings')+'. '+
        link(ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/figures/heat-head.png', 'Heat refinement figure')+'.', '',
        '### Poisson: a guarded small-system solver gives a modest query improvement', '',
        f"Source `{pr['source_commit']}`, job `{pr['job_id']}`. The unchanged decoder and nonlinear objective are tested with a specialized Gauss–Jordan solve. "
        'Its numerical guard and generic-solver fallback are inside the timed query. The table pairs it with the generic compiled solver and the direct FOM in the same job.', '',
        '| Intervals | Requested / retained modes | tau | Generic / specialized / FOM query (ms) | Paired generic/specialized | Worst physical error (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in ps['paired_ratios']:
        selected = {arm: next(g for g in ps['summaries'] if g['intervals'] == row['intervals'] and g['arm'] == arm
            and (arm == 'dst' or (g['requested_modes'] == row['requested_modes'] and g['tau'] == row['tau'])))
            for arm in ('rom_fused', 'rom_gj', 'dst')}
        assert all(g['invalid_count'] == 0 for g in selected.values())
        if row['tau'] == 0:
            assert selected['rom_gj']['nonstationary_count'] == 0
        value = {**row, 'generic_ms': 1000*selected['rom_fused']['latency_seconds'],
            'specialized_ms': 1000*selected['rom_gj']['latency_seconds'],
            'fom_ms': 1000*selected['dst']['latency_seconds'], 'worst_physical_error': selected['rom_gj']['physical_max']}
        values['poisson_kernel'].append(value)
        lines.append(f"| {row['intervals']} | {row['requested_modes']} / {row['retained_modes']} | {row['tau']:g} | {value['generic_ms']:.5g} / {value['specialized_ms']:.5g} / {value['fom_ms']:.5g} | {row['rom_fused_over_gj']:.5g} | {100*value['worst_physical_error']:.5g} |")
    lines += ['', 'The specialized solver preserves physical accuracy but does not beat the FOM. The residual-reduction stop may finish before stationarity; '
        'the stationary controls disable that stop.', '',
        f"Root independently checks {pr['verified_invocations']} invocation metrics and {pr['replayed_normal_systems']} saved linear systems. "
        f"The largest metric difference is `{pr['maximum_metric_disagreement']:.7g}` and the largest linear backward error is `{pr['maximum_linear_backward_error']:.7g}`. "
        f"There are {ps['audit']['specialized_timed_fallbacks']} timed fallbacks, {ps['agreement']['changed_counters']} solver-counter changes and {ps['agreement']['changed_reasons']} stop-reason changes. "
        f"The {pr['replay_counter_mismatches']} instrumented-replay counter mismatches are retained; exact trajectory identity is not claimed. "
        'These replay checks are numerical evidence, not timing measurements.', '',
        link(ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot03/FINDINGS.md', 'Complete Poisson solver findings')+'.', '']
    burgers_lines, burgers_values = corrected_burgers_rollout(data, manifest)
    lines += burgers_lines
    values.update(burgers_values)
    wave_lines, wave_values = wave_dynamics(data, manifest)
    lines += wave_lines
    values.update(wave_values)
    moment_lines, moment_values = wave_moment(data, manifest)
    lines += moment_lines
    values.update(moment_values)
    training_lines, training_values = poisson_training_factorial(data, manifest)
    lines += training_lines
    values.update(training_values)
    heads_lines, heads_values = wave_larger_heads(data, manifest)
    lines += heads_lines
    values.update(heads_values)
    speed_lines, speed_values = poisson_source_initialization(data, manifest)
    lines += speed_lines
    values.update(speed_values)
    transfer_lines, transfer_values = heat_wider_transfer(data, manifest)
    lines += transfer_lines
    values.update(transfer_values)
    return lines, values


def corrected_burgers_rollout(data, manifest):
    pilot, summary = data['burgers_rollout'], data['burgers_rollout_summary']
    audit, review = data['burgers_rollout_audit'], data['burgers_rollout_review']
    assert pilot['complete'] and pilot['network_weights_frozen']
    assert audit['source_sha256'] == review['source_json_sha256'] == manifest['burgers_rollout']['sha256']
    assert pilot['checkpoint_sha256'] == data['burgers']['checkpoint_sha256']
    assert pilot['physical_cases'] == data['burgers']['physical_cases']
    assert audit['output_checksums_verified'] and audit['source_hashes_against_commits_verified']
    assert review['verified_invocations'] == audit['invocations_verified'] == len(pilot['invocations'])
    values = {'burgers_corrected_rollout': []}
    lines = ['### Burgers: the corrected initial fit improves complete rollouts', '',
        f"Source `{pilot['commit']}`, job `{pilot['job_id']}`. The network weights, physical initial-condition family and efficient FOM candidates are unchanged. "
        'Fixed physical Gauss sampling charges interpolation from the supplied input field. Both complete-grid and common-grid output errors are preserved, '
        'with separate reference-refinement margins for each grid.', '',
        '| Output intervals | Target (%) | Selected ROM | Worst common / full-grid error (%) | ROM / FOM envelope query (ms) | Paired FOM/ROM | IC budget stops |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in summary['selections']:
        if row['norm'] != 'dense' or not row['empirical_reference_budget_passed']:
            continue
        rom, fom = row['rom'], row['fom']
        common = next(r for r in summary['selections'] if r['norm'] == 'common'
            and r['intervals'] == row['intervals'] and r['target'] == row['target'])
        assert common['rom']['name'] == rom['name'] and common['fom']['name'] == fom['name']
        case_ratios = [fom['times'][case]/rom['times'][case] for case in rom['times']]
        assert abs(statistics.median(case_ratios)-row['paired_ratio']) < 1e-14
        value = dict(intervals=row['intervals'], target=row['target'], selected_rom=rom['name'], selected_fom=fom['name'],
            common_error=rom['common']['worst'], full_grid_error=rom['dense']['worst'],
            rom_ms=rom['ms'], fom_ms=fom['ms'], paired_ratio=row['paired_ratio'],
            initial_budget_stops=rom['initial_budget_stops'], initial_improvement_stops=rom['initial_improvement_stops'],
            evolution_budget_stops=rom['evolution_budget_stops'], evolution_improvement_stops=rom['evolution_improvement_stops'],
            reference_margin=row['reference_margin'], rigorous_reference_bound=None)
        values['burgers_corrected_rollout'].append(value)
        lines.append(f"| {row['intervals']} | {100*row['target']:g} | `{rom['name']}` | {100*value['common_error']:.5g} / {100*value['full_grid_error']:.5g} | {rom['ms']:.5g} / {fom['ms']:.5g} | {row['paired_ratio']:.5g} | {rom['initial_budget_stops']} |")
    n = max(r['intervals'] for r in summary['configurations'])
    edge = next(r for r in summary['configurations'] if r['intervals'] == n and r['cold_rule'] == 'edge')
    gauss = next(r for r in summary['configurations'] if r['intervals'] == n and r['cold_rule'] == 'fixed_gauss'
        and r['ic_budget'] == max(a['ic_budget'] for a in pilot['arm_grid']) and r['dt'] == edge['dt'] and r['stall'] == edge['stall'])
    lines += ['', f"At {n} intervals, the matched primary rollout's worst complete-grid error changes from {100*edge['dense']['worst']:.5g}% with edge initialization "
        f"to {100*gauss['dense']['worst']:.5g}% with the larger-budget Gauss fit. The efficient FOM envelope remains faster at every reported target. "
        'The envelope permits a cheaper coarse solve and charges interpolation to the requested output. The native report retains every selected FOM configuration.', '',
        'Budget exits and configured small-improvement stops remain admissible only under the stated empirical physical-error check; '
        'they are not stationary-fit certificates. All such counts remain in the native records. '
        'The tighter target with an unresolved reference budget is not promoted to qualified evidence.', '',
        f"The owner independently reconstructs {audit['full_grid_artifacts_verified']} complete-grid artifacts representing all {audit['invocations_verified']} timed calls; "
        f"the largest complete-grid metric disagreement is `{audit['dense_error_recompute_max_difference']:.7g}`. "
        f"Root separately recomputes all common-grid errors, with maximum disagreement `{review['maximum_metric_disagreement']:.7g}`. "
        f"There are {audit['nonfinite_invocations']} nonfinite outputs and {audit['fom_nonlinear_tolerance_failures']} FOM tolerance failures. "
        'The reference margin is empirical and independent final confirmation remains open.', '',
        link(ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md', 'Complete corrected Burgers rollout findings')+'.', '']
    return lines, values


def burgers_timesteps(data, manifest):
    pilot, summary = data['burgers_steps'], data['burgers_steps_summary']
    audit, review = data['burgers_steps_audit'], data['burgers_steps_review']
    assert pilot['complete'] and pilot['network_weights_frozen']
    assert audit['source_sha256'] == review['source_json_sha256'] == manifest['burgers_steps']['sha256']
    assert pilot['checkpoint_sha256'] == data['burgers_rollout']['checkpoint_sha256']
    assert pilot['physical_cases'] == data['burgers_rollout']['physical_cases']
    assert audit['output_checksums_verified'] and audit['source_hashes_against_commits_verified']
    assert audit['invocations_verified'] == review['verified_invocations'] == len(pilot['invocations'])
    values = {'burgers_timesteps': [], 'burgers_timestep_envelope': []}
    lines = ['### Burgers: larger steps save some cost but introduce solver-limit effects', '',
        f"Source `{pilot['commit']}`, job `{pilot['job_id']}`. This keeps the frozen decoder, larger-budget Gauss initialization, physical cases and small-improvement stopping rule fixed. "
        'Every returned initial field agrees across the step-size controls. Both ROM and efficient same/coarse-grid FOM candidates are measured again within this job.', '',
        '| Output intervals | ROM step | Worst complete-grid error (%) | Complete query (ms) | Evolution / first-step budget exits |',
        '|---|---:|---:|---:|---:|']
    for row in sorted(summary['configurations'], key=lambda r: (r['intervals'], r['dt'], r['name'])):
        if row['method'] != 'rom':
            continue
        value = dict(intervals=row['intervals'], dt=row['dt'], full_grid_error=row['dense']['worst'],
            common_error=row['common']['worst'], rom_ms=row['ms'],
            evolution_budget_stops=row['evolution_budget_stops'], first_step_budget_stops=row['first_step_budget_stops'])
        values['burgers_timesteps'].append(value)
        lines.append(f"| {row['intervals']} | {row['dt']:g} | {100*value['full_grid_error']:.5g} | {row['ms']:.5g} | {row['evolution_budget_stops']} / {row['first_step_budget_stops']} |")
    lines += ['', 'Exit counts include all timing repetitions. Fewer time steps do not produce a proportional reduction in nonlinear work: larger steps require more solver trials, and several hit the configured budget. '
        'The existing predictor already extrapolates previous latent states when that reduces the weak residual; it has no previous-step history at startup. '
        'The larger-step accuracy changes therefore combine time discretization and incomplete nonlinear solves. They do not establish the error of a fully converged time integrator.', '',
        '| Output intervals | Empirical target (%) | Selected ROM step | ROM / FOM envelope query (ms) | Paired FOM/ROM |',
        '|---|---:|---:|---:|---:|']
    for row in summary['selections']:
        if row['norm'] != 'dense' or not row['empirical_reference_budget_passed']:
            continue
        rom, fom = row['rom'], row['fom']
        if rom is None or fom is None:
            continue
        assert abs(statistics.median(fom['times'][case]/rom['times'][case] for case in rom['times'])-row['paired_ratio']) < 1e-14
        value = dict(intervals=row['intervals'], target=row['target'], selected_rom=rom['name'], selected_fom=fom['name'],
            rom_ms=rom['ms'], fom_ms=fom['ms'], paired_ratio=row['paired_ratio'], reference_margin=row['reference_margin'], rigorous_reference_bound=None)
        values['burgers_timestep_envelope'].append(value)
        lines.append(f"| {row['intervals']} | {100*row['target']:g} | {rom['dt']:g} | {rom['ms']:.5g} / {fom['ms']:.5g} | {row['paired_ratio']:.5g} |")
    lines += ['', 'The efficient FOM envelope remains faster at the qualified development targets. Physical target qualification uses the explicit empirical reference allowance and is not a stationarity certificate. '
        'The native report retains initial/later errors, complete-grid and common-grid margins, staged component diagnostics, every FOM candidate and every configured-stop count.', '',
        f"The owner reconstructs {audit['full_grid_artifacts_verified']} dense output artifacts and verifies repeat hashes for all {audit['invocations_verified']} timed calls; "
        f"the largest complete-grid metric disagreement is `{audit['dense_error_recompute_max_difference']:.7g}`. "
        f"Root independently checks their common-grid errors, with maximum disagreement `{review['maximum_metric_disagreement']:.7g}`. "
        f"There are {audit['nonfinite_invocations']} nonfinite calls and {audit['fom_nonlinear_tolerance_failures']} FOM tolerance failures. "
        'Independent final confirmation remains open.', '',
        link(ROOT/'worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md', 'Complete Burgers timestep findings')+'.', '']
    return lines, values


def wave_dynamics(data, manifest):
    wave, summary = data['wave_dynamics'], data['wave_dynamics_summary']
    audit, review = data['wave_dynamics_audit'], data['wave_dynamics_review']
    assert wave['complete'] and audit['passed'] and not wave['final_test_opened']
    assert summary['result_sha256'] == audit['result_sha256'] == review['source_json_sha256'] == manifest['wave_dynamics']['sha256']
    assert summary['audit_sha256'] == manifest['wave_dynamics_audit']['sha256']
    assert audit['audited_invocations'] == review['verified_invocations'] == len(wave['invocations'])
    assert all(r['passed'] for r in summary['time_refinement'])
    cfg = wave['config']
    values = {'wave_dynamics': [], 'wave_absorbing_final_energy': []}
    lines = ['### Fresh waves: compression and autonomous dynamics both matter', '',
        f"Source `{review['source_commit']}`, job `{review['job_id']}`. The learned bank and existing nonlinear head remain frozen. "
        'The saved affine control has the same displacement and velocity dimensions as the nonlinear head; the larger affine and full-bank controls use additional coordinates. '
        'They all remain inside the learned neural bank. The speed-dependent affine propagator includes the nonzero-offset forcing and is charged inside the query.', '',
        '| Boundary | Intervals | Method | Displacement / phase dimensions | Median / worst time-max error, initial normalization (%) | Method / same-grid FOM query (ms) | Raw paired FOM/method |',
        '|---|---:|---|---:|---:|---:|---:|']
    for bc in cfg['boundaries']:
        for n in cfg['meshes']:
            baseline = {r['case']: r for r in review['rows'] if r['boundary'] == bc and r['intervals'] == n
                and r['method'] == ('dst' if bc == 'dirichlet' else 'rk4') and r['setting'] == (0 if bc == 'dirichlet' else max(cfg['fom_cfls']))}
            for method in ('rom', 'affine16', 'affine32', 'full64'):
                rows = [r for r in review['rows'] if r['boundary'] == bc and r['intervals'] == n and r['method'] == method
                    and r['setting'] == (max(cfg['rom_dts']) if method == 'rom' else 0)]
                assert {r['case'] for r in rows} == baseline.keys()
                k = cfg['latent'] if method == 'rom' else int(method.removeprefix('affine').removeprefix('full'))
                value = dict(boundary=bc, intervals=n, method=method, displacement_dimension=k, phase_dimension=2*k,
                    median_error=statistics.median(r['worst_initial_normalized'] for r in rows),
                    worst_error=max(r['worst_initial_normalized'] for r in rows),
                    method_ms=1000*statistics.median(r['query_median_seconds'] for r in rows),
                    fom_ms=1000*statistics.median(r['query_median_seconds'] for r in baseline.values()),
                    paired_ratio=statistics.median(baseline[r['case']]['query_median_seconds']/r['query_median_seconds'] for r in rows))
                values['wave_dynamics'].append(value)
                lines.append(f"| {bc} | {n} | {method} | {k} / {2*k} | {100*value['median_error']:.5g} / {100*value['worst_error']:.5g} | {value['method_ms']:.6g} / {value['fom_ms']:.6g} | {value['paired_ratio']:.5g} |")
    lines += ['', 'At equal dimension, the affine control improves reflective rollout error but worsens absorbing rollout error. '
        'The larger linear spaces improve both, while the nonlinear head can reconstruct individual snapshots better than its matched affine control. '
        'Snapshot reconstruction therefore does not by itself establish accurate autonomous dynamics. These observations do not establish an irreducible error for every nonlinear manifold of the same dimension.', '',
        'Some linear controls are faster than the listed same-grid FOM, especially for the absorbing problem. '
        'Those are classical linear-control results, and the benchmark has not yet swept cheaper coarse FOMs with charged output interpolation. '
        'They do not establish a nonlinear-decoder or cost-to-tolerance speed advantage.', '',
        '| Absorbing output intervals | Method | Final energy-state error / current reference norm, case range |',
        '|---|---|---:|']
    n = max(cfg['meshes'])
    for method in ('rom', 'affine16', 'affine32', 'full64'):
        rows = [r for r in review['rows'] if r['boundary'] == 'absorbing' and r['intervals'] == n and r['method'] == method
            and r['setting'] == (max(cfg['rom_dts']) if method == 'rom' else 0)]
        ratios = [r['final_current_relative_energy'] for r in rows]
        value = dict(intervals=n, method=method, minimum=min(ratios), maximum=max(ratios))
        values['wave_absorbing_final_energy'].append(value)
        lines.append(f"| {n} | {method} | {min(ratios):.6g}–{max(ratios):.6g} |")
    lines += ['', 'The absorbing reference becomes small. Even the full-bank linear control has a final energy-state error larger than the remaining reference; '
        'its small initial-normalized error must not be described as accurate relative prediction at late times. '
        'Zero-field fits and weak normal-force diagnostics are retained, without attributing the entire failure to either mechanism. '
        'The full-bank control represents zero exactly and has no normal-force residual within the bank by construction, yet still has late absorbing error; neither diagnostic alone certifies physical accuracy.', '',
        f"All {len(summary['time_refinement'])} nonlinear time-step comparisons pass, with maximum required difference `{max(r['maximum_required_difference'] for r in summary['time_refinement']):.7g}`. "
        f"The selected diagnostic fits have {sum(r['selected_fit_nonstationary'] for r in summary['representation'])} nonstationary exits, and maximum longer-budget objective change `{max(r['fit_budget_max_objective_change'] for r in summary['representation'] if r['arm'] == 'mlp16'):.3g}`. "
        'The owner independently reconstructs the full-grid ROM fields, affine propagators, coordinate bank, head derivatives, curvature and diagnostic fits. '
        f"Root separately checks all {review['verified_invocations']} declared timed calls on the common observation grid, with maximum metric difference `{review['maximum_metric_disagreement']:.7g}`. "
        'The audit reuses the saved reference trajectories; full-grid FOM metrics at nonreference time steps remain outside its reconstruction scope. Rigorous reference bounds remain unspecified.', '',
        link(ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/FINDINGS.md', 'Complete wave dynamics findings')+'. '+
        link(ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/dynamics-error-evolution.png', 'Wave error evolution')+'.', '']
    return lines, values


def wave_moment(data, manifest):
    diagnostic = data['wave_moment']
    assert diagnostic['native_sha256'] == manifest['wave_dynamics']['sha256']
    n = max(r['intervals'] for r in diagnostic['bank_diagnostics'])
    bank = next(r for r in diagnostic['bank_diagnostics'] if r['intervals'] == n)
    selected = [r for r in diagnostic['methods'] if r['intervals'] == n]
    lines = ['### Absorbing waves: a missing conservation property is a concrete diagnostic', '',
        'Read-only postprocessing of the archived dynamics run checks a global moment preserved by its full discrete equations:', '',
        '$$I(t)=\\langle 1,v(t)\\rangle_M+c\\langle 1,u(t)\\rangle_B,\\qquad \\dot I=0.$$', '',
        'Here $M$ contains area integration weights, $B$ contains outgoing-boundary weights including both corner contributions, and $c$ is the supplied speed. '
        'The native diagnostic derives this identity from the actual discrete Laplacian and damping. '
        f"At {n} intervals, the relative error in projecting the spatial constant into the learned bank is `{bank['constant_test_projection_error']:.7g}`. "
        'The reduced moment-generator rows are nonzero: this bank does not preserve the original invariant for general states.', '',
        '| Intervals | Case | Method | Initial moment error | Maximum drift from own start | Final moment error |',
        '|---|---:|---|---:|---:|---:|']
    for r in selected:
        label = f"frozen MLP16, dt={r['setting']:g}" if r['method'] == 'rom' else r['method']
        lines.append(f"| {n} | {r['case']} | {label} | {r['initial_error']:.7g} | {r['maximum_drift_from_own_initial']:.7g} | {r['final_error']:.7g} |")
    refs = [r for r in diagnostic['references'] if r['intervals'] == n]
    lines += ['', f"The saved full references have maximum moment drift `{max(r['reference_max_drift'] for r in refs):.7g}` on this mesh. "
        'The table separates initialization error from subsequent drift; its entries are signed moment errors or absolute drift, not relative field errors. '
        'Even unrestricted bank projection changes the input moment. The native findings retain that quantity separately from nonlinear fitting error.', '',
        'This establishes a missing discrete conservation property. It does not establish how much of the late physical error it causes. '
        'Adding a constant test direction and enforcing the initial moment are proposed separate interventions; neither has been tested here, and preserving this moment alone would not certify local-field accuracy.', '',
        link(ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/ABSORBING-MOMENT.md', 'Moment formulas, checks and complete traces')+'.', '']
    return lines, {'wave_absorbing_moment': dict(bank=bank, references=refs, methods=selected)}


def latest_overview(data):
    rows = []

    def timing_counts(rom_cases, fom_cases):
        counts = {}
        for label, cases in [('rom', rom_cases), ('fom', fom_cases)]:
            assert cases and all(len(times) > 1 for times in cases)
            counts[f'{label}_timing_outliers'] = sum(t > 2*statistics.median(times) for times in cases for t in times)
            counts[f'{label}_timing_repetitions'] = sum(map(len, cases))
        return counts

    heat = data['heat_transfer_audit']
    n = max(data['heat_transfer']['settings']['requested_intervals'])
    selection = next(s for s in heat['selections'] if s['cohort'] == 'union' and s['norm'] == 'full' and s['target'] == data['heat_transfer']['config']['accuracy_targets'][1] and s['intervals'] == n)
    group = next(g for g in heat['groups'] if g['cohort'] == 'union' and g['intervals'] == n and g['method'] == selection['rom'])
    rows.append(dict(pde='Heat', intervals=n, setting=selection['rom'], cases=len(group['cases']),
        metric='Current L2, full grid', worst_error=group['full_error'], rom_ms=1000*selection['rom_seconds'],
        fom_ms=1000*selection['fom_seconds'], paired_ratio=selection['paired_fom_over_rom'], baseline='Tested envelope', target=selection['target']))
    rows[-1].update(timing_counts(*[
        [r['timing_repetitions'] for r in data['heat_transfer_review']['rows'] if r['intervals'] == n and r['method'] == selection[method]]
        for method in ('rom', 'fom')]))
    n = max(r['intervals'] for r in data['burgers_steps_summary']['configurations'])
    selection = next(s for s in data['burgers_steps_summary']['selections'] if s['norm'] == 'dense' and s['intervals'] == n and s['target'] == .05)
    rom, fom = selection['rom'], selection['fom']
    rows.append(dict(pde='Burgers', intervals=n, setting=rom['name'], cases=len(rom['times']),
        metric='Initial L2, full grid', worst_error=rom['dense']['worst'], rom_ms=rom['ms'], fom_ms=fom['ms'],
        paired_ratio=selection['paired_ratio'], baseline='Tested envelope', target=selection['target']))
    groups = {method['name']: defaultdict(list) for method in (rom, fom)}
    for r in data['burgers_steps']['invocations']:
        if r['name'] in groups:
            groups[r['name']][r['case']].append(r['seconds'])
    rows[-1].update(timing_counts(*[list(groups[method['name']].values()) for method in (rom, fom)]))
    n = max(data['poisson_speed']['config']['intervals'])
    selection = next(s for s in data['poisson_speed_summary']['envelope'] if s['cohort'] == 'all_development' and s['intervals'] == n and s['target'] == data['poisson_speed']['config']['targets'][0])
    rom, fom = selection['rom'], selection['fom']
    rows.append(dict(pde='Poisson', intervals=n, setting=f"{rom['model']} / {rom['projection']} / {rom['initialization']}",
        cases=len(data['poisson_speed']['cohort']['parameters']), metric='Steady L2, common grid', worst_error=rom['physical_max'],
        rom_ms=1000*rom['latency_seconds'], fom_ms=1000*fom['latency_seconds'], paired_ratio=selection['median_case_cost_ratio'],
        baseline='Tested envelope', target=selection['target']))
    panels = [next(g for g in data['poisson_speed_summary']['summaries'] if g['cohort'] == 'all_development' and g['panel'] == 'primary' and g['intervals'] == n
        and all(g[k] == method[k] for k in ('model', 'arm', 'projection', 'initialization'))) for method in (rom, fom)]
    rows[-1].update(timing_counts(*[[c['raw_seconds'] for c in g['cases']] for g in panels]))
    cfg = data['wave_heads']['config']; n = max(cfg['meshes'])
    method = f"new_mlp32_seed{cfg['training']['optimizer_seeds'][0]}"
    for bc in cfg['boundaries']:
        g = next(r for r in data['wave_heads_summary']['groups'] if (r['boundary'],r['intervals'],r['method'],r['setting']) == (bc,n,method,max(cfg['nonlinear_dts'])))
        f = next(r for r in data['wave_heads_summary']['groups'] if (r['boundary'],r['intervals'],r['method'],r['setting']) == (bc,n,'dst' if bc=='dirichlet' else 'rk4',0. if bc=='dirichlet' else max(cfg['fom_cfls'])))
        rows.append(dict(pde='Reflective wave' if bc=='dirichlet' else 'Absorbing wave',intervals=n,
            setting=f"{method}, dt={g['setting']:g}",cases=len(g['cases']),metric='Initial wave state, common grid',
            worst_error=g['physical_error_worst'],rom_ms=1000*g['query_median'],fom_ms=1000*f['query_median'],
            paired_ratio=g['paired_FOM_over_method_ratio'],baseline='Same grid only',target=None))
        rows[-1].update(timing_counts(*[[c['query_repetitions'] for c in group['cases']] for group in (g, f)]))
    lines = ['## Latest audited development results', '',
        'These rows show the finest requested mesh from each latest completed study. Heat, Burgers and Poisson use their declared development cost-to-accuracy selections. '
        'Wave rows show the first predeclared larger-head seed at the largest repeated step; both seeds and all tested steps are retained in the detailed findings. '
        'Different physical norms and cohorts are explicit and must not be ranked as one cross-PDE accuracy score.', '',
        '| PDE | Output intervals | Cases | Error norm | Worst error (%) | ROM / FOM query (ms) | Paired FOM/ROM | Timing outliers, ROM / FOM | FOM comparison |',
        '|---|---:|---:|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['pde']} | {r['intervals']} | {r['cases']} | {r['metric']} | {100*r['worst_error']:.5g} | {r['rom_ms']:.6g} / {r['fom_ms']:.6g} | {r['paired_ratio']:.5g} | {r['rom_timing_outliers']} / {r['fom_timing_outliers']} | {r['baseline']} |")
    targets = ', '.join(f"{r['pde']} {100*r['target']:g}%" for r in rows if r['target'] is not None)
    lines += ['', f'The cost-to-accuracy targets shown are {targets}, with explicitly empirical reference allowances. '
        'Wave costs are raw same-grid comparisons, without a coarse-FOM envelope. Ratios above one would favor ROM; none of these rows does. '
        'The absorbing wave still has large late error relative to its remaining field. These are development results, not independent final confirmation.', '',
        'Query costs are medians of per-case repetition medians, including supplied host input and requested host outputs. Paired ratios are medians of per-case FOM/ROM ratios. '
        'Timing outliers count repetitions longer than twice their own case/configuration median; none is discarded. All comparisons pair methods within the same GPU job.', '',
        'The earlier pilot table and the controlled changes below preserve how these conclusions were obtained; earlier rows are not the current best settings.', '']
    return lines, rows


def poisson_source_initialization(data, manifest):
    pilot, summary = data['poisson_speed'], data['poisson_speed_summary']
    audit, review = data['poisson_speed_audit'], data['poisson_speed_review']
    assert pilot['complete'] and audit['passed'] and audit['remote_deleted']
    assert summary['source_sha256'] == review['source_json_sha256'] == manifest['poisson_speed']['sha256']
    assert len(pilot['rows']) == review['verified_timed_invocations'] == audit['primary_invocations']
    assert len(pilot['stationary_rows']) == review['verified_accuracy_controls'] == audit['stationary_invocations']
    cfg = pilot['config']
    projections = {'skinny_sine_products': 'sine products', 'forward_dst_and_gather': 'DST gather'}
    initializers = {'mean_training_code': 'mean code', 'nearest_cached_scaled_weak_prediction': 'nearest code'}
    values = {'poisson_speed_factorial': [], 'poisson_speed_envelope': [], 'poisson_speed_contrasts': summary['factorial_cost_contrasts']}
    lines = ['### Poisson: field-based initialization saves some cost; the FOM remains faster', '',
        f"Source `{review['source_commit']}`, job `{review['job_id']}`. Both checkpoints are frozen: the original model and the relative-loss continuation selected using all development sources on both meshes. "
        'The factorial crosses two equivalent sine-source projections with mean-code versus nearest-training-code initialization. '
        'Nearest lookup compares supplied-field weak coefficients with cached decoder predictions from training codes; it uses no Gaussian descriptors or evaluation answers.', '',
        '| Intervals | Checkpoint | Projection | Initialization | Median / worst physical error (%) | Query (ms) | Median solver attempts |',
        '|---|---|---|---|---:|---:|---:|']
    for r in summary['summaries']:
        if r['cohort'] != 'all_development' or r['panel'] != 'primary' or not r['model']:
            continue
        value = {k:r[k] for k in ('intervals','model','projection','initialization','physical_median','physical_max','latency_seconds','attempt_median','invalid_count','nonstationary_count','initial_residual_median','absolute_tau_threshold_median','final_residual_median')}
        values['poisson_speed_factorial'].append(value)
        lines.append(f"| {r['intervals']} | {r['model']} | {projections[r['projection']]} | {initializers[r['initialization']]} | {100*r['physical_median']:.5g} / {100*r['physical_max']:.5g} | {1000*r['latency_seconds']:.6g} | {r['attempt_median']:.5g} |")
    init_effect = [r['median_case_cost_ratio'] for r in summary['factorial_cost_contrasts'] if r['contrast'] == 'mean_over_nearest_initialization']
    projection_effect = [r['median_case_cost_ratio'] for r in summary['factorial_cost_contrasts'] if r['contrast'] == 'thin_over_dst_projection']
    lines += ['', f"Paired mean-start/nearest-start cost ratios range from {min(init_effect):.6g} to {max(init_effect):.6g}; ratios above one favor nearest starts. "
        f"Paired sine-product/DST-gather ratios range from {min(projection_effect):.6g} to {max(projection_effect):.6g}, so there is no consistent projection-cost gain. "
        'A projector selected by aggregate median latency can differ from the one favored by the median paired ratio; the native report retains both statistics.', '',
        r"The stopping threshold stays $\tau\|r(z_0)\|_2$ for each call's own initial code. A nearer start can require a tighter absolute residual, and may finish by stationarity rather than the relative-reduction stop. The target is never re-anchored. Cost changes therefore describe initialization together with the existing stopping procedure, not an isolated iteration-count effect.", '',
        '| Intervals | All-development target (%) | Selected checkpoint / projection / initialization | ROM / FOM envelope query (ms) | Paired FOM/ROM |',
        '|---|---:|---|---:|---:|']
    for r in summary['envelope']:
        if r['cohort'] != 'all_development':
            continue
        values['poisson_speed_envelope'].append(r)
        rom, fom = r['rom'], r['fom']
        label = 'Target unattained' if rom is None else f"{rom['model']} / {projections[rom['projection']]} / {initializers[rom['initialization']]}"
        rt = '—' if rom is None else f"{1000*rom['latency_seconds']:.6g}"
        ft = '—' if fom is None else f"{1000*fom['latency_seconds']:.6g}"
        ratio = '—' if r['median_case_cost_ratio'] is None else f"{r['median_case_cost_ratio']:.5g}"
        lines.append(f"| {r['intervals']} | {100*r['target']:g} | {label} | {rt} / {ft} | {ratio} |")
    stationary = [r for r in summary['summaries'] if r['cohort'] == 'all_development' and r['panel'] == 'stationary']
    lines += ['', f"All {review['verified_accuracy_controls']} single-call stationary controls are retained separately, with {sum(r['invalid_count'] for r in stationary)} invalid case/configuration outputs. "
        'They never enter the speed envelope. Projection parity compares the same initialization; different minima between mean and nearest starts are allowed and judged by their physical errors and solver checks.', '',
        'Timed outputs include the full field plus returned stop reasons, counters, latent states, guard results and initialization metadata. '
        'Stationarity is independently recomputed afterward from the same invocation, alongside physical-error and parity diagnostics. '
        'Offline cache/operator construction and compilation remain outside the query.', '',
        f"The owner audits all {audit['primary_invocations']+audit['stationary_invocations']} outputs and reconstructs the coordinate bank, decoder, cache, residuals and gradients. "
        f"Root recomputes their saved field errors, maximum disagreement `{review['maximum_metric_disagreement']:.7g}`, and independently replays {review['independent_nearest_lookup_panels']} source-field lookup panels. "
        f"There are {audit['lookup_index_mismatches']} lookup mismatches, {audit['projection_gate_failures']} projection-gate failures and {audit['timed_fallbacks']} timed guarded-solver fallbacks. "
        'The efficient tested FOM envelope remains faster. Reference allowance is empirical and sealed-final confirmation remains unopened.', '',
        link(ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot05/FINDINGS.md', 'Complete Poisson projection and initialization findings')+'.', '']
    return lines, values


def heat_wider_transfer(data, manifest):
    pilot, audit, review = data['heat_transfer'], data['heat_transfer_audit'], data['heat_transfer_review']
    assert pilot['complete'] and audit['passed'] and not pilot['final_cohort_opened']
    assert audit['results_sha256'] == review['source_json_sha256'] == manifest['heat_transfer']['sha256']
    assert audit['timed_invocations'] == review['verified_timed_invocations'] == pilot['timed_invocations']
    cfg, settings = pilot['config'], pilot['settings']
    target = cfg['accuracy_targets'][1]
    values = {'heat_transfer_accuracy': [], 'heat_transfer_envelope': []}
    lines = ['### Heat: the refined heads transfer across the wider mesh ladder', '',
        f"Source `{review['source_commit']}`, job `{review['job_id']}`. Both expanded-coverage heads and their training-code libraries remain byte-identical to the prior endpoints. "
        f"This run repeats {settings['cohorts'][0]['count']} original development inputs and adds {settings['cohorts'][1]['count']} fresh draws from the same restricted family. "
        f"The requested intervals are `{settings['requested_intervals']}`, with shared observations at {settings['observation_intervals']} intervals and complete-grid errors also checked. "
        'There is no new training or final-cohort evaluation.', '',
        '| Output intervals | Cohort | Frozen endpoint | Full-grid median / worst current-relative error (%) | Common-grid worst error (%) | Complete query (ms) |',
        '|---|---|---|---:|---:|---:|']
    for g in sorted(audit['groups'], key=lambda r: (r['intervals'], r['cohort'], r['method'])):
        if g['cohort'] == 'union' or g['model'] == 'fom':
            continue
        rows = [r for r in review['rows'] if (r['intervals'], r['cohort'], r['method']) == (g['intervals'], g['cohort'], g['method'])]
        assert {r['case'] for r in rows} == set(g['cases'])
        assert abs(max(r['worst_full_grid_error'] for r in rows)-g['full_error']) < 1e-12
        value = dict(intervals=g['intervals'], cohort=g['cohort'], model=g['model'], cases=len(rows),
            full_grid_median_error=statistics.median(r['worst_full_grid_error'] for r in rows), full_grid_worst_error=g['full_error'],
            common_worst_error=g['common_error'], query_ms=1000*g['cost_seconds'], valid=g['valid'],
            nonstationary_initial=g['nonstationary_initial'], nonstationary_steps=g['nonstationary_steps'])
        values['heat_transfer_accuracy'].append(value)
        lines.append(f"| {g['intervals']} | {g['cohort']} | {g['model']} | {100*value['full_grid_median_error']:.5g} / {100*g['full_error']:.5g} | {100*g['common_error']:.5g} | {value['query_ms']:.6g} |")
    lines += ['', 'Every FOM returns the supplied initial field exactly, then evolves a restricted initial field and interpolates later outputs where required. '
        'Host input restriction, GPU work, interpolation and complete contiguous host output are charged. The ROM returns its actual fitted initial field and charges full-field projection, fitting, evolution and readout. '
        f"The FOM envelope tests solver intervals `{settings['coarse_solver_intervals']}` where no larger than the requested mesh, plus the same-grid solve. Exact duplicate choices are timed once with explicit aliases.", '',
        '| Output intervals | Union full-grid current-relative target (%) | Selected ROM / FOM solver | ROM / FOM query (ms) | Paired FOM/ROM |',
        '|---|---:|---|---:|---:|']
    selected = [s for s in audit['selections'] if s['cohort'] == 'union' and s['norm'] == 'full' and s['target'] == target]
    for s in selected:
        values['heat_transfer_envelope'].append(s)
        rt = '—' if s['rom_seconds'] is None else f"{1000*s['rom_seconds']:.6g}"
        ft = '—' if s['fom_seconds'] is None else f"{1000*s['fom_seconds']:.6g}"
        ratio = '—' if s['paired_fom_over_rom'] is None else f"{s['paired_fom_over_rom']:.5g}"
        lines.append(f"| {s['intervals']} | {100*s['target']:g} | {s['rom'] or 'Unattained'} / {s['fom'] or 'Unattained'} | {rt} / {ft} | {ratio} |")
    qualified = [s for s in selected if s['rom'] is not None and s['fom'] is not None]
    wins = sum(s['paired_fom_over_rom'] > 1 for s in qualified)
    lines += ['', f"At the union-cohort {100*target:g}% target with empirical reference adjustment, a head qualifies on {len(qualified)} of {len(selected)} meshes and beats the tested FOM envelope on {wins}. "
        'The full-output contract limits how much spatial compression alone can save; the native figure separates input, device and output cost components. '
        'This study measures frozen-weight mesh transfer, not per-resolution retraining.', '',
        f"The native audit checks {audit['timed_invocations']} invocations and independently reproduces FOM propagation/interpolation to `{audit['independent_fom_kernel_relative_disagreement']:.7g}` relative disagreement, "
        f"and spectral reference trajectories to `{audit['independent_spectral_reference_relative_disagreement']:.7g}`. "
        f"Root recomputes every saved norm and checks source draws, frozen weights, repetitions and the initial-output policy, maximum metric difference `{review['maximum_metric_disagreement']:.7g}`. "
        f"There are {audit['nonstationary_initial']} nonstationary selected initial fits and {audit['nonstationary_steps']} nonstationary evolution steps across repetitions. "
        f"The observed spectral reference discrepancy is `{audit['reference_empirical_delta']:.7g}`, with no rigorous bound. "
        'Both cohorts, all candidates, raw timing arrays and outliers remain in the native audit.', '',
        link(ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/HEAT-TRANSFER-NOTES.md', 'Complete wider heat-transfer findings')+'. '+
        link(ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/heat-transfer.png', 'Heat accuracy and cost across resolution')+'. '+
        link(ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/heat-transfer-components.png', 'Heat query cost components')+'.', '']
    return lines, values


def wave_larger_heads(data, manifest):
    pilot, summary = data['wave_heads'], data['wave_heads_summary']
    audit, review = data['wave_heads_audit'], data['wave_heads_review']
    assert pilot['complete'] and audit['passed'] and not pilot['final_test_opened']
    assert summary['result_sha256'] == audit['result_sha256'] == review['source_json_sha256'] == manifest['wave_heads']['sha256']
    assert summary['audit_sha256'] == manifest['wave_heads_audit']['sha256']
    assert review['verified_invocations'] == audit['audited_comparison_invocations'] == len(pilot['invocations'])
    assert review['verified_accuracy_controls'] == audit['audited_accuracy_only_invocations'] == len(pilot['accuracy_controls'])
    assert all(not r['comparison_eligible'] for r in pilot['accuracy_controls'])
    figures = data['wave_heads_figures']
    assert figures['source_result_sha256'] == manifest['wave_heads']['sha256']
    for figure in figures['figures']:
        assert hashlib.sha256(Path(figure['path']).read_bytes()).hexdigest() == figure['sha256']
    cfg = pilot['config']
    n = max(cfg['meshes'])
    values = {'wave_larger_heads': [], 'wave_larger_absorbing_final': []}
    lines = ['### Fresh waves: larger nonlinear heads improve accuracy, with a remaining cost gap', '',
        f"Source `{review['source_commit']}`, job `{review['job_id']}`. Each boundary receives {len(cfg['training']['optimizer_seeds'])} fixed reconstruction-only endpoints with {cfg['new_latent_dimension']} displacement coordinates, "
        f"each trained for {cfg['training']['head_steps']} updates on the original {cfg['train_count']} regenerated training trajectories. "
        f"The spatial bank stays fixed with {cfg['weak_bank_equations']} weak equations. The equations-to-coordinate ratio changes from {cfg['weak_bank_equations']/cfg['latent']:g} to {cfg['weak_bank_equations']/cfg['new_latent_dimension']:g}; "
        'the nonlinear system remains overdetermined. No velocity or rollout loss is added.', '',
        f'The table shows the finer requested mesh ({n} intervals). The native findings retain both meshes, every repeated step-size setting and both seeds.', '',
        '| Boundary | Method | Step | Median / worst time-max error, initial normalization (%) | Method / same-grid FOM query (ms) | Raw paired FOM/method |',
        '|---|---|---:|---:|---:|---:|']
    for g in summary['groups']:
        if g['intervals'] != n or g['method'] in ('dst', 'rk4'):
            continue
        if g['method'].startswith('new_') and g['setting'] not in (min(cfg['nonlinear_dts']), max(cfg['nonlinear_dts'])):
            continue
        rows = [r for r in review['rows'] if r['comparison_eligible'] and
            (r['boundary'], r['intervals'], r['method'], r['setting']) == (g['boundary'], n, g['method'], g['setting'])]
        assert len(rows) == len(cfg['validation_indices'])
        assert abs(max(r['worst_initial_normalized'] for r in rows)-g['physical_error_worst']) < 1e-12
        baseline = next(b for b in summary['groups'] if b['boundary'] == g['boundary'] and b['intervals'] == n and
            (b['method'], b['setting']) == (('dst', 0.) if g['boundary'] == 'dirichlet' else ('rk4', max(cfg['fom_cfls']))))
        value = dict(boundary=g['boundary'], intervals=n, method=g['method'], dt=g['setting'],
            median_error=g['physical_error_median'], worst_error=g['physical_error_worst'], method_ms=1000*g['query_median'],
            fom_ms=1000*baseline['query_median'], raw_paired_ratio=g['paired_FOM_over_method_ratio'],
            failed_cases=g['failed_cases'], nonstationary_cases=g['nonstationary_cases'], unresolved_cases=g['time_refinement_failed_cases'])
        values['wave_larger_heads'].append(value)
        lines.append(f"| {g['boundary']} | {g['method']} | {g['setting']:g} | {100*value['median_error']:.5g} / {100*value['worst_error']:.5g} | {value['method_ms']:.6g} / {value['fom_ms']:.6g} | {value['raw_paired_ratio']:.5g} |")
        if g['boundary'] == 'absorbing' and (not g['method'].startswith('new_') or g['setting'] == max(cfg['nonlinear_dts'])):
            final = [c['final_current_relative']['energy_state'] for c in g['cases']]
            values['wave_larger_absorbing_final'].append(dict(method=g['method'], dt=g['setting'], minimum=min(final), maximum=max(final)))
    lines += ['', 'At the same step, the larger nonlinear heads improve initial-normalized rollout error but increase query cost. '
        'Using their larger tested step recovers some cost, while the efficient same-grid FOM remains faster. '
        'At the finer mesh shown, the larger nonlinear heads do not improve the all-case worst error of their matched affine control. '
        'A finer time integrator alone does not close the physical-error gap.', '',
        f"All {len(summary['time_refinement'])} adjacent nonlinear step comparisons pass the unchanged `{summary['time_refinement_target']:g}` development criterion; "
        f"their maximum required difference is `{max(r['maximum_required_difference'] for r in summary['time_refinement']):.7g}`. "
        f"The {review['verified_accuracy_controls']} finest-step queries are separately audited accuracy controls, excluded from every timing summary and selection. "
        'Their saved single-call latencies may include uncached compilation.', '',
        '| Absorbing method | Step | Final energy-state error / current reference norm, case range |',
        '|---|---:|---:|']
    for r in values['wave_larger_absorbing_final']:
        lines.append(f"| {r['method']} | {r['dt']:g} | {r['minimum']:.6g}–{r['maximum']:.6g} |")
    lines += ['', 'Late absorbing errors remain larger than the remaining reference field. Improved error relative to the initial state does not establish accurate relative prediction at late times. '
        'The separate moment diagnostic has not been corrected in this experiment.', '',
        f"The native audit verifies {audit['audited_comparison_invocations']} timed calls and {audit['audited_accuracy_only_invocations']} fine controls, "
        f"with {audit['failed_comparison_invocations']} failed timed trajectories and {audit['failed_accuracy_only_invocations']} failed fine trajectories. "
        'It reconstructs full-grid ROM fields, head derivatives and fitting diagnostics, verifies training/checkpoint lineage, and checks the affine generator. '
        f"Root independently recomputes all common-field metrics and accounting roles, maximum disagreement `{review['maximum_metric_disagreement']:.7g}`. "
        'Reference bounds remain empirical; the full-grid timed FOM reconstruction limitation from the prior wave audit remains. '
        'No coarse-FOM envelope, new development cohort or sealed-final confirmation is included.', '',
        link(ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/k32heads03/analysis/FINDINGS.md', 'Complete larger-head wave findings')+'. '+
        link(ROOT/'worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/k32heads03/analysis/heads32-error-evolution.png', 'Larger-head wave error evolution')+'.', '']
    return lines, values


def poisson_training_factorial(data, manifest):
    pilot, summary = data['poisson_training'], data['poisson_training_summary']
    audit, review = data['poisson_training_audit'], data['poisson_training_review']
    assert pilot['complete'] and audit['passed']
    assert summary['source_sha256'] == review['source_json_sha256'] == manifest['poisson_training']['sha256']
    assert audit['row_count'] == review['verified_invocations'] == len(pilot['rows'])
    cfg = pilot['config']
    target = cfg['targets'][1]
    endpoints = {r['model']: r for r in pilot['checkpoints']}
    values = {'poisson_training_accuracy': [], 'poisson_training_envelope': []}
    lines = ['### Poisson: training coverage and relative loss are distinct changes', '',
        f"Source `{review['source_commit']}`, job `{review['job_id']}`. The factorial crosses original versus expanded training coverage with global versus per-field relative squared-error normalization. "
        f"The prescribed budget is {cfg['training_steps_each']} updates per continuation, with unchanged architecture and identical initial weights. "
        f"Evaluation uses {cfg['existing_development_count']} previously inspected and {cfg['additional_development_count']} fresh development sources, outside training. "
        'The fresh development cohort is now available for method selection; it is not sealed final confirmation.', '',
        '| Intervals | Development cohort | Endpoint | Training sources / completed updates | Median / worst physical error (%) | Failures at target | Invalid / nonstationary cases |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in summary['summaries']:
        if row['cohort'] == 'all_development' or row['arm'] != 'rom_modular' or row['tau'] != 0:
            continue
        endpoint = endpoints[row['model']]
        value = dict(intervals=row['intervals'], cohort=row['cohort'], model=row['model'],
            training_sources=endpoint['training_count'], completed_updates=endpoint['steps_done'],
            valid_endpoint=endpoint['valid_endpoint'], median_error=row['physical_median'], worst_error=row['physical_max'],
            target=target, failures=row['targets'][str(target)]['failure_count'], cases=row['case_count'],
            invalid=row['invalid_count'], nonstationary=row['nonstationary_count'])
        values['poisson_training_accuracy'].append(value)
        lines.append(f"| {row['intervals']} | {row['cohort']} | {row['model']} | {value['training_sources']} / {value['completed_updates']} | {100*value['median_error']:.5g} / {100*value['worst_error']:.5g} | {value['failures']} / {value['cases']} at {100*target:g}% | {value['invalid']} / {value['nonstationary']} |")
    lines += ['', 'These are the tighter generic stationary-control solves. The failure count applies the physical-error target with its empirical reference adjustment and solver/endpoint gates. '
        'Original and expanded sets share an update/batch budget; the expanded set receives fewer visits per source. '
        'Both the spatial bank and nonlinear head are refined in this Poisson study. At query time each endpoint uses its own mean training code. '
        'Thus these results compare complete training procedures under the stated budget, without isolating a universal effect of more data.', '',
        '| Intervals | All-development target (%) | Selected ROM endpoint / solver / tau | ROM / eligible FOM query (ms) | Paired FOM/ROM |',
        '|---|---:|---|---:|---:|']
    for row in summary['envelope']:
        if row['cohort'] != 'all_development':
            continue
        rom, fom = row['rom'], row['fom']
        value = dict(intervals=row['intervals'], target=row['target'], rom=rom, fom=fom,
            paired_ratio=row['median_case_cost_ratio'], rigorous_reference_bound=None)
        values['poisson_training_envelope'].append(value)
        label = f"{rom['model']} / {rom['arm']} / {rom['tau']:g}" if rom else 'Target unattained'
        rt = f"{1000*rom['latency_seconds']:.5g}" if rom else '—'
        ft = f"{1000*fom['latency_seconds']:.5g}" if fom else '—'
        ratio = f"{row['median_case_cost_ratio']:.5g}" if row['median_case_cost_ratio'] is not None else '—'
        lines.append(f"| {row['intervals']} | {100*row['target']:g} | {label} | {rt} / {ft} | {ratio} |")
    lines += ['', 'The complete-query envelope uses every development case and includes the charged coarse-grid FOM option. '
        'Per-case timing repetition arrays, component medians, outliers, failed gates and paired factorial error contrasts are retained in the native findings. '
        'Model selection for any later speed experiment must use all cases and both meshes rather than an inspected example. '
        'Offline training durations include compilation and diagnostics and are not a paired warm training-speed comparison.', '',
        f"Root independently recomputes {review['verified_invocations']} invocation metrics, checks declared source draws and endpoint accounting, and finds maximum metric disagreement `{review['maximum_metric_disagreement']:.7g}`. "
        'The owner additionally verifies source/checkpoint history and training normalization samples against an independent CPU reference. '
        f"There are {audit['specialized_agreement_failures']} specialized-solver agreement failures and {audit['specialized_timed_fallbacks']} timed fallbacks. "
        'Reference adjustment remains empirical; no continuum certificate or global nonlinear optimum is claimed.', '',
        link(ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot04/FINDINGS.md', 'Complete Poisson training findings')+'.', '']
    return lines, values


if __name__ == '__main__':
    main()
