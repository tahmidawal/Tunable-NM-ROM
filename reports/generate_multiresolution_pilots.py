"""Generate the initial four-PDE pilot comparison from immutable run artifacts."""
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
        'This report covers the first frozen-network mesh-transfer pilots and bounded numerical improvements of the current separable NM-ROM against efficient FOM solvers. '
        'The numbers are provisional development evidence; independent confirmation and the full resolution study remain open.', '',
        'The tested frozen decoders produce solutions on new meshes, but these primary configurations have not established a complete-query advantage over efficient FOMs. '
        'Increasing resolution mostly preserves the ROM error in these pilots. Further work must address representation, initialization or reduced-solver cost, according to the PDE.', '',
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
        '**Burgers:** errors use the initial reference norm. The first reference refinement estimate leaves target qualification unresolved. '
        'Cold-fit budget exits are retained in the native records and independent audit; an observed small error is not proof of a stationary initial fit. '
        'The follow-up separately tests more starting guesses and a finer reference.', '',
        '**Poisson:** errors are relative solution norms for each steady source. Tighter stationary solves still leave a worst-case error floor. '
        'The reference has empirical refinement evidence; development qualification is not an independent final-cohort result.', '',
        '**Waves:** the reported metric is the maximum of initial-normalized displacement, velocity and energy-state errors. '
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
        '- **Burgers:** resolve reference space/time error; compare the original cold start with several training-code guesses; retain the efficient same-grid and coarse-grid FOMs.',
        '- **Heat:** after the measured tolerance improvement, address the nonlinear-head reconstruction gap with controlled original-cohort versus expanded-coverage head refinement. The spatial bank and head architecture can remain fixed for that diagnostic.',
        '- **Poisson:** the test-mode and representation diagnostics point to bank/head capacity or training coverage for accuracy. A specialized small-matrix solver is a separate remaining runtime test.',
        '- **Waves:** use matched-dimensional linear and nonlinear controls with a frozen spatial bank to separate compression from autonomous dynamics. Finer rendering alone cannot address the current error.', '',
        'These are development decisions. The complete study still needs the full mesh ladder, separately labeled per-resolution training, '
        'independent data/training repeats, validation-selected settings and sealed final evaluation.', '',
        '## Visual artifacts', '',
        link(REPAIR/'heat-pilot01-accuracy-cost.png', 'Heat accuracy and complete-query cost')+'. '+
        link(REPAIR/'dirichlet-wave-evolution-case1.png', 'Reflective wave evolving')+'. '+
        link(REPAIR/'absorbing-wave-evolution-case1.png', 'Absorbing wave evolving')+'.', '',
        'The wave still sequences show reference displacement, predicted displacement and absolute difference with fixed scales. '
        'They use saved fields from an actual finer-mesh ROM solve; the display resolution is labeled. '
        'Displacement pictures do not replace the velocity, energy or vanishing-field diagnostics. '
        'PDF exports and figure source/provenance are beside the images.', '',
        '## Reproduction', '',
        'Run `/home/tahmid/Dev/.venv/bin/python reports/generate_multiresolution_pilots.py` from a checkout with the recorded experiment worktrees. '
        'The adjacent JSON manifest identifies every source artifact by content hash. All numerical table values are generated; none are hand-entered.', '',
        '## Plain-language glossary', '',
        '- **PDE / FOM / NM-ROM / ROM:** partial differential equation / full spatial solver / nonlinear-manifold reduced solver / reduced solver.',
        '- **Frozen / intervals / cases:** unchanged network weights / grid cells along one axis / distinct physical inputs. Timing repetitions are not additional cases.',
        '- **ROM setting / dt / CN / stall / tau:** chosen solver configuration / time step / Crank–Nicolson time formula / relative improvement stopping rule / requested weak-residual reduction.',
        '- **Current L2 / initial L2 / steady L2:** field error divided by the current reference norm / initial reference norm / steady reference solution norm.',
        '- **Initial wave state / energy-state error:** the separately normalized displacement, velocity and energy measures / error combining displacement gradients with velocity.',
        '- **Median / worst / query ms / paired ratio:** middle case result / largest case error / complete input-to-output milliseconds / per-case FOM time divided by ROM time, then a cohort median.',
        '- **Bank / head / latent / weak test mode:** learned spatial features / their nonlinear coefficient map / compressed state coordinates / smooth function averaging the PDE equation.',
        '- **Cold start / stationary / budget exit:** initial reduced-state fitting / meeting a local derivative convergence check / exhausting allowed iterations.',
        '- **DST / coarse-grid envelope:** direct sine-transform solution / least-cost qualifying full solve allowing fewer cells and charging interpolation to the requested output.',
        '- **Reference refinement / uncertainty / qualification:** comparing finer trusted solves / remaining reference error / meeting accuracy and numerical-validity requirements with that uncertainty included.',
        '- **Development / validation / sealed final / provisional:** preliminary experiment data / data used to select settings / untouched independent confirmation data / evidence with the stated limitations.',
        '- **Compiled / segmented / projection / compression:** one prepared executable / separately launched stages / representing a field in a spatial span / constraining that representation through fewer latent coordinates.',
        '- **Strict / relaxed / gradient tolerance / field drift:** more demanding stopping control / less demanding stopping setting / required smallness of the objective derivative / change from the strict-control trajectory divided by the current reference norm.',
        '- **Paired improvement / requested modes / retained modes:** strict-control time divided by changed-method time, summarized across cases / desired minimum number of weak tests / actual number when tied sine eigenmodes are retained together.',
        '- **Commit / manifest / checksum:** saved source revision / inventory of source artifacts / content fingerprint checking exact file bytes.', '']
    followup_lines, followup_values = followups(data, manifest)
    where = lines.index('## Next experiments justified by the diagnostics')
    lines[where:where] = followup_lines
    (HERE/'2026-09-07-multiresolution-pilots.md').write_text('\n'.join(lines))
    (HERE/'2026-09-07-multiresolution-pilots.json').write_text(json.dumps({'artifacts': manifest, 'comparisons': comparisons, 'followups': followup_values}, indent=2)+'\n')


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
        link(ROOT/'worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot02/FINDINGS.md', 'Complete Poisson follow-up findings')+'.', '',
        'All cost ratios in this report use ratios of per-case timing medians before the cohort median. Some native exploratory reports also retain medians of per-repetition ratios under an explicit different label; those statistics are not interchangeable.', '']
    return lines, values


if __name__ == '__main__':
    main()
