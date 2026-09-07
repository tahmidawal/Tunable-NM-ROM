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

    lines = ['# Initial multiresolution pilots: accuracy and complete-query cost', '',
        'This report covers the first frozen-network mesh-transfer pilots of the current separable NM-ROM against efficient FOM solvers. '
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
        '- **Heat:** keep the checkpoint fixed and compare solver tolerances and compiled full-query execution. Treat the separate nonlinear-head reconstruction gap as an accuracy task.',
        '- **Poisson:** increase smooth test-mode coverage and measure full-bank projection versus best recorded nonlinear-head fits; compare compiled and segmented queries.',
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
        '- **Commit / manifest / checksum:** saved source revision / inventory of source artifacts / content fingerprint checking exact file bytes.', '']
    (HERE/'2026-09-07-multiresolution-pilots.md').write_text('\n'.join(lines))
    (HERE/'2026-09-07-multiresolution-pilots.json').write_text(json.dumps({'artifacts': manifest, 'comparisons': comparisons}, indent=2)+'\n')


if __name__ == '__main__':
    main()
