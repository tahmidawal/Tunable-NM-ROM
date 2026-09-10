"""Generate both nonlinear heat acceleration panels from accepted native evidence."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs'
NAMES = {
    'nmrom': 'Original NMROM',
    'nmrom_adaptive': 'Original stepping, adaptive initialization',
    'exp_project_full': 'Linear prediction, full-budget nonlinear projection',
    'exp_project_full_adaptive': 'Linear prediction, full-budget projection, adaptive initialization',
    'exp_project2': 'Linear prediction, bounded nonlinear projection',
    'exp_project2_adaptive': 'Linear prediction, bounded projection, adaptive initialization',
    'linear_weak_exact': 'Free-coefficient linear bank control',
    'fom_same_grid': 'Same-grid direct FOM',
    'fom_coarse16': 'Coarse direct FOM with interpolation',
}


def read_panel(name):
    record = RUNS / name
    paths = [record / 'archive/outputs/results.json', record / 'analysis/audit.json', record / 'REMOTE-CLEANUP.json']
    result, audit, cleanup = [json.loads(p.read_text()) for p in paths]
    assert result['complete'] and audit['passed'] and cleanup['remote_absent']
    assert audit['result_sha256'] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    lookup = {(r['intervals'], r['method']): r for r in audit['summaries']}
    return dict(name=name, record=record, paths=paths, result=result, audit=audit, lookup=lookup)


def step_counts(panel, n, name):
    rows = [r for r in panel['result']['rows'] if r['intervals'] == n and r['method'] == name]
    reps = [rep for row in rows for rep in row['repetitions']]
    total = sum(len(rep['solver'].get('steps', [])) for rep in reps)
    affected = sum(any(s[2] != 1 for s in row['repetitions'][0]['solver'].get('steps', [])) for row in rows)
    return total, affected


def gpu_table(panel):
    settings = panel['result']['settings']; audit = panel['audit']
    lines = ['| Intervals | Method | Median GPU ms | Worst relative error (%) | Meets physical development target | Nonstationary steps / all steps | Affected cases | GPU timing outliers |',
             '| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for n in settings['requested_intervals']:
        for name in NAMES:
            if (n, name) not in panel['lookup']: continue
            row = panel['lookup'][n, name]
            total, affected = step_counts(panel, n, name)
            qualifies = row['worst_physical_error'] + audit['maximum_reference_refinement'] <= settings['accuracy_target']
            lines.append(f"| {n} | {NAMES[name]} | {row['device_median_ms']:.6f} | {100*row['worst_physical_error']:.6f} | {'yes' if qualifies else 'no'} | {row['nonstationary_step_count']} / {total} | {affected} | {row['device_outliers']} |")
    lines += ['', f"Provisional development results: all {len(panel['result']['cases'])} cases and {settings['timing_repetitions']} repetitions per case are retained. Physical qualification uses worst full-field error plus the empirical reference-refinement allowance at the declared {100*settings['accuracy_target']:.6g}% target. It does not imply solver stationarity. Step counts include repeated invocations; affected cases count unique first-repetition case trajectories. FOM and free-bank controls have no nonlinear steps.", '']
    return lines


def accounting_table(panel):
    lines = ['| Intervals | Method | Median host query ms | Host outliers | Skipped second fits / queries | Queries with nonstationary attempted initial fits | Total initial attempts | Total time-step attempts | Worst initial error (%) | Largest energy increase |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in panel['audit']['summaries']:
        lines.append(f"| {row['intervals']} | {NAMES[row['method']]} | {row['host_median_ms']:.6f} | {row['host_outliers']} | {row['skipped_second_initializations']} / {row['invocations']} | {row['nonstationary_fit_count']} | {row['total_initial_attempts']} | {row['total_step_attempts']} | {100*row['worst_initial_error']:.6f} | {row['largest_energy_increase']:.9g} |")
    return lines + ['', 'Counts include every retained timed repetition. A negative largest energy increase means every consecutive requested output loses energy. A skipped initialization has its own recorded reason and is excluded from the nonstationary-fit count. Host query time adds measured input/output transfers to the same invocation. Neither solver is charged for data generation, offline assembly or compilation; setup and warmup costs are retained in the native results.', '']


def per_case_table(panel, n):
    rows = {(r['case'], r['method']): r for r in panel['result']['rows'] if r['intervals'] == n}
    lines = [f'### Per-case comparison at {n} intervals', '',
        'Cohort medians can hide a slow difficult query. Each row below retains the median over that case\'s repetitions; a speed ratio below unity means the projected method is slower.', '',
        '| Case | Original GPU ms | Projected GPU ms | Original / projected speed ratio | Projected worst error (%) | Nonstationary steps in first repetition |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in panel['result']['cases']:
        cid = case['case']; base = rows[cid, 'nmrom']; full = rows[cid, 'exp_project_full']
        old_ms = 1000*median(r['phases']['device_seconds'] for r in base['repetitions'])
        new_ms = 1000*median(r['phases']['device_seconds'] for r in full['repetitions'])
        error = max(max(r['vs_physical']['relative_current']) for r in full['repetitions'])
        failed = sum(s[2] != 1 for s in full['repetitions'][0]['solver']['steps'])
        lines.append(f'| {cid} | {old_ms:.6f} | {new_ms:.6f} | {old_ms/new_ms:.3f} | {100*error:.6f} | {failed} |')
    return lines + ['']


def unconverged_table(panel):
    failed = [(row, i, step) for row in panel['result']['rows'] if row['method'] == 'exp_project_full'
        for i, step in enumerate(row['repetitions'][0]['solver']['steps']) if step[2] != 1]
    if not failed: return ['Every full-budget projection in this panel met the recorded gradient stopping rule.', '']
    tol = panel['result']['settings']['gradient_tolerance']
    times = panel['result']['config']['times']
    lines = ['### Remaining convergence failures', '',
        f'The following distinct first-repetition corrections miss the declared normalized gradient threshold {tol:.9g}. Extra attempts do not establish convergence for these cases; the measured physical errors and speed gains do not remove this limitation.', '',
        '| Intervals | Case | Output time | Attempts | Accepted attempts | Relative weak residual | Normalized gradient |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row, i, step in failed:
        lines.append(f"| {row['intervals']} | {row['case']} | {times[i+1]:.6g} | {int(step[0])} | {int(step[1])} | {step[3]:.12g} | {step[4]:.12g} |")
    return lines + ['', 'A useful next controlled test is a shorter prediction interval, charging the extra projections and checking both physical error and stationarity. Increasing the correction cap again without diagnosing the difficult step is not an established solution.', '']


def evidence(panel):
    audit = panel['audit']; result = panel['result']; meta = result['metadata']
    config_name = 'config-nonlinear-convergence.json' if panel['name'] == 'nonlinear07' else 'config-nonlinear-fast.json'
    config = panel['record'] / 'archive/experiments/mr-heat2d' / config_name
    return [f"Job `{meta['job_id']}`, `{meta['gpu']}` on `{meta['node']}`, scientific source `{audit['source_commit']}`. GPU preflight, float64 and highest matrix precision were checked. The private remote directory was removed after checksum collection.", '',
        f"Independent NumPy/SciPy audit: {audit['unique_fields_checked']} unique full fields; {audit['timed_invocations_checked']} timing/error invocations; {audit['metric_entries_checked']} metric entries, maximum discrepancy {audit['maximum_metric_difference']:.12g}. Reference-refinement delta {audit['maximum_reference_refinement']:.12g}; this is empirical agreement, not a rigorous continuum error bound.", '',
        f"For {audit['nonlinear_decoder_checks']} nonlinear trajectories, every requested time was independently reconstructed at the common physical observation grid from saved latent coordinates and frozen neural weights; maximum relative mismatch {audit['maximum_sampled_decoder_error']:.12g}. This checks sampled decoder identity alongside full-field error auditing; it is not an independent full-grid neural reconstruction.", '',
        f"The audit reconstructs {audit['projected_weak_checks']} projected residual/gradient pairs with maximum difference {audit['maximum_projected_weak_difference']:.12g}, and checks every applicable adaptive-initialization gate. Reduced operators agree with independent QR algebra within {max(r['relative_error'] for r in audit['operator_errors']):.12g}. Original NMROM fields match the earlier frozen-head archive within {max(r['relative_difference'] for r in audit['prior_nmrom_field_parity']):.12g}. Timing comparisons use this allocation only.", '',
        f"[Raw results](../{panel['paths'][0].relative_to(ROOT).as_posix()}) · [Independent audit](../{panel['paths'][1].relative_to(ROOT).as_posix()}) · [Configuration](../{config.relative_to(ROOT).as_posix()})", '']


def main():
    screen = read_panel('nonlinear06'); follow = read_panel('nonlinear07')
    result = follow['result']; settings = result['settings']; cfg = result['config']
    n = max(settings['requested_intervals']); base = follow['lookup'][n, 'nmrom']; full = follow['lookup'][n, 'exp_project_full']
    same = follow['lookup'][n, 'fom_same_grid']; coarse = follow['lookup'][n, 'fom_coarse16']
    fast = screen['lookup'][n, 'exp_project2_adaptive']
    screen_base = screen['lookup'][n, 'nmrom']; adaptive = screen['lookup'][n, 'nmrom_adaptive']
    original_cap = screen['result']['settings'].get('full_projection_budget', cfg['step_budget'])
    follow_cap = settings['full_projection_budget']
    worst_initial_row = max((r for r in result['rows'] if r['intervals'] == n and r['method'] == 'exp_project_full'),
        key=lambda r: r['repetitions'][0]['vs_physical']['relative_current'][0])
    worst_case = worst_initial_row['case']
    bank_row = next(r for r in result['rows'] if r['intervals'] == n and r['case'] == worst_case and r['method'] == 'linear_weak_exact')
    bank_initial_error = bank_row['repetitions'][0]['vs_physical']['relative_current'][0]
    lines = ['# Accelerating heat while retaining the nonlinear decoder', '',
        'Completed and independently audited development experiments with the frozen current nonlinear decoder. The numbers are provisional for paper claims: the convergence follow-up was chosen after the initial screen, and independent final cases remain unopened.', '',
        f"In the convergence follow-up at {n} intervals, the original NMROM takes {base['device_median_ms']:.6f} ms and linear prediction plus nonlinear projection takes {full['device_median_ms']:.6f} ms, a {base['device_median_ms']/full['device_median_ms']:.3f}× same-job speed ratio. Worst current-relative errors are {100*base['worst_physical_error']:.6f}% and {100*full['worst_physical_error']:.6f}%. The projection method has {full['nonstationary_step_count']} nonstationary timed steps on this mesh.", '',
        f"The same-job direct FOM takes {same['device_median_ms']:.6f} ms with {100*same['worst_physical_error']:.6f}% worst error; its interpolated coarse counterpart takes {coarse['device_median_ms']:.6f} ms with {100*coarse['worst_physical_error']:.6f}%. The nonlinear acceleration {'does' if full['device_median_ms']<coarse['device_median_ms'] else 'does not'} beat that matched-target coarse GPU baseline. All headline ratios use one allocation.", '',
        f"A smaller improvement passes the stationarity checks in the initial screen: adaptive initialization with original stepping takes {adaptive['device_median_ms']:.6f} ms versus its paired original NMROM at {screen_base['device_median_ms']:.6f} ms, reducing median GPU time by {100*(1-adaptive['device_median_ms']/screen_base['device_median_ms']):.3f}%. Its worst error is {100*adaptive['worst_physical_error']:.6f}%, with {adaptive['nonstationary_step_count']} nonstationary time steps and {adaptive['nonstationary_fit_count']} queries containing nonstationary attempted initial fits. This result is also development-only.", '',
        f"All nonlinear methods retain {cfg['k']} latent variables, {cfg['r']} learned bank functions and {cfg['modes_per_axis']**2} smooth weak tests, with no retraining. The linear control evolves {cfg['r']} free coefficients and is a different model class. Every method receives a full supplied initial field and returns all {len(cfg['times'])} requested full fields.", '',
        '## Convergence follow-up', '',
        f"The initial screen allowed {original_cap} correction attempts per projection. Some solves exhausted that budget despite acceptable physical error, so calling that method converged was premature. The follow-up raises only the full-projection limit to {follow_cap}, retains the original initialization, and reruns the original NMROM and all direct controls in the same job. This is a development follow-up, not independent confirmation.", '']
    lines += gpu_table(follow)
    lines += per_case_table(follow, n)
    lines += unconverged_table(follow)
    lines += ['### Initialization, transfers and energy', ''] + accounting_table(follow)
    lines += ['### Follow-up evidence', ''] + evidence(follow)
    lines += ['## Initial screen: all declared variants', '',
        f"The most aggressive combined variant uses at most {screen['result']['settings']['projection_budget']} correction attempts per projection and conditionally skips the second initial fit. At {n} intervals it takes {fast['device_median_ms']:.6f} ms, but its worst error is {100*fast['worst_physical_error']:.6f}%, {'above' if fast['worst_physical_error']>settings['accuracy_target'] else 'below'} the declared physical target. There is no hidden fallback. It remains a reported control, not a selected success.", '']
    lines += gpu_table(screen)
    lines += ['### Initialization, transfers and energy', ''] + accounting_table(screen)
    lines += ['### Initial-screen evidence', ''] + evidence(screen)
    lines += ['## Method and limits', '',
        f"The original integrator uses {round(cfg['times'][-1]/settings['dt'])} internal Crank–Nicolson steps. The new method uses {len(cfg['times'])-1} linear predictions and nonlinear projections, one per requested observation interval. It changes the time integrator, not just the algebra used to produce the old trajectory.", '',
        r"Write the bank as $G=QR$ and the verified reduced linear generator in orthogonal coordinates as $L$. Precompute $P_a=R^{-1}\exp(\Delta tL)R$. Each nonlinear update solves", '',
        r"$$z_{j+1}\approx\arg\min_z\|B h(z)-B P_a h(z_j)\|_2^2,\qquad u_{j+1}=G h(z_{j+1}).$$", '',
        r"Here $B$ maps bank coefficients to smooth weak moments, and $h$ is the frozen nonlinear head. The free linear predictor is intermediate; every returned nonlinear-method field is decoded from the latent state. A matrix exponential does not make the constrained nonlinear rollout exact.", '',
        f"The screen's adaptive initializer runs the nearest-code fit first, and tries the mean-code fit when stationarity, finiteness or the full-field initial-error gate of {100*settings['initial_gate']:.6g}% fails. The error gate includes the component outside the learned bank. This is a changed initialization policy and can change trajectories. The convergence follow-up isolates projection cost by retaining the original two-start policy.", '',
        f"The follow-up projected method has worst initial error {100*full['worst_initial_error']:.6f}% at {n} intervals. Faster time stepping does not repair the initial nonlinear fit or the decoder's representation limit. Accuracy work should next diagnose and improve that initial representation/fit; speed work should reduce its online cost without hiding the supplied-field input or requested field output.", '',
        f"On that same worst-initial-error case ({worst_case}), the free-bank initial projection error is only {100*bank_initial_error:.6f}%. This identifies a gap beyond the fixed spatial bank's representation error. The present experiment does not separate nonlinear-head representation limits from local-fit optimization limits; a converged local fit is not proof of a global optimum.", '',
        'The current test uses fixed diffusivity and the existing single-bump family. It does not establish multi-bump, variable-diffusivity or other-PDE performance. The full resolution ladder, per-resolution training and independent final-cohort confirmation remain open. Endpoint results and post-screen convergence tuning are insufficient for a paper-wide scaling claim.', '',
        f"Timing outlier rule: {follow['audit']['outlier_rule']}", '',
        '## Glossary', '',
        '- **Intervals:** cells per spatial axis; homogeneous boundary values are implicit in saved interior arrays.',
        '- **NMROM / FOM:** nonlinear-manifold reduced model / full-grid numerical solver.',
        '- **Bank / head / latent:** frozen spatial functions / nonlinear coefficient map / evolving compressed coordinates.',
        '- **Linear prediction / nonlinear projection:** evolve free bank coefficients, then fit them back to the nonlinear decoder using weak moments.',
        '- **Full-budget / bounded projection:** use the stated larger correction limit / allow only the stated small number of attempts. Neither label assumes convergence.',
        '- **Adaptive initialization / two-start policy:** conditionally try a second starting guess / always fit both declared guesses and select the better fit.',
        '- **LM / attempt / stationarity:** damped nonlinear least squares / an accepted or rejected correction trial / satisfying the gradient stopping rule.',
        '- **Nonstationary steps / affected cases:** timed corrections that miss the stopping criterion / distinct first-repetition case trajectories containing such a correction.',
        '- **Queries with nonstationary attempted initial fits:** invocations where at least one attempted starting-guess fit misses stationarity; not necessarily the selected fit.',
        '- **Weak moments / residual:** averages against smooth sine test functions / discrepancy in those averages.',
        '- **Relative weak residual / normalized gradient:** weak discrepancy divided by the target moment norm / least-squares gradient divided by the Jacobian norm, used for the stopping rule.',
        '- **Crank–Nicolson / matrix exponential:** the original implicit time-step formula / direct propagation for a fixed linear differential equation.',
        '- **Median GPU / host query:** median blocked GPU-input-to-GPU-output time / same invocation including measured CPU–GPU field transfers.',
        '- **Worst relative / initial error:** largest field error divided by the current reference norm over all times / initial time only, expressed as percentages.',
        '- **Physical development target / cohort:** declared field-error threshold / fixed group of sampled cases used for development, distinct from final confirmation.',
        '- **Outlier:** timing above the stated within-case threshold; no such samples are removed.',
        '- **Energy increase:** consecutive change in discrete squared-field energy; positive values indicate growth.',
        '- **Free-coefficient control / coarse FOM:** linear learned-bank model / smaller-grid direct solve interpolated to requested output points.',
        '- **Refinement allowance / sampled identity check:** empirical difference between reference resolutions / independent decoder reconstruction at shared physical observation nodes.',
        '- **QR / orthogonal coordinates:** factorization into orthonormal and triangular matrices / coordinates defined by that orthonormal bank.',
        '- **Scientific source / checksum collection:** exact code version that produced a run / validation of collected files against their recorded hashes.']
    target = ROOT / 'reports/2026-09-10-heat-nonlinear-acceleration.md'
    target.write_text('\n'.join(lines)+'\n')
    paths = [p for panel in (screen, follow) for p in panel['paths']]
    paths.append(Path(__file__).resolve())
    manifest = dict(source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(target)


if __name__ == '__main__': main()
