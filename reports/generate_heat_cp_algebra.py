"""Generate the direct CP algebra transfer report from audited native results."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/cp_algebra08'
NAMES = {
    'nmrom': 'Original NMROM',
    'nmrom_cholesky': 'Cholesky only',
    'nmrom_gram_lu': 'Precomputed Gram only',
    'nmrom_gram_cholesky': 'Gram + Cholesky (declared primary)',
    'linear_weak_exact': 'Free-coefficient linear bank control',
    'fom_same_grid': 'Same-grid direct FOM',
    'fom_coarse16': 'Coarse direct FOM with interpolation',
}


def main():
    paths = [RECORD / 'archive/outputs/results.json', RECORD / 'analysis/audit.json',
             RECORD / 'COLLECTION-CHECK.json', RECORD / 'REMOTE-CLEANUP.json', RECORD / 'smoke/SMOKE-CHECK.json']
    result, audit, collection, cleanup, smoke = [json.loads(p.read_text()) for p in paths]
    assert result['complete'] and audit['passed'] and collection['archive_verified'] and cleanup['remote_absent']
    assert audit['result_sha256'] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    settings = result['settings']; cfg = result['config']; meta = result['metadata']
    lookup = {(r['intervals'], r['method']): r for r in audit['summaries']}
    n = max(settings['requested_intervals']); primary = settings['primary_method']
    base = lookup[n, 'nmrom']; new = lookup[n, primary]
    chol = lookup[n, 'nmrom_cholesky']
    same = lookup[n, 'fom_same_grid']; coarse = lookup[n, 'fom_coarse16']
    speed = base['device_median_ms']/new['device_median_ms']
    lines = ['# Transferring the CP Gram and Cholesky optimizations to heat NMROM', '',
        'Completed and independently audited development experiment. Numbers are provisional for paper claims because this uses one frozen training checkpoint and existing development cases; final confirmation cases remain unopened.', '',
        f"At {n} intervals, the declared Gram + Cholesky method takes {new['device_median_ms']:.6f} ms versus {base['device_median_ms']:.6f} ms for the original NMROM: a {speed:.3f}× ratio of same-job median GPU times. Worst current-relative errors are {100*new['worst_physical_error']:.6f}% and {100*base['worst_physical_error']:.6f}%. This algebra change {'reduces' if speed>1 else 'does not reduce'} the measured cohort median GPU cost.", '',
        f"Cholesky alone takes {chol['device_median_ms']:.6f} ms, a {100*(1-chol['device_median_ms']/base['device_median_ms']):.3f}% reduction in median GPU time, with worst error {100*chol['worst_physical_error']:.6f}%. This is a separately declared ablation, not a replacement for the joint method's primary result.", '',
        f"The same-grid direct FOM takes {same['device_median_ms']:.6f} ms with {100*same['worst_physical_error']:.6f}% worst error. The coarse direct FOM takes {coarse['device_median_ms']:.6f} ms with {100*coarse['worst_physical_error']:.6f}% worst error. The primary nonlinear method {'is' if new['device_median_ms']<coarse['device_median_ms'] else 'is not'} faster than the coarse GPU control at the declared {100*settings['accuracy_target']:.6g}% development target.", '',
        '## All declared comparisons', '',
        'All timings below come from one allocation. GPU time runs from supplied GPU initial field to all requested GPU fields, blocked. Host time adds measured input and output transfers for that same invocation. Data generation, mesh assembly and compilation are excluded from every online time and recorded separately.', '',
        '| Intervals | Method | Median GPU ms | Median host ms | Worst relative error (%) | Original / method GPU ratio | Nonstationary fits / steps | GPU / host outliers |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mesh in settings['requested_intervals']:
        for name in settings['methods']:
            row = lookup[mesh,name]
            lines.append(f"| {mesh} | {NAMES[name]} | {row['device_median_ms']:.6f} | {row['host_median_ms']:.6f} | {100*row['worst_physical_error']:.6f} | {lookup[mesh,'nmrom']['device_median_ms']/row['device_median_ms']:.3f} | {row['nonstationary_fit_count']} / {row['nonstationary_step_count']} | {row['device_outliers']} / {row['host_outliers']} |")
    lines += ['', f"Provisional development results: {len(result['cases'])} cases, {settings['timing_repetitions']} repetitions per case and method. Error is the worst over every case, requested time and retained repetition. Outlier rule: {audit['outlier_rule']}", '',
        f"The nonlinear variants all retain the same {cfg['k']} latent coordinates, {cfg['r']} spatial-bank functions, {cfg['modes_per_axis']**2} smooth weak tests, initial fitting policy, {round(cfg['times'][-1]/settings['dt'])} Crank–Nicolson steps of size {settings['dt']}, and normalized gradient tolerance {settings['gradient_tolerance']:.6g}. Every method returns all {len(cfg['times'])} requested fields. The linear bank control has free coefficients and is a different model class.", '',
        '## Per-case primary comparison', '',
        f'These are medians within each case at {n} intervals. A ratio below unity means the primary method is slower.', '',
        '| Case | Original GPU ms | Gram + Cholesky GPU ms | Original / primary | Primary worst error (%) |',
        '| ---: | ---: | ---: | ---: | ---: |']
    rows = {(r['intervals'],r['case'],r['method']):r for r in result['rows']}
    ratios = []
    for case in result['cases']:
        cid = case['case']; a = rows[n,cid,'nmrom']; b = rows[n,cid,primary]
        am = median(r['phases']['device_seconds'] for r in a['repetitions'])*1000
        bm = median(r['phases']['device_seconds'] for r in b['repetitions'])*1000
        error = max(max(r['vs_physical']['relative_current']) for r in b['repetitions'])
        ratios.append(am/bm)
        lines.append(f'| {cid} | {am:.6f} | {bm:.6f} | {am/bm:.3f} | {100*error:.6f} |')
    lines += ['', f"Median of per-case speed ratios: {median(ratios):.3f}×; primary faster in {sum(r>1 for r in ratios)} of {len(ratios)} cases. This is distinct from the ratio of cohort median times reported above.", '',
        '## Numerical equivalence and convergence', '',
        f"Predeclared field and internal-latent parity gates {'passed' if audit['algebra_parity_passed'] else 'failed'}: tolerances {settings['field_parity_tolerance']:.6g} and {settings['latent_parity_tolerance']:.6g}. All initial-fit and time-step iteration/acceptance/termination counters {'agree' if audit['all_counts_equal'] else 'do not agree'} with the same-job original. Rounding differences are measured, not assumed absent.", '',
        '| Variant | Maximum field relative difference | Maximum internal-latent relative difference | Comparisons failing a parity gate | Comparisons with different counters |',
        '| --- | ---: | ---: | ---: | ---: |']
    for method in settings['methods']:
        ps = [p for p in audit['algebra_parity'] if p['method']==method]
        if not ps: continue
        lines.append(f"| {NAMES[method]} | {max(p['field_relative_difference'] for p in ps):.12g} | {max(p['internal_latent_relative_difference'] for p in ps):.12g} | {sum(not(p['field_pass'] and p['latent_pass']) for p in ps)} | {sum(not(p['initial_counts_equal'] and p['step_counts_equal']) for p in ps)} |")
    lines += ['', f"Independent NumPy head derivatives reproduce {audit['time_step_weak_checks']} saved time-step residual/gradient pairs, maximum discrepancy {audit['maximum_time_step_weak_difference']:.12g}. They reproduce {audit['initial_fit_weak_checks']} initial-fit diagnostic pairs within {audit['maximum_initial_fit_weak_difference']:.12g}. Initial projection targets are recovered through independent QR from sampled linear-control initial fields; this is not a full-grid independent reconstruction of the input projection.", '',
        f"Full-field errors and hashes were audited for {audit['timed_invocations_checked']} timed invocations and {audit['unique_fields_checked']} distinct arrays; maximum metric discrepancy {audit['maximum_metric_difference']:.12g}. Sampled neural reconstruction over all requested output times has maximum relative mismatch {audit['maximum_sampled_decoder_error']:.12g}. It complements full-field metric checks without claiming a separate full-grid neural evaluation.", '',
        f"Independent QR operator and Gram checks agree within {max(x['relative_error'] for x in audit['operator_errors']):.12g}. The unchanged baseline reproduces earlier archived fields within {max(x['relative_difference'] for x in audit['prior_nmrom_field_parity']):.12g}. The maximum continuum-spectral reference-refinement delta is {audit['maximum_reference_refinement']:.12g}; this remains empirical evidence, not a rigorous continuum bound.", '',
        '## What was transferred, and what remains', '',
        r"For $r=(Bh(z)-b)/s$ and $D=\partial h/\partial z$, precompute $S=B^\top B$ offline and evaluate the normal matrix as $H=D^\top SD/s^2$. Here $B$ maps bank coefficients to weak moments, $b$ is their target, and $s=\max(\|b\|,\epsilon)$ normalizes the residual with a small positive floor. The gradient uses the same contraction, anchored at the actual initial residual to limit cancellation. The explicit small weak residual remains the loss used to accept or reject each trial.", '',
        r"Cholesky factors the positive-definite damped system instead of using general LU. Neither transformation changes the mathematical objective, time integrator or decoder. The test includes each transformation separately and both together; no post-selection arm is promoted to a predeclared result.", '',
        'The historical CP optimization removed full-grid Jacobian work. This heat solver already projects to a small weak test space, so the transferable contraction acts on a much smaller matrix. Its payoff must be measured; historical speed ratios cannot be inherited. The full rollout was already compiled, so compiling it again does not reproduce the older eager-to-compiled gain.', '',
        'The weak solve has no full-grid Jacobian, but still performs iterative nonlinear fits and time-step corrections. A grid-independent iteration can remain more expensive than this direct heat FOM. Reading the supplied full field and producing all requested full fields also continue to scale with resolution.', '',
        f"The original worst initial error is {100*base['worst_initial_error']:.6f}% at {n} intervals. Algebraic speed changes cannot repair that initial fitting/representation error. The next distinct architecture experiment is a learned initializer from supplied-field features, trained jointly with the coefficient head, with an explicit scale coordinate if useful. That proposal still requires accuracy and whole-query timing tests; it has not been trained or validated here.", '',
        'This experiment uses the existing single-bump family and fixed diffusivity. It does not establish multi-bump, variable-coefficient, Burgers, Poisson or wave performance. The old heat CP evidence also used a different iterative FOM, precision and error summary; its speedups are not comparable to the current direct/coarse controls. See the [historical source review](2026-07-30-cp-heat-optimization-transfer.md).', '',
        '## Reproducible evidence', '',
        f"Job `{meta['job_id']}` on `{meta['node']}`, `{meta['gpu']}`, scientific source `{audit['source_commit']}`. GPU preflight, float64, highest matrix precision, seed regeneration, private job directory and complete logs passed. Timing repetitions are retained; no timing outliers are discarded. Source and output checksums passed before the exact remote attempt directory was removed.", '',
        f"The local smoke completed in {smoke['elapsed_seconds']:.6f} seconds and passed an independent exact least-squares fixture with a nonlinear head. Local smoke timings are not benchmark evidence.", '',
        f"[Raw results](../{paths[0].relative_to(ROOT).as_posix()}) · [Independent audit](../{paths[1].relative_to(ROOT).as_posix()}) · [Configuration](../{(RECORD/'archive/experiments/mr-heat2d/config-cp-algebra.json').relative_to(ROOT).as_posix()}) · [Archive manifest](../{(RECORD/'ARCHIVE.json').relative_to(ROOT).as_posix()})", '',
        '## Glossary', '',
        '- **Intervals:** spatial cells per axis; saved arrays contain interior nodes.',
        '- **CP / NMROM / FOM:** tensor-product decoder / nonlinear-manifold reduced model / full-grid numerical solver.',
        '- **Bank / head / latent coordinates:** learned spatial functions / nonlinear coefficient map / compressed variables being solved.',
        '- **Gram / normal equations / Jacobian:** matrix of inner products / small least-squares update system / derivatives of outputs with respect to solved variables.',
        '- **LU / Cholesky:** general matrix factorization / factorization specialized to symmetric positive-definite systems.',
        '- **Weak tests / residual / stationarity:** smooth averages of the equation / their discrepancy / meeting the specified gradient stopping rule.',
        '- **Crank–Nicolson / damped solve:** original time discretization / least-squares correction stabilized by a positive diagonal term.',
        '- **Median GPU / host ms:** middle retained blocked device-query time / same invocation including CPU–GPU transfers, in milliseconds.',
        '- **Worst relative error:** maximum full-field error divided by the reference norm at the same time, over all cases, times and repetitions.',
        '- **Original / method ratio:** original median time divided by the listed method median; greater than unity is faster than the original.',
        '- **Nonstationary fits / steps:** queries containing any initial fit missing the gradient rule / individual time steps missing it; counts include repetitions.',
        '- **Outliers:** timings above the stated threshold within the same case, mesh and method; all remain included.',
        '- **Case / development target:** seeded initial condition / provisional physical-error threshold; neither is a final-cohort validation claim.',
        '- **Parity / counters:** agreement in trajectory or latent coordinates / attempted and accepted corrections and termination reasons.',
        '- **Free-coefficient control / coarse direct FOM:** linear bank model with unconstrained coefficients / smaller-grid direct propagation interpolated to output nodes.',
        '- **QR / reference refinement / sampled reconstruction:** independent orthogonal-triangular factorization / difference between reference resolutions / checking decoder values at common physical nodes.',
        '- **Invocation / checkpoint / source checksum:** one timed solve producing the graded fields / frozen trained parameters / content hash proving exact provenance.']
    target = ROOT/'reports/2026-09-10-heat-cp-algebra-comparison.md'
    target.write_text('\n'.join(lines)+'\n')
    paths.append(Path(__file__).resolve())
    manifest = dict(source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(target)


if __name__ == '__main__': main()
