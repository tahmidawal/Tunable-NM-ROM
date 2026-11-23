"""Reconstruct every registered W1 trajectory and evaluate the frozen width gates.

This is a NumPy saved-array audit. It neither reruns a solver nor selects cases.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_bank(output):
    return np.concatenate([np.load(p)['bank'] for p in sorted(output.glob('bank-columns-*.npz'))], axis=1)


def spatial(fields, n):
    padded = np.pad(fields.reshape(-1, n-1, n-1), ((0, 0), (1, 1), (1, 1)))
    c = padded[:, 1:-1, 1:-1]
    xm, xp = padded[:, :-2, 1:-1], padded[:, 2:, 1:-1]
    ym, yp = padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]
    adv = c*n*(np.where(c > 0, c-xm, xp-c)+np.where(c > 0, c-ym, yp-c))
    lap = n*n*(xm+xp+ym+yp-4*c)
    grad = np.stack(np.gradient(padded, 1/n, axis=(1, 2)), axis=1)
    return adv.reshape(len(fields), -1), lap.reshape(len(fields), -1), grad.reshape(len(fields), -1)


def compare_metrics(field, reference, n):
    fa, fl, fg = spatial(field, n)
    ra, rl, rg = spatial(reference, n)
    norm = lambda a: np.linalg.norm(a, axis=1)
    return np.stack((norm(field-reference)/np.linalg.norm(reference[0]),
        norm(fg-rg)/np.maximum(norm(rg), 1e-300), norm(fa-ra)/np.maximum(norm(ra), 1e-300),
        norm(fl-rl)/np.maximum(norm(rl), 1e-300), .5*np.sum(field**2, axis=1)/n**2), axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outputs', type=Path, nargs='+')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    lane = Path(__file__).resolve().parent
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=True)
    admission = read(lane/'inputs/width-admission.json')
    audit = dict(scope='post-pilot frozen-prefix development; same-grid same-step discrete target',
        no_cross_job_timing_claim=True, primary_metric='maximum over all 51 times, including initialization; initial-field norm denominator',
        common_observer='smooth rows 8193 through 9216', sources={}, cases={}, baseline_reproduction=[], comparisons=[])
    banks, environments, source_summaries, outputs = {}, {}, {}, {}
    for output in args.outputs:
        env = read(output/'environment.json')
        cfg = env['config']
        width = int(cfg['bank_width'])
        assert width not in outputs, 'Do not pool repeated attempts.'
        assert cfg['case_ids'] == list(range(6))
        assert cfg['solve_rows'] == admission['allowed_test_spaces'][str(width)]
        assert cfg['observer_start'] == 8192 and cfg['observer_rows'] == 1024
        for key, expected in admission['fixed_contract'].items():
            assert cfg[key] == expected, key
        assert env['checkpoint_sha256'] == admission['source_checkpoint_sha256']
        assert env['rotation_sha256'] == admission['source_rotation_sha256']
        assert env['source_sha256']['core.py'] == admission['source_core_sha256']
        cohort = read(output/'cohort-drawing.json')
        assert cohort['guard_passed'] and cohort['legacy_same_host_bitwise']
        assert cohort['selected_case_ids'] == list(range(6))
        assert cohort['first_four_columns_exact'] and cohort['local_portability_max_ulp'] <= 4
        manifest = []
        for line in (output.parent/'RESULTS.sha256').read_text().splitlines():
            expected, rel = line.split('  ', 1)
            manifest.append(dict(path=rel, sha256=expected, verified=sha(output.parent/rel) == expected))
        assert all(row['verified'] for row in manifest)
        collect = read(output.parent/'collection.json')
        environment_ok = bool(collect['checksum_verified'] and collect['gpu_log_valid'] and collect['no_environmental_warnings']
            and collect['scheduler_state'] == 'COMPLETED' and env['backend'] == 'gpu' and env['x64'] and env['matmul_precision'] == 'highest')
        G = load_bank(output)
        assert G.dtype == np.float64
        n = cfg['mesh_intervals']
        operators = np.load(output/'operators.npz')
        gram = G.T @ G/n**2
        gram_error = float(np.linalg.norm(gram-operators['gram'])/np.linalg.norm(gram))
        eig = np.linalg.eigvalsh(gram)
        R = np.load(output/'initialization-qr.npz')['Rmass']
        singular = np.linalg.svd(R, compute_uv=False)
        ranks = {str(t):dict(gram=int(np.sum(eig > t*eig[-1])), qr=int(np.sum(singular**2 > t*singular[0]**2))) for t in (1e-10, 1e-12, 1e-14)}
        rank_pass = all(a == width for vals in ranks.values() for a in vals.values())
        factor_error = float(np.linalg.norm(R.T @ R-gram)/np.linalg.norm(gram))
        src = dict(output=str(output.resolve()), job_id=env['job_id'], commit=env['commit'],
            environment_sha256=sha(output/'environment.json'), summary_sha256=sha(output/'summary.json'),
            cohort_sha256=cohort['parameter_sha256'], cohort_guard_passed=True, bank_dtype=str(G.dtype),
            verified_file_count=len(manifest), retention_verified=True, environment_passed=environment_ok,
            gram_reconstruction_relative=gram_error, qr_factor_relative=factor_error, ranks=ranks,
            numerical_basis_passed=bool(rank_pass and gram_error <= 1e-10 and factor_error <= 1e-10))
        audit['sources'][str(width)] = src
        banks[width], environments[width], outputs[width] = G, env, output
        source_summaries[width] = read(output/'summary.json')
        assert set(source_summaries[width]['cases']) == set(map(str, range(6)))
    assert 128 in outputs, 'The fresh baseline must be included.'
    assert len({s['cohort_sha256'] for s in audit['sources'].values()}) == 1
    cfg0 = environments[128]['config']
    n = cfg0['mesh_intervals']
    times = np.arange(round(cfg0['horizon']/cfg0['dt'])+1)*cfg0['dt']
    table = []
    for width in sorted(outputs):
        output, G = outputs[width], banks[width]
        cfg, summary = environments[width]['config'], source_summaries[width]
        prefix_gaps = {str(r):float(np.linalg.norm(G[:, :r]-banks[r])/np.linalg.norm(banks[r])) for r in banks if r <= width}
        shared_bank_gap = max(prefix_gaps.values())
        audit['sources'][str(width)]['R128_bank_prefix_relative_gap'] = prefix_gaps['128']
        audit['sources'][str(width)]['all_narrower_bank_prefix_relative_gaps'] = prefix_gaps
        for case_id in range(6):
            folder = output/f'case-{case_id:02d}'
            case = summary['cases'][str(case_id)]
            refs = [np.load(folder/f'reference-{level}.npz') for level in (0, 1)]
            ref = refs[1]['fields']
            norm0 = np.linalg.norm(ref[0])
            reference_gap = float(np.linalg.norm(ref-refs[0]['fields'], axis=1).max()/norm0)
            ref_baseline = np.load(outputs[128]/f'case-{case_id:02d}'/'reference-1.npz')['fields']
            common_reference_gap = float(np.linalg.norm(ref-ref_baseline, axis=1).max()/norm0)
            reference_pass = bool(case['reference_eligible'] and reference_gap <= cfg['reference_max_disagreement']
                and common_reference_gap <= cfg['reference_max_disagreement'] and all(np.isfinite(z['fields']).all()
                and z['iterations'].max() < 20 and z['residual'].max() <= 1e-10 for z in refs))
            proj = np.load(folder/'projection.npz')
            projected = proj['coeff'] @ G.T
            projection_error = np.linalg.norm(projected-ref, axis=1)/norm0
            projection_metric_gap = float(np.max(np.abs(projection_error-proj['error'])))
            projection_orthogonality = float(np.linalg.norm((projected-ref) @ G/n**2)/(np.linalg.norm(ref)*np.linalg.norm(G)/n**2))
            rec = dict(reference_passed=reference_pass, reference_disagreement=reference_gap,
                common_baseline_reference_gap=common_reference_gap, projection_metric_gap=projection_metric_gap,
                projection_orthogonality=projection_orthogonality, projection_error=projection_error.tolist(), arms={})
            for m in cfg['solve_rows']:
                arm = case['arms'][str(m)]
                repetition_checks = []
                for level, gtol in cfg['solver_gtols'].items():
                    repetitions = cfg['timing_repetitions'] if level == 'tight' else 1
                    for rep in range(repetitions):
                        returned = np.load(folder/f'm{m}-{level}-rep{rep}.npz')
                        stats = returned['stats']
                        repetition_checks.append(dict(level=level, repetition=rep,
                            finite=bool(np.isfinite(returned['coeff']).all()),
                            stationary=bool(np.all(stats[:, 1] == 1) and np.all(stats[:, 2] <= gtol)),
                            saved_timing_matches=bool(float(returned['elapsed_seconds']) == arm['levels'][level]['timing_repetitions'][rep])))
                z = np.load(folder/f'm{m}-tighter-rep0.npz')
                field = z['coeff'] @ G.T
                metric = compare_metrics(field, ref, n)
                error = metric[:, 0]
                evolution_distance = np.linalg.norm(field-projected, axis=1)/norm0
                tight = np.load(folder/f'm{m}-tight-rep0.npz')
                tolerance_gap = float(np.linalg.norm((z['coeff']-tight['coeff']) @ G.T, axis=1).max()/norm0)
                threshold = min(1e-6, .01*error.max()) if error.max() >= 1e-4 else 1e-6
                stationary = all(np.all(zl['stats'][:, 1] == 1) and np.all(zl['stats'][:, 2] <= cfg['solver_gtols'][level])
                    and np.isfinite(zl['coeff']).all() for level, zl in [('tight', tight), ('tighter', z)])
                metric_gap = float(np.max(np.abs(metric-z['metrics'])))
                field_metric_gap = float(np.max(np.abs(error-z['metrics'][:, 0])))
                evolution_metric_gap = float(np.max(np.abs(evolution_distance-z['evolution_distance'])))
                pythagorean = float(np.max(np.abs(error**2-projection_error**2-evolution_distance**2)))
                residual = np.load(folder/f'm{m}-residuals.npz')
                shell = np.linalg.norm(residual['scaled'][:, 8192:9216], axis=1)
                unscaled_shell = np.linalg.norm(residual['unscaled'][:, 8192:9216], axis=1)
                array_pass = bool(projection_metric_gap <= 1e-10 and projection_orthogonality <= 1e-9
                    and field_metric_gap <= 1e-10 and evolution_metric_gap <= 1e-10 and metric_gap <= 1e-9
                    and pythagorean <= 1e-10 and stationary and tolerance_gap <= threshold
                    and all(r['finite'] and r['saved_timing_matches'] and (r['stationary'] or r['level'] == 'prod') for r in repetition_checks))
                eligible = bool(arm['numerically_eligible'] and reference_pass and array_pass and shared_bank_gap <= 1e-10
                    and audit['sources'][str(width)]['environment_passed'] and audit['sources'][str(width)]['numerical_basis_passed'])
                details = dict(error=error.tolist(), projection_error=projection_error.tolist(), evolution_distance=evolution_distance.tolist(),
                    gradient_error=metric[:, 1].tolist(), upwind_advection_error=metric[:, 2].tolist(), laplacian_error=metric[:, 3].tolist(),
                    energy=metric[:, 4].tolist(), independent_scaled_shell=shell.tolist(), independent_unscaled_shell=unscaled_shell.tolist(),
                    max_metric_difference=metric_gap, field_metric_difference=field_metric_gap, evolution_metric_difference=evolution_metric_gap,
                    pythagorean_max_absolute=pythagorean, tight_tighter_difference=tolerance_gap, tight_tighter_threshold=float(threshold),
                    repetition_checks=repetition_checks, saved_array_passed=array_pass, numerically_eligible=eligible)
                rec['arms'][str(m)] = details
                table.append(dict(R=width, M=m, case=case_id, max_discrete_error=float(error.max()),
                    initial_error=float(error[0]), evolved_error_max=float(error[1:].max()), final_error=float(error[-1]),
                    maximum_error_time=float(times[int(np.argmax(error))]), projection_error_max=float(projection_error.max()),
                    evolution_distance_max=float(evolution_distance.max()), shell_max=float(shell.max()),
                    unscaled_shell_max=float(unscaled_shell.max()), gradient_error_max=float(metric[:, 1].max()),
                    upwind_advection_error_max=float(metric[:, 2].max()), laplacian_error_max=float(metric[:, 3].max()),
                    tighter_iterations=int(z['stats'][:, 0].sum()), tighter_rejected=int(z['stats'][:, 4].sum()),
                    tight_tighter=tolerance_gap, eligible=eligible))
                if width == 128:
                    previous = lane/'runs'/('p03' if case_id < 2 else 'p04b')/'output'/f'case-{case_id:02d}'
                    old = np.load(previous/'m2048-tighter-rep0.npz')
                    old_source = previous.parent
                    old_field = old['coeff'] @ load_bank(old_source).T
                    curve_gap = float(np.max(np.abs(error-old['metrics'][:, 0])))
                    field_gap = float(np.linalg.norm(field-old_field, axis=1).max()/norm0)
                    audit['baseline_reproduction'].append(dict(case=case_id, maximum_error_curve_gap=curve_gap,
                        stationary_field_gap=field_gap, stationary_field_threshold=float(threshold),
                        passed=bool(eligible and curve_gap <= 1e-8 and field_gap <= threshold)))
            audit['cases'][f'R{width}-case{case_id}'] = rec
    baseline_ok = all(r['passed'] for r in audit['baseline_reproduction'])
    audit['baseline_reproduction_passed'] = baseline_ok
    audit['aggregates'] = {}
    for width in sorted(outputs):
        for m in environments[width]['config']['solve_rows']:
            rows = [r for r in table if r['R'] == width and r['M'] == m]
            audit['aggregates'][f'R{width}-M{m}'] = dict(case_count=len(rows), qualified_count=sum(r['eligible'] for r in rows),
                maximum_error_median=float(np.median([r['max_discrete_error'] for r in rows])),
                maximum_error_worst=float(max(r['max_discrete_error'] for r in rows)),
                evolved_error_worst=float(max(r['evolved_error_max'] for r in rows)),
                evolution_distance_worst=float(max(r['evolution_distance_max'] for r in rows)),
                initial_time_is_maximum_count=sum(r['maximum_error_time'] == 0 for r in rows),
                tighter_iterations_median=float(np.median([r['tighter_iterations'] for r in rows])),
                tighter_rejected_total=sum(r['tighter_rejected'] for r in rows))
    baseline_rows = {r['case']:r for r in table if r['R'] == 128}
    worst_baseline = max(r['max_discrete_error'] for r in baseline_rows.values())
    for width in sorted(outputs):
        if width == 128:
            continue
        for m in environments[width]['config']['solve_rows']:
            rows = [r for r in table if r['R'] == width and r['M'] == m]
            ratios = [r['max_discrete_error']/baseline_rows[r['case']]['max_discrete_error'] for r in rows]
            worst_ratio = max(r['max_discrete_error'] for r in rows)/worst_baseline
            audit['comparisons'].append(dict(R=width, M=m, baseline='R128-M2048', all_cases_eligible=all(r['eligible'] for r in rows),
                worst_error_ratio=worst_ratio, case_maximum_error_ratios=ratios, no_case_regression_above_two_percent=max(ratios) <= 1.02,
                prospective_primary_improvement_passed=bool(baseline_ok and all(r['eligible'] for r in rows) and worst_ratio <= .8 and max(ratios) <= 1.02)))
    pointwise = []
    registered_arms = [(width, m) for width in sorted(outputs) for m in environments[width]['config']['solve_rows']]
    for (previous_width, previous_m), (width, m) in itertools.combinations(registered_arms, 2):
        kind = 'within_width_test_change' if previous_width == width else ('common_M_width_change' if previous_m == m else 'joint_width_and_test_change')
        for case_id in range(6):
            old_arm = audit['cases'][f'R{previous_width}-case{case_id}']['arms'][str(previous_m)]
            new_arm = audit['cases'][f'R{width}-case{case_id}']['arms'][str(m)]
            difference = np.asarray(new_arm['error'])-np.asarray(old_arm['error'])
            uncertainty = old_arm['tight_tighter_difference']+new_arm['tight_tighter_difference']
            pointwise.append(dict(case=case_id, comparison_kind=kind, previous_R=previous_width, previous_M=previous_m, R=width, M=m,
                increased_time_count=int(np.sum(difference > 1e-10)), maximum_increase=float(max(0., difference.max())),
                maximum_signed_error_change=float(difference.max()), summed_solver_uncertainty=uncertainty,
                maximum_increase_exceeds_solver_uncertainty=bool(difference.max() > uncertainty),
                time_of_maximum_increase=float(times[int(np.argmax(difference))])))
    audit['all_saved_array_checks_passed'] = all(a['saved_array_passed'] for c in audit['cases'].values() for a in c['arms'].values())
    audit['all_arms_numerically_eligible'] = all(r['eligible'] for r in table)
    audit['matrix_complete'] = set(outputs) == {128, *admission['admitted_widths']}
    write(dest/'analysis.json', audit)
    write(dest/'table.json', table)
    write(dest/'pointwise-counterexamples.json', dict(scope='descriptive all-pairs audit of registered arms; primary gates unchanged',
        increase_count_threshold=1e-10, comparisons=pointwise))
    write(dest/'combined-source.json', dict(sources=audit['sources'], raw_summaries={str(k):v for k,v in source_summaries.items()}, scope=audit['scope']))
    plot(audit, times, dest)
    print(json.dumps({key:audit[key] for key in ['baseline_reproduction_passed', 'all_saved_array_checks_passed', 'all_arms_numerically_eligible', 'matrix_complete', 'aggregates', 'comparisons']}))


def plot(audit, times, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {128:'#56616c', 256:'#167f9c', 512:'#9a43a9'}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), sharex=True)
    for case_id, ax in enumerate(axes.flat):
        for width in sorted(map(int, audit['sources'])):
            case = audit['cases'][f'R{width}-case{case_id}']
            for m, arm in case['arms'].items():
                label = f'R{width}, M{m}' + ('' if arm['numerically_eligible'] else ' — unqualified')
                ax.plot(times, 100*np.asarray(arm['error']), color=colors[width], linestyle='-' if m == '2048' else '--', label=label)
            ax.plot(times, 100*np.asarray(case['projection_error']), color=colors[width], linestyle=':', alpha=.65)
        ax.set(title=f'Development case {case_id}', xlabel='Time', ylabel='Error / initial-field norm (%)')
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('MEASURED native-initialized trajectories — same-grid / same-step discrete target\nDotted lines: each frozen prefix’s best projection; all 51 times retained', fontsize=12)
    fig.tight_layout()
    fig.savefig(dest/'measured-width-trajectories.png', dpi=180)
    fig.savefig(dest/'measured-width-trajectories.svg')
    plt.close(fig)
    keys = list(audit['aggregates'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ['maximum_error_median','maximum_error_worst'], ['Median of six case maxima', 'Worst of six case maxima']):
        values = [100*audit['aggregates'][key][metric] for key in keys]
        ax.bar(np.arange(len(keys)), values, color=[colors[int(key.split('-')[0][1:])] for key in keys])
        ax.set_xticks(np.arange(len(keys)), [key.replace('-', '\n')+'\n'+str(audit['aggregates'][key]['qualified_count'])+'/6\nqualified' for key in keys], fontsize=8)
        ax.set(title=title, ylabel='Error / initial-field norm (%)')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('MEASURED same-grid / same-step accuracy — maximum includes initialization and all evolved times')
    fig.tight_layout()
    fig.savefig(dest/'measured-width-maxima.png', dpi=180)
    fig.savefig(dest/'measured-width-maxima.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
