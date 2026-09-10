"""Audit captured wave fields with NumPy/SciPy and generate the device-query report.

No experiment modules, JAX, model evaluation or new simulation are used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
from scipy.fft import dstn, idstn

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'frozen_mlp32_seed691200': 'MLP32', 'frozen_mlp16_seed691200': 'MLP16',
         'dst': 'DST FOM', 'rk4': 'RK4 FOM'}
BOUNDARIES = {'dirichlet': 'Reflective', 'absorbing': 'Absorbing'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def weights(n, bc):
    w = np.ones(n-1 if bc == 'dirichlet' else n+1)
    if bc == 'absorbing':
        w[[0, -1]] = .5
    return np.outer(w, w)/n**2, w


def squared_energy_norm(u, v, n, bc, c):
    mass, edge = weights(n, bc)
    if bc == 'dirichlet':
        z = np.pad(u, [(0, 0)]*(u.ndim-2)+[(1, 1), (1, 1)])
        potential = sum(np.sum(np.diff(z, axis=ax)**2, axis=(-2, -1)) for ax in (-2, -1))
    else:
        potential = np.sum(np.diff(u, axis=-2)**2*edge, axis=(-2, -1))
        potential += np.sum(np.diff(u, axis=-1)**2*edge[:, None], axis=(-2, -1))
    return np.sum(mass*v*v, axis=(-2, -1))+c*c*potential


def field_metrics(u, v, ut, vt, n, bc, c, vanishing):
    mass, _ = weights(n, bc)
    norm = lambda x: np.sqrt(np.sum(mass*x*x, axis=(-2, -1)))
    e = np.sqrt(squared_energy_norm(ut, vt, n, bc, c))
    arrays = ((norm(u-ut), norm(ut), float(norm(ut[0]))),
              (norm(v-vt), norm(vt), float(e[0])),
              (np.sqrt(squared_energy_norm(u-ut, v-vt, n, bc, c)), e, float(e[0])))
    result = {}
    for name, (absolute, current, initial) in zip(('displacement', 'velocity', 'energy_state'), arrays):
        zero = current <= 1e-14*initial
        rel = np.divide(absolute, current, out=np.full_like(current, np.nan), where=~zero)
        result[name] = dict(absolute=absolute, initial_normalized=absolute/initial,
            current_relative=rel, reference_norm=current, initial_scale=initial,
            reference_zero=zero, reference_vanishing=current <= vanishing*initial,
            max_initial_normalized=float(np.max(absolute/initial)),
            max_absolute=float(np.max(absolute)), max_current_relative=float('nan') if np.all(zero) else float(np.nanmax(rel)))
    result['mean_error'] = np.sum(mass*(u-ut), axis=(-2, -1))
    result['energy_fraction'] = squared_energy_norm(u, v, n, bc, c)/e[0]**2
    result['truth_energy_fraction'] = e**2/e[0]**2
    return result


def initial_fields(p, n, bc):
    cx, cy, sx, sy, amplitude, c, vx, vy, sigx, sigy = p
    axis = np.linspace(0, 1, n+1)
    if bc == 'dirichlet':
        axis = axis[1:-1]
    x, y = np.meshgrid(axis, axis, indexing='ij')
    tx, ty = (x-cx)/sx, (y-cy)/sy
    def bump(t):
        return np.where(abs(t) < 1, np.exp(1-1/np.maximum(1-t*t, 1e-30)), 0.)
    u = amplitude*bump(tx)*bump(ty)*np.exp(-.5*(((x-cx)/sigx)**2+((y-cy)/sigy)**2))
    dx = -2*tx/(sx*np.maximum(1-tx*tx, 1e-30)**2)-(x-cx)/sigx**2
    dy = -2*ty/(sy*np.maximum(1-ty*ty, 1e-30)**2)-(y-cy)/sigy**2
    return u, -c*u*(vx*dx+vy*dy)


def parameters(seed, count):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        sx, sy = rng.uniform(.36, .42, 2)
        cx = rng.uniform(sx+.025, 1-sx-.025)
        cy = rng.uniform(sy+.025, 1-sy-.025)
        amplitude, c = rng.uniform(.7, 1.3), rng.uniform(.85, 1.15)
        vx, vy = rng.uniform(-.5, .5, 2)
        if i % 4 == 0:
            vx = vy = 0.
        sigx, sigy = rng.uniform(.12, .16, 2)
        result.append([cx, cy, sx, sy, amplitude, c, vx, vy, sigx, sigy])
    return np.asarray(result)


def table(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def cleaned(value):
    if isinstance(value, dict):
        return {str(k): cleaned(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cleaned(v) for v in value]
    if isinstance(value, np.ndarray):
        return cleaned(value.tolist())
    if isinstance(value, np.generic):
        return cleaned(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def figures(rows, curves, cfg, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'MLP32': '#0072B2', 'MLP16': '#D55E00', 'DST FOM': '#009E73', 'RK4 FOM': '#009E73'}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, bc in enumerate(cfg['boundaries']):
        selected = [row for row in rows if row['boundary'] == bc]
        for name in dict.fromkeys(r['method'] for r in selected):
            rr = [r for r in selected if r['method'] == name]
            axes[0, col].plot([r['intervals'] for r in rr], [r['query_ms'] for r in rr],
                             'o-', label=name, color=colors[name])
        axes[0, col].set(title=BOUNDARIES[bc], ylabel='Complete GPU query (ms)', yscale='log',
                         xlabel='Intervals per axis', xticks=cfg['meshes'])
        axes[0, col].legend(fontsize=8)
        for name in ('MLP32', 'MLP16'):
            cc = [r for r in curves if r['boundary'] == bc and r['intervals'] == max(cfg['meshes'])
                  and r['method'] == name]
            initial = np.max([r['metrics']['displacement']['initial_normalized'] for r in cc], axis=0)
            current = np.nanmax([r['metrics']['displacement']['current_relative'] for r in cc], axis=0)
            times = np.arange(len(initial))*cfg['observation_dt']
            axes[1, col].plot(times, 100*current, color=colors[name], label=name+' / current norm')
            axes[1, col].plot(times, 100*initial, '--', color=colors[name], label=name+' / initial norm')
        axes[1, col].set(xlabel='Time', ylabel='Worst displacement error across cases (%)', yscale='log',
                        title=f"{max(cfg['meshes'])} intervals: both normalizations")
        axes[1, col].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Fresh-wave device-resident comparison — development cases')
    for suffix in ('png', 'pdf'):
        kwargs = {'dpi': 170} if suffix == 'png' else {'metadata': {'CreationDate': None, 'ModDate': None}}
        fig.savefig(dest.with_name(dest.stem+'-scaling').with_suffix('.'+suffix), **kwargs)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT/'reports/2026-09-10-fresh-wave-device-comparison.md')
    args = ap.parse_args()
    run = args.run.resolve(); out = run/'cluster/out/pilot'
    result = json.loads((out/'result.json').read_text())
    cfg, prov = result['config'], result['provenance']
    submission = json.loads((run/'submission.json').read_text())
    cleanup = json.loads((run/'cleanup.json').read_text())
    assert result['complete'] and not result['final_test_opened']
    assert prov['jax_backend'] == 'gpu' and prov['x64'] and prov['matmul_precision'] == 'highest'
    assert str(prov['job_id']) == str(submission['job_id'])
    assert prov['source_commit'] == submission['source_commit']
    assert cleanup['remote_deleted_and_absence_checked'] and cleanup['all_three_manifests_verified']
    logs = '\n'.join(p.read_text() for p in (run/'cluster/logs').glob('*'))
    assert 'jax_backend=gpu' in logs and 'fresh_wave_device_replay_complete' in logs
    for bad in ('captured constant', 'large constant', 'Out of memory', 'OUT_OF_MEMORY', 'No space left', 'Traceback'):
        assert bad.lower() not in logs.lower(), bad
    manifest = {}
    tree = run.parents[3]
    for rel, expected in submission['source_hashes'].items():
        assert sha(run/'cluster'/rel) == expected
        git_path = 'experiments/'+str(Path(rel).relative_to('code'))
        committed = subprocess.check_output(['git', '-C', str(tree), 'show',
            submission['source_commit']+':'+git_path])
        assert hashlib.sha256(committed).hexdigest() == expected
    for name, expected in result['output_sha256'].items():
        assert sha(out/name) == expected
        manifest[str((out/name).relative_to(ROOT))] = expected
    for p in (out/'result.json', run/'submission.json', run/'cleanup.json', Path(__file__).resolve()):
        manifest[str(p.relative_to(ROOT))] = sha(p)
    expected_pars = parameters(cfg['validation_seed'], max(cfg['validation_indices'])+1)
    audit = dict(field_groups=0, timed_invocations=0, metric_array_checks=0,
                 maximum_metric_absolute_difference=0., reference_checks=[], step_halving_checks=[])
    curves = []; groups = defaultdict(list)
    assert len(result['invocations']) == cfg['expected_timed_invocations']
    assert len(result['accuracy_controls']) == cfg['expected_accuracy_only_queries']
    for row in result['invocations']:
        groups[row['field_artifact']].append(row)
    for row in result['accuracy_controls']:
        groups[row['field_artifact']].append(row)
    def compare(actual, expected):
        aa = np.asarray(actual, dtype=float); ee = np.asarray(expected, dtype=float)
        np.testing.assert_allclose(aa, ee, rtol=2e-10, atol=2e-12, equal_nan=True)
        delta = np.abs(aa-ee)
        finite = delta[np.isfinite(delta)]
        if finite.size:
            audit['maximum_metric_absolute_difference'] = max(audit['maximum_metric_absolute_difference'], float(np.max(finite)))
        audit['metric_array_checks'] += 1
    for refrow in result['references']:
        bc, n, ci = refrow['boundary'], refrow['intervals'], refrow['case']
        with np.load(out/f'reference_{bc}_{n}_{ci}.npz') as f:
            ut, vt, u0, v0, par = [f[k] for k in ('u', 'v', 'u0', 'v0', 'parameters')]
        np.testing.assert_array_equal(par, expected_pars[ci])
        for actual, expected in zip((u0, v0), initial_fields(par, n, bc)):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=5e-14)
        assert array_sha(u0) == refrow['input_sha256']['u']
        assert array_sha(v0) == refrow['input_sha256']['v']
        mass, _ = weights(n, bc)
        refaudit = dict(boundary=bc, intervals=n, case=ci, recorded_refinement_passed=refrow['refinement_passed'])
        if bc == 'dirichlet':
            modes = np.arange(1, n)
            lam = 4*n*n*np.sin(np.pi*modes/(2*n))**2
            omega = par[5]*np.sqrt(lam[:, None]+lam[None, :])
            a, b = dstn(u0, type=1, norm='ortho'), dstn(v0, type=1, norm='ortho')
            deviations = []
            for ti in range(len(ut)):
                t = ti*cfg['observation_dt']; co, si = np.cos(omega*t), np.sin(omega*t)
                ru = idstn(co*a+si/omega*b, type=1, norm='ortho')
                rv = idstn(-omega*si*a+co*b, type=1, norm='ortho')
                deviations.append(max(float(np.max(abs(ru-ut[ti]))), float(np.max(abs(rv-vt[ti])))))
            refaudit['independent_scipy_max_absolute_discrepancy'] = max(deviations)
            assert max(deviations) < 1e-10
        else:
            boundary = np.zeros(n+1); boundary[[0, -1]] = 2*n*par[5]
            invariant = np.sum(mass*(vt+(boundary[:, None]+boundary)*ut), axis=(-2, -1))
            refaudit['invariant_drift'] = float(np.max(abs(invariant-invariant[0])))
            assert refaudit['invariant_drift'] < 1e-10
            refaudit['recorded_temporal_refinement_max'] = max(refrow['temporal_refinement'][key]['max_initial_normalized']
                for key in ('displacement', 'velocity', 'energy_state'))
        audit['reference_checks'].append(refaudit)
        for artifact, records in groups.items():
            row = records[0]
            if (row['boundary'], row['intervals'], row['case']) != (bc, n, ci):
                continue
            with np.load(out/artifact) as f:
                u, v = f['u'], f['v']
            assert u.dtype == np.float64 and v.dtype == np.float64 and u.shape == ut.shape and v.shape == vt.shape
            actual = field_metrics(u, v, ut, vt, n, bc, par[5], cfg['vanishing_fraction'])
            for record in records:
                assert record['output_sha256'] == dict(u=array_sha(u), v=array_sha(v))
                assert record['output_bytes'] == u.nbytes+v.nbytes
                for key, value in actual.items():
                    if isinstance(value, dict):
                        for subkey, val in value.items():
                            compare(val, record['same_grid_discrepancy'][key][subkey])
                    else:
                        compare(value, record['same_grid_discrepancy'][key])
                if record['comparison_eligible']:
                    seconds = record['seconds']
                    compare(seconds['complete_device_query'], sum(seconds[k] for k in
                        ('initialization_and_parameter_projection', 'evolution', 'dense_device_output')))
                    assert seconds['complete_device_query'] > 0
                    audit['timed_invocations'] += 1
            curves.append(dict(boundary=bc, intervals=n, case=ci, method=NAMES[row['method']],
                               primary=row['comparison_eligible'], metrics=actual))
            audit['field_groups'] += 1
            print('audited', artifact, flush=True)
        for method in (cfg['primary_method'], cfg['control_method']):
            ident = f'{bc}_{n}_{ci}_{method}'
            with np.load(out/(ident+'.npz')) as coarse, np.load(out/(ident+'_refined.npz')) as fine:
                differences = field_metrics(coarse['u'], coarse['v'], fine['u'], fine['v'],
                                            n, bc, par[5], cfg['vanishing_fraction'])
            reference_scales = field_metrics(ut[:1], vt[:1], ut[:1], vt[:1], n, bc, par[5], cfg['vanishing_fraction'])
            maxima = {key: float(np.max(differences[key]['absolute'])/reference_scales[key]['initial_scale'])
                      for key in ('displacement', 'velocity', 'energy_state')}
            recorded = next(r for r in result['time_refinement'] if
                (r['boundary'], r['intervals'], r['case'], r['method']) == (bc, n, ci, method))
            for key, value in maxima.items():
                compare(value, recorded['maxima_on_reference_initial_scales'][key])
            refined_record = next(r for r in result['accuracy_controls'] if r['invocation_id'] == ident+'_refined')
            assert bool(refined_record['completed'] and max(maxima.values()) <= cfg['rom_refinement_target']) == recorded['passed']
            audit['step_halving_checks'].append(dict(boundary=bc, intervals=n, case=ci, method=method,
                                                     maxima=maxima, passed=recorded['passed']))
    rows = []
    for bc in cfg['boundaries']:
        for n in cfg['meshes']:
            methods = (cfg['primary_method'], cfg['control_method'], 'dst' if bc == 'dirichlet' else 'rk4')
            for method in methods:
                selected = [r for r in result['invocations'] if (r['boundary'], r['intervals'], r['method']) == (bc, n, method)]
                bycase = [[r for r in selected if r['case'] == ci] for ci in cfg['validation_indices']]
                assert all(sorted(r['repetition'] for r in rr) == list(range(cfg['repetitions'])) for rr in bycase)
                primary = [rr[0] for rr in bycase]
                timings = {k: 1000*median(median(r['seconds'][k] for r in rr) for rr in bycase)
                           for k in selected[0]['seconds']}
                errs = {}
                for field in ('displacement', 'velocity', 'energy_state'):
                    errs[field] = {}
                    for norm in ('current_relative', 'initial_normalized'):
                        values = [np.asarray(r['same_grid_discrepancy'][field][norm], float) for r in primary]
                        errs[field][norm] = dict(mean=float(np.nanmean(values)),
                            median_case_mean=float(np.median([np.nanmean(x) for x in values])), worst=float(np.nanmax(values)))
                rows.append(dict(boundary=bc, intervals=n, method=NAMES[method], query_ms=timings['complete_device_query'],
                    timing_components_ms=timings, errors=errs,
                    raw_timings_seconds=[[r['seconds']['complete_device_query'] for r in rr] for rr in bycase],
                    timing_outliers=sum(r['seconds']['complete_device_query'] > 1.5*median(x['seconds']['complete_device_query'] for x in rr)
                                        for rr in bycase for r in rr),
                    completed_cases=sum(r['completed'] for r in primary),
                    stationary_fits=sum(r.get('fit_stationary', False) for r in primary) if method.startswith('frozen') else None,
                    case_count=len(primary)))
            fom = rows[-1]['query_ms']
            for row in rows[-len(methods):]:
                row['fom_over_rom'] = fom/row['query_ms']
    data = dict(provenance=prov, config=cfg, rows=rows, audit=audit,
                source_sha256=manifest, curves=curves, time_refinement=result['time_refinement'],
                reference_acceptance=all(x['refinement_passed'] for x in result['references']))
    figures(rows, [c for c in curves if c['primary']], cfg, args.out)
    headline = []
    evolution_share = []
    for r in rows:
        if r['method'] == NAMES[cfg['primary_method']] and r['intervals'] == max(cfg['meshes']):
            ratio = r['fom_over_rom']
            relation = f'{ratio:.3f} times faster' if ratio > 1 else f'{1/ratio:.3f} times slower'
            headline.append(f"{BOUNDARIES[r['boundary']]}: the primary head is {relation} than its named FOM, "
                f"with mean/worst current-relative displacement error "
                f"{100*r['errors']['displacement']['current_relative']['mean']:.3f}% / "
                f"{100*r['errors']['displacement']['current_relative']['worst']:.3f}%.")
            evolution_share.append(f"{BOUNDARIES[r['boundary']].lower()} "
                f"{100*r['timing_components_ms']['evolution']/r['query_ms']:.3f}%")
    lines = ['# Fresh reflective and absorbing wave comparison on the GPU', '',
        'This report measures the fresh, verified wave models under the user-requested device-resident comparison. Numbers are provisional development results from existing validation cases; they do not establish final paper accuracy or a continuum-error guarantee.', '',
        f"At {max(cfg['meshes'])} intervals per axis: "+' '.join(headline), '',
        f"All rows use allocation {prov['job_id']} on {', '.join(prov['device_kind'])}, f64 and highest matrix precision. Source commit: `{prov['source_commit']}`. Only post-reset wave mathematics and checkpoints are used; the discarded earlier wave experiments remain excluded.", '',
        f"The frozen learned spatial bank has rank {cfg['frozen_bank_rank']}. MLP32 is the preselected primary head and MLP16 is its control, both using optimizer seed {cfg['primary_method'].split('seed')[-1]}. Neither head is retrained for these meshes. Each boundary uses {len(cfg['validation_indices'])} existing development cases, generated from seed {cfg['validation_seed']}, indices {cfg['validation_indices']}.", '',
        f"The unit-square wave starts from a Gaussian core with a smooth compact cutoff, with varying position, width, amplitude, propagation speed and initial velocity. Both displacement and velocity fields are supplied to the ROM. The horizon is {cfg['end_time']:g}, with {round(cfg['end_time']/cfg['observation_dt'])+1} full outputs. The primary ROM step is {cfg['primary_dt']:g}; {cfg['accuracy_only_dt']:g} is an accuracy-only refinement, excluded from speed selection.", '',
        'The timer starts with ready full input fields on the GPU and includes full projection, every initial-fit start, speed-dependent operator preparation, evolution, and full GPU displacement/velocity outputs. Mesh-only assembly, compilation and host copies are excluded. Reflective Dirichlet boundaries store interior unknowns with prescribed zero boundary values; absorbing boundaries include boundary unknowns. Intervals per axis therefore differ from stored nodes per axis.', '',
        'Reduced stiffness and boundary-damping matrices are preassembled; no empirical quadrature fit occurs in these queries. The reduced evolution uses fixed model dimensions, while full-field projection and decoding grow with the mesh.', '',
        'The replay encloses projection and every initial-fit start in one compiled initializer. A frozen-head GPU smoke check verified numerical parity with the earlier fresh-wave query algorithm. This also changes execution overhead; a difference from earlier host-query timings cannot be attributed only to omitted host transfers. All speed ratios below compare the methods within this allocation.', '',
        '## Same-grid query costs', '',
        'The reflective FOM uses an exact propagator for the discrete spatial operator, evaluated by sine transforms. The absorbing FOM uses the verified boundary-damped RK4 solver. These are named same-grid comparisons; there is no coarse-grid FOM selection.', '',
        table(['Boundary', 'Intervals/axis', 'Method', 'Query ms', 'Initialization / evolution / output ms', 'FOM / method', 'Timing outliers'],
              [[BOUNDARIES[r['boundary']], r['intervals'], r['method'], f"{r['query_ms']:.3f}",
                ' / '.join(f"{r['timing_components_ms'][k]:.3f}" for k in ('initialization_and_parameter_projection', 'evolution', 'dense_device_output')),
                f"{r['fom_over_rom']:.6g}", f"{r['timing_outliers']}/{cfg['repetitions']*r['case_count']}"] for r in rows]), '',
        f"Times are medians across cases of per-case medians from {cfg['repetitions']} repetitions. Component medians need not sum exactly to the query median. Timing outliers exceed 1.5 times their own case's repetition median; this diagnostic does not remove any measurement. FOM/method above one means less raw device-query time, independently of accuracy.", '',
        f"At the largest mesh, reduced evolution accounts for the following ratios of median component time to median primary query time: {', '.join(evolution_share)}. The frozen RK4 implementation evaluates decoder geometry, QR and SVD at every stage: {4*round(cfg['end_time']/cfg['primary_dt'])+1} evaluations per primary trajectory, including initialization. The timer establishes evolution as the bottleneck. It does not separately identify the contributions of those operations; that requires profiling or a controlled follow-up. Their repeated cost is independent of the full spatial mesh, but it can still exceed a named FOM's cost.", '',
        '## Accuracy of those timed outputs', '',
        'Displacement error uses the mass-weighted L2 norm. Current-relative divides by the reference field at that time; initial-normalized divides by its initial displacement norm. The energy-state norm combines displacement-gradient and velocity error. Velocity initial normalization uses the initial energy-state norm so zero initial velocity is defined. Undefined current-relative zero-reference entries remain null in the evidence JSON; vanishing reference fields are flagged rather than hidden.', '',
        table(['Boundary', 'Intervals', 'Method', 'Displacement current mean / median case / worst (%)', 'Displacement initial worst (%)', 'Velocity current worst (%)', 'Energy-state current worst (%)', 'Completed; stationary initial fits'],
              [[BOUNDARIES[r['boundary']], r['intervals'], r['method'],
                ' / '.join(f"{100*r['errors']['displacement']['current_relative'][k]:.3f}" for k in ('mean', 'median_case_mean', 'worst')),
                f"{100*r['errors']['displacement']['initial_normalized']['worst']:.3f}",
                f"{100*r['errors']['velocity']['current_relative']['worst']:.3f}",
                f"{100*r['errors']['energy_state']['current_relative']['worst']:.3f}",
                f"{r['completed_cases']}/{r['case_count']}; "+('n/a' if r['stationary_fits'] is None else f"{r['stationary_fits']}/{r['case_count']}")] for r in rows]), '',
        'Mean averages valid case-time entries; median case is the median of case time means; worst is the maximum across all cases and output times. A nonstationary initial fit still returns a scored field but does not certify a converged minimizer.', '',
        f"![Costs and both displacement-error normalizations]({args.out.stem}-scaling.png)", '',
        f"[Download the figure as PDF]({args.out.stem}-scaling.pdf). Both figure rows use logarithmic vertical axes.", '',
        '## Temporal checks and independent audit', '',
        f"The absorbing timing CFL is {cfg['fom_cfl']:g}. Its scoring reference uses CFL {cfg['reference_refinement_cfl']:g}, checked against {cfg['reference_cfl']:g} on the same mesh; the declared maximum initial-normalized difference target is {cfg['reference_refinement_target']:g}. The recorded reference checks pass: {data['reference_acceptance']}. Reflective scoring is exact in time for the discrete spatial operator.", '',
        table(['Boundary', 'Intervals', 'Case', 'Method', 'Step-halving displacement / velocity / energy difference (%)', 'Pass'],
              [[BOUNDARIES[r['boundary']], r['intervals'], r['case'], NAMES[r['method']],
                ' / '.join(f"{100*r['maxima_on_reference_initial_scales'][k]:.5f}" for k in ('displacement', 'velocity', 'energy_state')), r['passed']]
               for r in result['time_refinement']]), '',
        f"ROM refinement uses the same physical reference initial scales and target {cfg['rom_refinement_target']:g}. Failed refinement rows remain unresolved; their raw timings do not establish a temporally resolved accuracy result. Refined output timings may include compilation and are excluded from comparisons.", '',
        f"Root independently checked {audit['field_groups']} full displacement/velocity artifact pairs, {audit['timed_invocations']} timed invocation identities and {audit['metric_array_checks']} metric arrays/scalars with NumPy. The {len(audit['step_halving_checks'])} ROM step-halving checks were reconstructed from both saved fields. Later timed repetitions share a full artifact only after both complete output hashes match. Every artifact and staged source hash was checked, including source content against the recorded git commit. All reflective references were also compared against an independent SciPy sine propagator, and the absorbing references' conserved moment was checked from full fields. The recorded absorbing coarse/fine reference difference is retained from the run; root does not claim an independent reconstruction of the unsaved coarser reference.", '',
        f"The maximum recomputed metric absolute departure is {audit['maximum_metric_absolute_difference']:.6e}. This is an audit discrepancy, not a model-error bound. Full fields and original timing arrays are archived in [{run.name}]({Path('..')/run.relative_to(ROOT)}). The exact remote attempt was checksum-collected and deleted. Existing experiment branches remain separate.", '',
        '## Glossary', '',
        '- FOM: the full spatially discretized wave solver. ROM: the reduced model using a learned decoder.',
        '- MLP16 / MLP32: nonlinear decoder heads with the named configuration-space dimensions. The velocity uses the decoder tangent; it is also part of the dynamical state.',
        '- Frozen bank / head: learned spatial functions and nonlinear map whose weights are unchanged for this comparison. Rank is the number of bank functions.',
        '- Reflective / Dirichlet: zero boundary displacement, so outgoing waves reflect. Absorbing: the verified damping boundary operator allows energy to leave.',
        '- DST: discrete sine transform, used to propagate the reflective discrete wave exactly in time. RK4: fourth-order Runge–Kutta stepping. CFL: a step-size factor relative to grid spacing and wave speed.',
        '- QR / SVD: matrix factorizations used to solve the tangent-space equation and check whether the decoder Jacobian is close to losing rank. Decoder geometry includes its value, tangent and directional curvature.',
        '- Device-resident query: ready GPU input through ready full GPU output. Initial fitting finds latent coordinates from the supplied field. Stationary indicates the configured optimization and conditioning checks passed.',
        '- Same-grid: both methods refer to the same spatially discrete equation. Intervals are mesh cells per axis; prescribed reflective boundary values are omitted from stored unknowns.',
        '- Current-relative / initial-normalized: error divided by the current reference norm / the named initial reference scale. Mean, median case and worst use the aggregation described beside the table.',
        '- Energy-state norm: mass-weighted velocity plus stiffness-weighted displacement-gradient norm. Conserved moment: the discrete spatial integral of velocity plus the absorbing boundary contribution.',
        '- Refinement: halve the time step and compare fields. Its pass threshold is a development check, not a proof of total error. Accuracy-only rows cannot be selected for speed claims.',
        '- Development cases: existing validation inputs, not a newly opened independent final-test set. Timing outliers are counted, retained repetitions. Provisional means the limited cohort and declared checks do not certify final paper results.', '']
    args.out.write_text('\n'.join(lines))
    args.out.with_suffix('.json').write_text(json.dumps(cleaned(data), indent=2, allow_nan=False)+'\n')
    print(json.dumps(cleaned(audit), indent=2))


if __name__ == '__main__':
    main()
