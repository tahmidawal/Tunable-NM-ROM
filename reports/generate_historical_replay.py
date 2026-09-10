"""Generate the historical-protocol replay report and independently check fields.

Run after both owners have checksum-collected their completed mesh ladders.
No numerical experiment modules or JAX are imported.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOLD = ROOT / 'worktrees/2026-08-29-b2d-tensor/experiments/separable-decoder'
QOLD = ROOT / 'worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]])


def scaling_figure(burgers, poisson, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    blue, orange, green, gray = '#0072B2', '#D55E00', '#009E73', '#666666'
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, rows in enumerate((burgers, poisson)):
        xs = [r['N'] for r in rows]
        ax = axes[0, col]
        ax.plot(xs, [r['rom_ms'] for r in rows], 'o-', color=blue,
                label='Tensor ROM query' if col == 0 else 'QF ROM query')
        if col == 0:
            ax.plot(xs, [r['fom_ms'] for r in rows], 's-', color=orange,
                    label='Original dense-preconditioned FOM')
            ax.plot(xs, [r['solve_ms'] for r in rows], '^--', color=gray,
                    label='Reduced evolution only')
        else:
            ax.plot(xs, [r['cg_ms'] for r in rows], 's-', color=orange,
                    label='Original CG')
            ax.plot(xs, [r['direct_ms'] for r in rows], 'd-', color=green,
                    label='Original direct spectral')
            ax.plot(xs, [r['full_ms'] for r in rows], '^--', color=gray,
                    label='Full-grid weak ROM')
        ax.set_yscale('log')
        ax.set_ylabel('GPU time (ms; logarithmic scale)')
        ax.set_title('Burgers' if col == 0 else 'Poisson')
        ax.legend(fontsize=8)
        err_ax = axes[1, col]
        err_ax.plot(xs, [100*r['mean_error'] for r in rows], 'o-', color=blue,
                    label='Mean ROM relative error')
        err_ax.plot(xs, [100*r['worst_error'] for r in rows], 's--', color=orange,
                    label='Worst time/case' if col == 0 else 'Worst source')
        err_ax.set_ylabel('Relative field error (%)')
        err_ax.set_ylim(bottom=0)
        err_ax.legend(fontsize=8)
        for panel in axes[:, col]:
            panel.set_xscale('log', base=2)
            panel.set_xticks(xs, [str(x) for x in xs])
            panel.set_xlabel('Nodes per axis, including boundaries')
            panel.grid(True, alpha=0.22)
    fig.suptitle('Historical device-resident comparison — development cohorts', fontsize=13)
    fig.supxlabel(
        f'GPU input and full GPU output; same-grid references. '
        f'Burgers: {burgers[0]["cases"]} trajectories. '
        f'Poisson: {poisson[0]["cases"]} held-out sources, tau={poisson[0]["tau"]:g}.',
        fontsize=8)
    for suffix in ('pdf', 'png'):
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None
        fig.savefig(dest.with_name(dest.name+'-scaling').with_suffix('.'+suffix), dpi=200,
                    metadata=metadata)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--burgers-dir', type=Path, required=True)
    ap.add_argument('--poisson-dir', type=Path, required=True)
    ap.add_argument('--stem', default='2026-09-10-historical-burgers-poisson-replay')
    args = ap.parse_args()
    sources, checks = {}, []
    field_checks = []

    def remember(path):
        key = str(path.resolve().relative_to(ROOT))
        sources[key] = dict(sha256=digest(path), bytes=path.stat().st_size)
        return key

    def read(path):
        remember(path)
        return json.loads(path.read_text())

    def agree(label, actual, expected, atol=1e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, (label, a.shape, b.shape)
        delta = float(np.max(np.abs(a-b)))
        assert np.all(np.isfinite(a)) and np.allclose(a, b, rtol=1e-10, atol=atol), (label, delta)
        checks.append(dict(label=label, max_absolute_difference=delta))

    def timing(groups, label, recorded_ms):
        groups = list(groups)
        assert groups and all(len(g) >= 3 for g in groups), label
        assert all(np.all(np.isfinite(g)) and min(g) > 0 for g in groups), label
        mid = median(median(g) for g in groups)
        agree(label, mid, recorded_ms)
        return sum(sum(t > 2*median(g) for t in g) for g in groups), sum(map(len, groups))

    burgers = []
    for n in (64, 256, 512, 1024):
        path = args.burgers_dir / f'n{n}/sep_b2d_tensor_n{n}.json'
        d = read(path)
        old = read(BOLD / f'runs/b2dtensor/n{n}/out/sep_b2d_tensor_n{n}.json')
        c = d['config']
        assert d['complete'] and c['backend'] == 'gpu'
        assert c['N'] == n
        assert c['x64'] and c['matmul_precision'] == 'highest'
        for key in ('N', 'k', 'r', 'M', 'm_nnls', 'n_test', 'seed', 'data_seed', 'test_seed',
                    'n_train_traj', 'n_val_traj', 'step_tol', 'stall', 'extrap', 'tr_factor',
                    'gn_budget', 'ic_enc_budget', 'enc_steps', 'num_steps', 'dt', 'newton_tols', 'lin_fracs'):
            assert c[key] == old['config'][key], (n, key, c[key], old['config'][key])
        v = d['variants']['tensor']
        match = d['matched']['arms']['tensor']
        pair = match['paired']
        assert len(pair['per_traj']) == c['n_test']
        assert all(len(p[k]) == c['pair_reps'] for p in pair['per_traj'] for k in ('a_raw_ms', 'b_raw_ms'))
        out_a, count = timing((p['a_raw_ms'] for p in pair['per_traj']), f'B{n} paired ROM', pair['rom_ms'])
        out_b, _ = timing((p['b_raw_ms'] for p in pair['per_traj']), f'B{n} paired FOM', pair['fom_ms'])
        pts = np.asarray([p['timed_output_errors']['a']['per_time'] for p in pair['per_traj']])
        fom_pts = np.asarray([p['timed_output_errors']['b']['per_time'] for p in pair['per_traj']])
        fom_rung = next(r for r in d['fom'] if r['newton_tol'] == match['matched']['newton_tol']
                       and r['lin_tol'] == match['matched']['lin_tol'])
        agree(f'B{n} paired FOM vs native rung', fom_pts, [p['per_time'] for p in fom_rung['per_traj']])
        agree(f'B{n} timed-pair vs native error', pts, [p['per_time'] for p in v['per_traj']])
        agree(f'B{n} native mean', float(pts.mean()), v['err_traj_rel_mean'])
        agree(f'B{n} paired ratio', pair['fom_ms']/pair['rom_ms'], pair['speedup'])
        roll = [t*1000 for p in v['per_traj'] for t in p['raw_s']['roll']]
        agree(f'B{n} latent median', median(roll), v['roll_ms_median'])
        fp = path.with_name(f'sep_b2d_tensor_n{n}_tensor_full_case0.npz')
        remember(fp)
        with np.load(fp) as fields:
            truth = fields['truth'].reshape(len(pts[0]), -1)
            assert truth.shape == (len(pts[0]), n*n)
            assert int(fields['N']) == n and int(fields['case']) == 0
            agree(f'B{n} captured dt', float(fields['dt']), c['dt'])
            for side, expected in [('rom', pts[0]), ('fom', fom_pts[0])]:
                pred = fields[side].reshape(truth.shape)
                assert pred.dtype == np.float64 and truth.dtype == np.float64
                err = np.linalg.norm(pred-truth, axis=1)/np.linalg.norm(truth, axis=1)
                agree(f'B{n} full case0 {side}', err, expected)
                field_checks.append(dict(pde='burgers', N=n, case=0, method=side,
                                         states=len(err), maximum_error=float(max(err))))
        burgers.append(dict(N=n, job=c['slurm_job'], gpu=c['gpu'], k=c['k'], r=c['r'],
            M=c['M'], cases=c['n_test'], solve_ms=v['roll_ms_median'],
            steps=c['num_steps'], dt=c['dt'], final_time=c['num_steps']*c['dt'],
            archived_training_steps=c['ckpt_cfg']['steps'],
            archived_max_training_snapshots=c['ckpt_cfg']['max_snaps'],
            rom_ms=pair['rom_ms'], fom_ms=pair['fom_ms'], ratio=pair['speedup'],
            mean_error=float(pts.mean()), median_case_mean_error=float(np.median(pts.mean(axis=1))),
            worst_error=float(pts.max()), fom_mean_error=float(fom_pts.mean()),
            fom_worst_error=float(fom_pts.max()), rom_outliers=out_a, fom_outliers=out_b,
            repetitions_per_method=count, newton_tol=match['matched']['newton_tol'],
            linear_tol=match['matched']['lin_tol'], stop_reasons=v['stop_reasons'],
            blowups=v['n_blowups'],
            archived_mean_error=old['variants']['tensor']['err_traj_rel_mean'],
            archived_error_difference=float(pts.mean()-old['variants']['tensor']['err_traj_rel_mean']),
            archived_per_time_max_difference=float(np.max(np.abs(pts-np.asarray([
                p['per_time'] for p in old['variants']['tensor']['per_traj']])))),
            path=str(path.resolve().relative_to(ROOT))))

    poisson = []
    initial_attempt = read(args.poisson_dir.parent.parent / 'replay01/attempt-status.json')
    for n in (128, 256, 512, 1024):
        path = args.poisson_dir / f'poisson_n{n}.json'
        d = read(path)
        c = d['config']
        assert d['complete'] and c['backend'] == 'gpu'
        assert c['N'] == n
        assert c['x64'] and c['matmul_precision'] == 'highest'
        fp = path.parent / d['fields_file']
        field_key = remember(fp)
        assert sources[field_key]['sha256'] == d['fields_sha256']
        with np.load(fp) as fields:
            truth = fields['truth']
            assert truth.dtype == np.float64
            assert truth.shape == (2*c['n_src_per_cohort'], n-2, n-2)
            truth_norms = np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
            supplied = fields['sources']
            assert supplied.shape == truth.shape and supplied.dtype == np.float64
            padded = np.pad(truth, ((0, 0), (1, 1), (1, 1)))
            minus_lap = -(padded[:, 2:, 1:-1] + padded[:, :-2, 1:-1]
                          + padded[:, 1:-1, 2:] + padded[:, 1:-1, :-2]
                          - 4*truth) / (1/(n-1))**2
            residuals = np.linalg.norm((minus_lap-supplied).reshape(len(truth), -1), axis=1)/np.linalg.norm(supplied.reshape(len(truth), -1), axis=1)
            numpy_truth_residual_max = float(max(residuals))
            assert numpy_truth_residual_max < c['fom_res_tol'], (n, numpy_truth_residual_max)
            del supplied, padded, minus_lap
            # Load one method at a time; each saved array contains both cohorts.
            for key in sorted({r['field_key'] for r in d['rows']}):
                pred = fields[key]
                assert pred.dtype == np.float64 and pred.shape == truth.shape
                errors = np.linalg.norm((pred-truth).reshape(len(truth), -1), axis=1)/truth_norms
                if n == 1024 and key == 'spectral_dense':
                    assert max(errors) < 1e-11, 'largest-mesh captured direct/reference disagreement'
                for row in [r for r in d['rows'] if r['field_key'] == key]:
                    ix = row['source_ids']
                    assert len(ix) == row['n_sources'] == c['n_src_per_cohort']
                    assert set(map(str, ix)) == set(row['time_raw_s'])
                    assert all(len(g) == c['reps'] for g in row['time_raw_s'].values())
                    agree(f'P{n} {key} {row["cohort"]} full fields', errors[ix], row['err_rel_l2_all'])
                    agree(f'P{n} {key} mean error', mean(row['err_rel_l2_all']), row['err_rel_l2'])
                    agree(f'P{n} {key} max error', max(row['err_rel_l2_all']), row['err_rel_l2_max'])
                    timing(([t*1000 for t in g] for g in row['time_raw_s'].values()),
                           f'P{n} {key} {row["cohort"]} median', row['time_ms'])
                field_checks.append(dict(pde='poisson', N=n, method=key, cases=len(errors), maximum_error=float(max(errors))))
        for row in [r for r in d['rows'] if r['method'] == 'qf']:
            cohort = row['cohort']
            cg = min((r for r in d['rows'] if r['method'] == 'cg' and r['cohort'] == cohort
                      and r['err_rel_l2'] <= row['err_rel_l2']), key=lambda r:r['time_ms'])
            direct = next(r for r in d['rows'] if r['method'] == 'spectral_dense' and r['cohort'] == cohort)
            full = next(r for r in d['rows'] if r['method'] == 'full' and r['cohort'] == cohort and r['tau'] == row['tau'])
            outliers = [sum(sum(t > 2*median(g) for t in g) for g in r['time_raw_s'].values()) for r in (row, cg, direct)]
            poisson.append(dict(N=n, job=c['slurm_job'], gpu=c['gpu'], k=c['k'], r=c['r'],
                M=c['M'], cohort=cohort, tau=row['tau'], cases=row['n_sources'],
                rom_ms=row['time_ms'], full_ms=full['time_ms'], cg_ms=cg['time_ms'],
                direct_ms=direct['time_ms'], ratio=cg['time_ms']/row['time_ms'], cg_tol=cg['fom_tol'],
                mean_error=row['err_rel_l2'], median_error=row['err_rel_l2_median'],
                worst_error=row['err_rel_l2_max'], cg_mean_error=cg['err_rel_l2'],
                direct_mean_error=direct['err_rel_l2'], censored_fraction=row['censored_frac'],
                truth_residual_limit=c['fom_res_tol'],
                truth_residual=d['gates']['truth_residual_max'],
                truth_numpy_residual=numpy_truth_residual_max,
                truth_vs_direct=d['gates'].get('truth_vs_direct_relative_max'),
                stop_reasons=row['stop_reasons'], rom_cg_direct_outliers=outliers,
                repetitions_per_method=row['total_repetitions'], extension=c['historical_qf_extension'],
                path=str(path.resolve().relative_to(ROOT))))
        if n != 1024:
            old = read(QOLD / f'runs/b1dqf/qf_n{n}/out/sep_poisson_qf_qf_N{n}.json')
            for row in [r for r in poisson if r['N'] == n]:
                oldrow = next(r for r in old['rows'] if r['method'] == 'qf' and r['cohort'] == row['cohort'] and r['tau'] == row['tau'])
                row['archived_mean_error'] = oldrow['err_rel_l2']
                row['archived_error_difference'] = row['mean_error']-oldrow['err_rel_l2']

    for name, rows in [('burgers', burgers), ('poisson', poisson)]:
        assert len({(r['job'], r['gpu']) for r in rows}) == 1, f'{name}: mixed allocation/hardware'
    primary_p = [r for r in poisson if r['tau'] == 0.001 and r['cohort'] == 'heldout_seed0']
    dest = ROOT / 'reports' / args.stem
    scaling_figure(burgers, primary_p, dest)
    def wins(rows):
        ns = [str(r['N']) for r in rows if r['ratio'] > 1]
        return ', '.join(ns) if ns else 'none of the tested meshes'

    tokens = dict(
        MAIN_FINDING=f'Under this protocol, the Burgers tensor beats its named FOM at nodes per axis {wins(burgers)}. '
            f'Poisson QF beats the original CG comparator at nodes per axis {wins(primary_p)}. '
            + ('The direct Poisson solver remains faster at every tested mesh.'
               if all(r['direct_ms'] < r['rom_ms'] for r in primary_p)
               else 'The direct Poisson comparison is shown separately below.'),
        PRIMARY_TAU=str(primary_p[0]['tau']),
        FIGURE_NAME=args.stem+'-scaling',
        TRUTH_TABLE=table(['Poisson nodes/axis', 'Recorded / NumPy reference residual', 'Historical residual limit', 'Independent CG/direct field discrepancy'],
            [[r['N'], f'{r["truth_residual"]:.3e} / {r["truth_numpy_residual"]:.3e}', f'{r["truth_residual_limit"]:.1e}',
              f'{r["truth_vs_direct"]:.3e}' if r['truth_vs_direct'] is not None else 'See measured direct error in source rows'] for r in primary_p]),
        CONFIG_TABLE=table(['PDE / nodes per axis', 'Latent coordinates', 'Spatial bank size', 'Weak modes', 'Archived training steps / maximum snapshots'],
            [[f'Burgers / {r["N"]}', r['k'], r['r'], r['M'], f'{r["archived_training_steps"]} / {r["archived_max_training_snapshots"]}'] for r in burgers]
            + [[f'Poisson / {r["N"]}', r['k'], r['r'], r['M'], 'Frozen checkpoint; see source configuration'] for r in primary_p]),
        BG_JOB=str(burgers[0]['job']), BG_GPU=burgers[0]['gpu'],
        BG_STEPS=str(burgers[0]['steps']), BG_DT=str(burgers[0]['dt']),
        BG_FINAL_TIME=str(burgers[0]['final_time']),
        P_JOB=str(poisson[0]['job']), P_GPU=poisson[0]['gpu'],
        BURGERS_TABLE=table(['Nodes/axis', 'Latent solve ms', 'ROM / FOM query ms', 'FOM/ROM', 'ROM mean / median case / worst error (%)', 'FOM mean error (%)', 'Newton / linear tolerance'],
            [[r['N'], f'{r["solve_ms"]:.3f}', f'{r["rom_ms"]:.3f} / {r["fom_ms"]:.3f}', f'{r["ratio"]:.3f}',
              f'{100*r["mean_error"]:.3f} / {100*r["median_case_mean_error"]:.3f} / {100*r["worst_error"]:.3f}',
              f'{100*r["fom_mean_error"]:.3f}', f'{r["newton_tol"]:g} / {r["linear_tol"]:g}'] for r in burgers]),
        BURGERS_STOPS=table(['Nodes/axis', 'Tensor step stopping reasons', 'Blowups'],
            [[r['N'], ', '.join(f'{key}: {value}' for key, value in sorted(r['stop_reasons'].items())), r['blowups']] for r in burgers]),
        POISSON_TABLE=table(['Nodes/axis', 'QF ROM ms', 'CG ms', 'CG/ROM', 'Direct ms', 'ROM mean / median / worst error (%)', 'CG tol / mean error (%)'],
            [[str(r['N'])+(' (extension)' if r['extension'] else ''), f'{r["rom_ms"]:.3f}', f'{r["cg_ms"]:.3f}', f'{r["ratio"]:.3f}', f'{r["direct_ms"]:.3f}',
              f'{100*r["mean_error"]:.3f} / {100*r["median_error"]:.3f} / {100*r["worst_error"]:.3f}',
              f'{r["cg_tol"]:g} / {100*r["cg_mean_error"]:.3f}'] for r in primary_p]),
        POISSON_FULL_TABLE=table(['Nodes/axis', 'Cohort', 'Stopping threshold', 'Full / QF query ms', 'Mean / worst QF error (%)', 'Censored (%)'],
            [[r['N'], r['cohort'], r['tau'], f'{r["full_ms"]:.3f} / {r["rom_ms"]:.3f}', f'{100*r["mean_error"]:.3f} / {100*r["worst_error"]:.3f}', f'{100*r["censored_fraction"]:.1f}'] for r in poisson]),
        REPRODUCTION_TABLE=table(['PDE / mesh', 'Archived mean error (%)', 'Replay mean error (%)', 'Absolute mean-error difference'],
            [[f'Burgers / {r["N"]}', f'{100*r["archived_mean_error"]:.8f}', f'{100*r["mean_error"]:.8f}', f'{abs(r["archived_error_difference"]):.3e}'] for r in burgers]
            + [[f'Poisson QF / {r["N"]}', f'{100*r["archived_mean_error"]:.8f}', f'{100*r["mean_error"]:.8f}', f'{abs(r["archived_error_difference"]):.3e}'] for r in primary_p if 'archived_mean_error' in r]),
        OUTLIERS=table(['PDE / mesh / cohort / threshold', 'ROM / FOM outliers', 'Repetitions per method'],
            [[f'Burgers / {r["N"]}', f'{r["rom_outliers"]} / {r["fom_outliers"]}', r['repetitions_per_method']] for r in burgers]
            + [[f'Poisson / {r["N"]} / {r["cohort"]} / {r["tau"]}', ' / '.join(map(str,r['rom_cg_direct_outliers'])), r['repetitions_per_method']] for r in poisson]),
        CHECK_COUNT=str(len(checks)), FIELD_COUNT=str(len(field_checks)),
        BG_CASES=str(burgers[0]['cases']), P_CASES=str(primary_p[0]['cases']),
        SOURCE_LINKS='\n'.join(f'- [{p}](../{p})' for p in sources if p.endswith('.json') and 'historical-' in p),
    )
    doc = TEMPLATE
    for key, value in tokens.items():
        doc = doc.replace('@@'+key+'@@', value)
    assert '@@' not in doc
    dest.with_suffix('.json').write_text(json.dumps(dict(status='Provisional historical-protocol development replay; final cohorts unopened.',
        sources=sources, checks=checks, field_checks=field_checks, burgers=burgers, poisson=poisson,
        diagnostic_initial_poisson_attempt=initial_attempt), indent=2)+'\n')
    dest.with_suffix('.md').write_text(doc)
    print(f'Wrote {dest.name}: {len(checks)} checks, {len(field_checks)} full-field groups.')


TEMPLATE = r'''# Replaying the earlier Burgers tensor and Poisson comparison

This report contains new GPU measurements under the user's requested historical device-resident protocol. Results are provisional development evidence from the historical cohorts; they are not independent final paper confirmation or replacements for the separately scoped modern benchmark.

@@MAIN_FINDING@@

The input field starts on the GPU. The timer includes initialization or source projection, solution and full-field reconstruction on the GPU. Poisson returns all interior unknowns with prescribed zero boundary values; Burgers returns full grid snapshots. Host transfers, offline training/setup and compilation are outside this primary timer. Both solvers use the same discrete mesh; no coarse-grid FOM envelope is substituted. The [replay protocol](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/HISTORICAL-COMPARISON-REPLAY.md) records this user-selected change in comparison scope.

The restored families use single Gaussian peaks with varying position, width and amplitude. Burgers evolves a scalar viscous field; Poisson solves a steady constant-coefficient equation. Both use unit-square grids with zero Dirichlet boundary values. Inputs to the ROM are supplied fields, with viscosity also supplied for Burgers; the Gaussian generating parameters are not substituted for field-based initialization.

![Device-query runtime and relative-error scaling](<@@FIGURE_NAME@@.png>)

[Download the scaling figure as PDF](<@@FIGURE_NAME@@.pdf>). Each PDE has its own GPU allocation, named below; compare scaling within each panel. The lower panels retain the distinct error summaries defined in their tables. Lines connect measured meshes and do not imply an asymptotic fit or error bound.

## Burgers tensor and original same-grid FOM

Every mesh runs in allocation @@BG_JOB@@ on @@BG_GPU@@, using its frozen historical checkpoint and the restored trained sampled-field initializer. The original tensor, sampled and full residual controls are retained in the raw results. The table uses the tensor's paired complete-query measurements against the original dense-sine-preconditioned, tolerance-stopped Newton FOM, on @@BG_CASES@@ historical test trajectories.

The original rollout uses @@BG_STEPS@@ steps of size @@BG_DT@@ to final time @@BG_FINAL_TIME@@ and reconstructs every step, including the initial field.

@@BURGERS_TABLE@@

Times are cohort medians of per-trajectory repetition medians, and FOM/ROM divides those medians. The latent-solve column is the original pooled median of evolution-only repetitions; it excludes initialization and decoding and must not be substituted for complete-query cost. Errors use the current-time same-grid reference norm. Mean averages over trajectories and times, median case is the median of time-averaged trajectory errors, and worst is the maximum over every trajectory and time. The FOM tolerance is selected using cohort mean accuracy, not a worst-case physical-error guarantee.

The polynomial tensor matches the backward-upwind operator algebraically. Its equivalence to sign-dependent upwinding depends on decoded signs; the retained native gates and undershoot measurements remain part of its scope. Loading the archived trained checkpoint replaces the original in-job training at the intermediate mesh. The initializer is still fitted offline from the recorded seed.

@@BURGERS_STOPS@@

The historical stall stopping rule is retained. A stall exit means the configured progress criterion stopped the iteration; it does not certify a stationary latent minimizer. The reported accuracy belongs to the fields returned at those stops.

## Poisson quadrature-free solve and original CG baseline

Every mesh runs in allocation @@P_JOB@@ on @@P_GPU@@. This table uses the preselected stopping threshold `tau=@@PRIMARY_TAU@@` and @@P_CASES@@ held-out sources from the historical seed. The unpreconditioned CG row is the fastest recorded same-grid tolerance with mean error no larger than the ROM's. The original direct spectral solver is separately named in the same table.

@@POISSON_TABLE@@

The largest quadrature-free mesh is an extension: the historical QF ladder stopped earlier, while the old largest-mesh CG comparison used the sampled cached decoder. A ROM win against CG is a claim about that iterative baseline; the direct column records the available alternative on this constant-coefficient rectangular problem. No cross-job old/new timing ratio is used to claim an algorithm improvement.

The full-grid and QF paths use the same frozen decoder, initialization, trust-LM solver and weak equation. The QF path evaluates a preassembled reduced matrix; the full path evaluates its mathematically equivalent full-grid contraction. Empirical quadrature fitting is omitted from this replay. The additional threshold and historical fresh-seed cohort are retained below rather than selecting a headline configuration after seeing timings.

@@POISSON_FULL_TABLE@@

Censored means the solver stopped without satisfying its accepted convergence reason. These runs still return fields with measured errors; censoring is not a convergence certificate and is retained explicitly.

## Measurement audit and limits

The initial Poisson attempt completed the smaller meshes but stopped before timing the largest mesh: it inherited the smaller-grid reference-residual guard. The archived largest-grid experiment explicitly used a different guard. That recorded setting was restored, an independent CG/direct field-agreement check was added, and the entire ladder was rerun in one allocation. Only that complete rerun enters these tables; the initial attempt remains in the owner's diagnostic archive. The reference CG solver tolerance itself was unchanged.

@@TRUTH_TABLE@@

The generator checked @@CHECK_COUNT@@ timing/error identities and @@FIELD_COUNT@@ groups of full captured fields. Poisson errors were independently recomputed from the last timed output of every method and source, and a separate NumPy stencil calculation checked the reference equation against every saved supplied source. Burgers full-grid reconstruction checks cover the predeclared first trajectory at every mesh; remaining trajectories retain native full-grid errors and sampled output fields, so an independent full-grid audit of the entire Burgers cohort is not claimed.

The following accuracy comparison checks reproduction of the archived configuration. Absolute differences use the dimensionless relative-error fraction; the evidence JSON additionally retains every Burgers time/case error's maximum departure from its archive. No archived largest-mesh Poisson QF result exists to compare with its extension.

@@REPRODUCTION_TABLE@@

Every outlier below exceeds twice its own case's repetition median. All repetitions remain in the aggregates. Poisson's three counts are ROM / selected CG / direct; Burgers has ROM / paired FOM.

@@OUTLIERS@@

Mesh-specific historical checkpoints are restored; this is not frozen-weight mesh transfer. GPU backend, double precision, source/checkpoint hashes, stopping records and raw timings are retained with the owner artifacts. Different historical hardware and instrumentation prevent interpreting an absolute runtime difference from the archive as an isolated numerical-method change. The [historical audit](2026-09-10-historical-poisson-and-burgers-cost-audit.md) documents those prior comparisons and the modern protocol separately.

@@CONFIG_TABLE@@

The checkpoint training settings change across the ladder. An accuracy change between meshes therefore cannot be attributed to resolution alone. The replay performs no new spatial-bank or nonlinear-head training.

@@SOURCE_LINKS@@

## Glossary

- **N / nodes per axis:** grid points including boundary nodes; the interior has fewer points along each axis.
- **Gaussian peak / Dirichlet boundary:** a localized bell-shaped field / a prescribed field value at the boundary, zero here.
- **ROM:** reduced-order model, solving for latent coordinates and reconstructing the field.
- **FOM:** full-order model, solving the discretized field equations.
- **Tensor:** preassembled quadratic coefficients used for Burgers advection.
- **Latent solve:** reduced time evolution after initialization, before field decoding.
- **Query / ms:** the declared input-to-output computation and its elapsed milliseconds.
- **FOM/ROM or CG/ROM:** baseline time divided by ROM time; values above one favor the ROM against that named baseline.
- **QF:** quadrature-free; the repeated weak equation uses an exact preassembled reduced matrix.
- **CG:** conjugate gradient, the historical iterative Poisson solver.
- **Unpreconditioned:** CG uses the original equation directly, without an auxiliary solver transforming its conditioning.
- **Direct:** the original dense sine-transform Poisson solve.
- **Newton / linear tolerance:** stopping settings for Burgers nonlinear corrections and their linear subproblems.
- **Weak residual:** the PDE mismatch projected onto smooth test functions.
- **Trust-LM:** the historical trust-limited Levenberg–Marquardt nonlinear fitting algorithm.
- **Threshold / tau:** the configured stopping threshold, not a bound on solution error.
- **Mean / median / worst error:** specified summaries of relative field errors over the recorded cases and, for Burgers, times.
- **Cohort / held-out / fresh seed:** the named set of sources; held-out sources were excluded from fitting, while the historical fresh cohort uses its separately recorded seed. Neither is newly sealed final evidence in this replay.
- **Censored:** a solve terminating outside the accepted convergence reasons.
- **Stalled / blowup:** a solve stopped by its configured lack-of-progress rule / a trajectory returning nonfinite fields before completion.
- **Checkpoint / initializer:** saved trained weights / the procedure producing the initial latent guess.
- **Latent coordinates / spatial bank size / weak modes:** the number of solved reduced variables / learned spatial features / projected test equations.
- **Training steps / maximum snapshots:** archived optimizer updates / the cap on training fields, not work charged to a query in this replay.
- **Outlier / repetitions:** a slow recorded invocation under the stated threshold / repeated timings of the same query.
- **Absolute mean-error difference:** the magnitude of replay mean error minus archived mean error, expressed as a fraction rather than a percentage.
- **Reference residual / limit:** the discretized equation mismatch for the reference field / its recorded acceptance threshold.
- **CG/direct field discrepancy:** the relative difference between independently computed iterative and direct reference solutions.
- **Allocation:** a single scheduled GPU job, keeping the mesh ladder on one device.
'''

if __name__ == '__main__':
    main()
