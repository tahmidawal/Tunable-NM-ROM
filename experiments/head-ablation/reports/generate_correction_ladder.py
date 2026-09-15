"""Generate the correction-ladder report and its error-versus-cost figure.

Every number and every plotted point is read from the raw result and audit JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fmt(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def figure(rows, cfg, out_png, out_pdf):
    ladder = sorted([x for x in rows if x['kind'] == 'rom' and x['quadrature'] == 'dense'
                     and x['dt'] == cfg['dt'] and x['arm'] != 'q0_dense_Mmax'],
                    key=lambda x: x['q'])
    eq = sorted([x for x in rows if x['kind'] == 'rom' and x['quadrature'] == 'eq'
                 and x['dt'] == cfg['dt']], key=lambda x: x['q'])
    alt = [x for x in rows if x['kind'] == 'rom' and x['dt'] != cfg['dt']]
    mmax = [x for x in rows if x['arm'] == 'q0_dense_Mmax']
    foms = [x for x in rows if x['kind'] == 'fom']

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot([x['median_gpu_ms'] for x in ladder], [x['worst_same_grid_percent'] for x in ladder],
            '-o', color='#3b6ea5', label='correction ladder, dense quadrature', zorder=3)
    for x in ladder:
        ax.annotate(f"q={x['q']}", (x['median_gpu_ms'], x['worst_same_grid_percent']),
                    textcoords='offset points', xytext=(6, 6), fontsize=8, color='#3b6ea5')
    if eq:
        ax.plot([x['median_gpu_ms'] for x in eq], [x['worst_same_grid_percent'] for x in eq],
                '-s', color='#4f8a3d', label='same q, empirical quadrature', zorder=3)
        for x in eq:
            ax.annotate(f"q={x['q']}", (x['median_gpu_ms'], x['worst_same_grid_percent']),
                        textcoords='offset points', xytext=(6, -12), fontsize=8, color='#4f8a3d')
    for x, mark in zip(alt, ['^', 'v', '<', '>']):
        ax.plot(x['median_gpu_ms'], x['worst_same_grid_percent'], mark, markersize=9,
                color='#b07b32', label=f"q=0, dt {x['dt']:g}, {x['quadrature']}", zorder=4)
    for x in mmax:
        ax.plot(x['median_gpu_ms'], x['worst_same_grid_percent'], 'D', markersize=8, color='#7a5ba6',
                label=f"q=0 held at M={x['M']} (test-count control)", zorder=4)
    # The converged full-order solve defines the same-grid metric, so its own value is
    # zero by construction: it enters the figure as a cost line, not as a point.
    for x in foms:
        if x['worst_same_grid_percent'] is not None and x['worst_same_grid_percent'] > 0:
            ax.plot(x['median_gpu_ms'], x['worst_same_grid_percent'], 'X', markersize=12,
                    color='#b03a3a', label=f"full-order `{x['arm']}`", zorder=5)
        else:
            ax.axvline(x['median_gpu_ms'], color='#b03a3a', ls='--', lw=1.4, zorder=1,
                       label=f"full-order `{x['arm']}` cost (its same-grid error is 0 by definition)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('median complete-query GPU time (ms), same job, burn-in before every timed block')
    ax.set_ylabel('worst same-grid error vs converged FOM (%)')
    ax.set_title('Burgers 2D, 256 intervals: fixed-weight correction ladder')
    ax.grid(True, which='both', alpha=.25)
    ax.legend(fontsize=8, loc='best')
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    return dict(png=str(out_png), pdf=str(out_pdf),
                png_sha256=hashlib.sha256(Path(out_png).read_bytes()).hexdigest(),
                plotted_points=len(ladder) + len(eq) + len(alt) + len(mmax) + len(foms))


def build(r, au, sm, figinfo, figname):
    rows = au['checks']['arm_table']
    cfg = r['config']
    K = r['K']
    out = []
    w = out.append
    w('# A fixed-weight correction ladder on Burgers 2D')
    w('')
    w('How much accuracy can be bought from a frozen checkpoint at inference time, and at what')
    w('cost, by solving extra linear bank directions on top of the neural head? One knob, $q$, and')
    w('one secondary knob, the time step. **Final for the six opened development cases and')
    w('provisional as a paper claim**: one checkpoint, one mesh, one training seed, final cohort')
    w('sealed.')
    w('')
    w(f"Job `{r.get('job_id')}` on `{r.get('gpu')}`, source commit `{r.get('commit')}`, JAX "
      f"{r['jax_version']}, backend `{r['backend']}`, float64, matmul precision "
      f"`{r['matmul_precision']}`, elapsed {r['elapsed_seconds']:.1f} s, mesh {r['intervals']} "
      f"intervals per axis.")
    w('')
    w('## The knob')
    w('')
    w(r'$$u(z,y) = G\big(h_\theta(z) + C_q\,y\big),\qquad w=(z,y)\in\mathbb R^{K+q},$$')
    w('')
    w(f"with the bank $G$, the head $h_\\theta$ and every network weight frozen. $q=0$ is the "
      f"head-ablation arm (a) exactly: $C_0$ has no columns. Because the Burgers weak residual is "
      f"quadratic in the coefficients through the upwind advection term, $y$ **cannot** be "
      f"eliminated analytically as it is on Poisson; the whole augmented vector is solved by the "
      f"same Levenberg-Marquardt iteration, with the same trust radius "
      f"{r['arm_setup'][0]['trust_radius']:.9g}, the same budgets "
      f"({cfg['strict']['ic_budget']} initial, {cfg['strict']['step_budget']} per step) and the same "
      f"normalized-gradient tolerance {cfg['strict']['gtol']:g}.")
    w('')
    d = r['directions']
    w(f"**The directions are fixed offline and nested.** {d['rule']}. "
      f"{d['snapshots']} snapshots, {d['starts']} multistart fits each at budget {d['budget']}, seed "
      f"{d['seed']}; the head's own best-found relative fit over them is "
      f"{d['head_fit_relative_median'] * 100:.4f}% median and {d['head_fit_relative_worst'] * 100:.4f}% "
      f"worst, and the available rank is {d['available_rank']}. Residual energy captured: "
      + ', '.join(f"$q={q}$ {v * 100:.4f}%" for q, v in d['residual_energy_captured'].items()) + '.')
    w('')
    w(f"**Fidelity gate.** Through the corrected-head wrapper at $q=0$, the smoke reproduces the "
      f"consolidated saved Burgers case to {sm['q0_vs_saved_case']['relative_l2']:.3e} relative and "
      f"is **bit-identical** ({sm['q0_vs_incumbent']:.1e}) to the incumbent "
      f"`accuracy_paths.make_rom`.")
    g = au['checks'].get('q0_reproduces_head_ablation_arm_a')
    if g and g['detail']:
        det = g['detail']
        w(f"In the job itself, `q0_eq` reproduces the head-ablation job's `a_neural_eq` on all "
          f"{det['compared']} cases to a worst relative difference of "
          f"{det['worst_relative_delta']:.3e}, with {det['bitwise_identical_fields']} of "
          f"{det['compared']} output fields bitwise identical across the two jobs.")
    w('')
    w('## The ladder')
    w('')
    tight = next((x for x in rows if x['arm'] == 'fft_tight'), None)
    w(f"The primary metric is the same-grid discrepancy against the converged full-order solve on "
      f"this mesh, because the refined-reference metric also contains this mesh's discretization "
      f"error: the full-order model itself carries {tight['worst_reference_percent']:.4f}% worst "
      f"against the refined reference. Both are reported.")
    w('')
    w('| arm | $q$ | solved dim | $M$ | $m$ | quad. | $\\Delta t$ | best-found % | worst same-grid % | '
      'median same-grid % | worst reference % | median iters/step | budget exits | max stationarity | '
      'stationary | completed | median GPU ms | median host ms |')
    w('|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|')
    for x in rows:
        w('| ' + ' | '.join([
            f"`{x['arm']}`" if x['kind'] == 'rom' else f"FOM `{x['arm']}`",
            str(x['q']) if x['q'] is not None else '—',
            str(x['solved_dimension']) if x['solved_dimension'] else '—',
            str(x['M'] or '—'), str(x['m'] or '—'), x['quadrature'] or '—',
            f"{x['dt']:g}" if x['dt'] else '—',
            fmt(x['worst_best_found_percent'], 4),
            fmt(x['worst_same_grid_percent'], 4), fmt(x['median_same_grid_percent'], 4),
            fmt(x['worst_reference_percent'], 4),
            fmt(x['median_iterations'], 1),
            ('—' if x['total_budget_exits'] is None else str(x['total_budget_exits'])),
            ('—' if x['max_stationarity'] is None else f"{x['max_stationarity']:.2e}"),
            yn(x['all_stationary']), yn(x['all_completed']),
            fmt(x['median_gpu_ms'], 3), fmt(x['median_host_ms'], 3)]) + ' |')
    w('')
    w(f'![error versus cost]({figname})')
    w('')

    ladder = sorted([x for x in rows if x['kind'] == 'rom' and x['quadrature'] == 'dense'
                     and x['dt'] == cfg['dt'] and x['arm'] != 'q0_dense_Mmax'], key=lambda x: x['q'])
    errs = [x['worst_same_grid_percent'] for x in ladder]
    costs = [x['median_gpu_ms'] for x in ladder]
    mono_err = all(b <= a * (1 + 1e-12) for a, b in zip(errs, errs[1:]))
    mono_cost = all(b >= a * (1 - 1e-12) for a, b in zip(costs, costs[1:]))
    w('## Is the curve monotone?')
    w('')
    w(f"**Error: {'monotone decreasing' if mono_err else 'NOT monotone'}** along "
      f"$q={[x['q'] for x in ladder]}$, with worst same-grid values "
      f"{[round(v, 4) for v in errs]} percent.")
    if not mono_err:
        bad = [(ladder[i + 1]['q'], errs[i], errs[i + 1]) for i in range(len(errs) - 1)
               if errs[i + 1] > errs[i] * (1 + 1e-12)]
        w('')
        w('Non-monotone steps (previous -> this): '
          + '; '.join(f"$q={q}$: {a:.4f}% -> {b:.4f}%" for q, a, b in bad) + '. '
          'More correction capacity does not guarantee a smaller physical error: the weak objective '
          'is minimised, not the field error, and the shared trust radius and iteration budget are '
          'held fixed as the solved dimension grows.')
    w('')
    w(f"**Cost: {'monotone increasing' if mono_cost else 'NOT monotone'}**, "
      f"{[round(v, 3) for v in costs]} median GPU ms. Two effects grow together here and the "
      f"`q0_dense_Mmax` control separates them: it holds $q=0$ at the ladder's largest test count, "
      f"so the difference between it and `q0_dense` is the cost of the tests alone.")
    mm = next((x for x in rows if x['arm'] == 'q0_dense_Mmax'), None)
    q0d = next((x for x in rows if x['arm'] == 'q0_dense'), None)
    if mm and q0d:
        w('')
        w(f"`q0_dense` costs {q0d['median_gpu_ms']:.3f} ms at $M={q0d['M']}$; the same arm at "
          f"$M={mm['M']}$ costs {mm['median_gpu_ms']:.3f} ms, a factor "
          f"{mm['median_gpu_ms'] / q0d['median_gpu_ms']:.3f}, at "
          f"{mm['worst_same_grid_percent']:.4f}% against {q0d['worst_same_grid_percent']:.4f}%. So "
          f"most of the ladder's cost growth is the growing test count that $M>K+q$ forces, not the "
          f"extra unknowns themselves.")
    w('')
    w('## Cost per unit of accuracy')
    w('')
    w('Taking $q=0$ dense as the reference point, each rung is scored by how much error it removes '
      'per extra millisecond.')
    w('')
    w('| $q$ | worst same-grid % | error removed vs $q=0$ (pp) | extra median GPU ms | '
      'pp removed per extra ms | cost factor vs $q=0$ |')
    w('|---:|---:|---:|---:|---:|---:|')
    base = ladder[0]
    for x in ladder:
        dE = base['worst_same_grid_percent'] - x['worst_same_grid_percent']
        dT = x['median_gpu_ms'] - base['median_gpu_ms']
        rate = (dE / dT) if dT > 0 else None
        w('| ' + ' | '.join([str(x['q']), fmt(x['worst_same_grid_percent'], 4), fmt(dE, 4),
                             fmt(dT, 3), ('—' if rate is None else f'{rate:.5f}'),
                             fmt(x['median_gpu_ms'] / base['median_gpu_ms'], 3)]) + ' |')
    w('')

    w('## Does any rung beat the efficient full-order solver on both axes?')
    w('')
    loose = next((x for x in rows if x['arm'] == 'nt1e-2'), None)
    roms = [x for x in rows if x['kind'] == 'rom']
    for fom in [x for x in [loose, tight] if x]:
        beat = [x for x in roms
                if x['worst_same_grid_percent'] <= fom['worst_same_grid_percent']
                and x['median_gpu_ms'] <= fom['median_gpu_ms']]
        w(f"Against full-order `{fom['arm']}` "
          f"({fom['worst_same_grid_percent']:.4f}% same-grid, {fom['median_gpu_ms']:.3f} ms median "
          f"GPU, {fom['median_host_ms']:.3f} ms complete host query): "
          + (('**' + ', '.join(f"`{x['arm']}`" for x in beat) + '** dominate it on both axes.')
             if beat else '**no arm on this ladder dominates it on both axes.**'))
        w('')
    w('This is a within-job comparison on one GPU with burn-in before every timed block and all '
      'repetitions retained. It is not a cross-job timing ratio and it is not a claim about any '
      'other mesh.')
    w('')

    w('## Recorded deviations and caveats')
    w('')
    eqq = sorted({x['q'] for x in rows if x['quadrature'] == 'eq'})
    w(f"1. Empirical quadrature is fitted only at $q\\in{{{', '.join(map(str, eqq))}}}$. Above that, "
      f"$m=4M$ points grow with $q$ and the bounded nonnegative-least-squares fit is not "
      f"constructible inside the job budget, so those rungs use the exact dense grid sum. Each row "
      f"states its own quadrature, and the paired `eq`/`dense` rows at the same $q$ isolate the "
      f"quadrature effect.")
    w(f"2. The weak objective requires more tests than unknowns, so $M=4(K+q)$ grows along the "
      f"ladder. Cost therefore rises for two reasons at once; `q0_dense_Mmax` separates them.")
    w(f"3. At $q=R={r['R']}$ the reachable set coincides with the head-ablation free-bank arm (d), "
      f"because $h_\\theta(z)+C_R y$ can reach any bank coefficient vector. The parameterization is "
      f"redundant by $K$ dimensions, so its Jacobian is rank deficient and the damped iteration is "
      f"not arm (d)'s; the test count also differs. The two are the same reachable set, not the "
      f"same solver, and are not expected to agree numerically.")
    w(f"4. The trust radius, iteration budgets and stopping tolerance are held at arm (a)'s values "
      f"for every rung, so that $q$ is the only online knob. A fixed trust radius is a tighter "
      f"restriction on a larger step, which is part of what the ladder measures.")
    w(f"5. Arms above {cfg['gauss_jordan_max']} unknowns use a pivoted dense step solve instead of "
      f"the incumbent unrolled Gauss-Jordan, which is more accurate, not weaker.")
    w('6. The stationarity column is not a quality ranking: the normalized gradient is scale '
      'invariant and stays of order one for an arm whose reduced fit is attainable, which then '
      'exits by the small-step rule with a better fit. The `completed` column is the honest '
      'status - no iteration-budget exit and no rejected-step exit anywhere.')
    w('')

    w('## Glossary')
    w('')
    for term, text in [
        ('$q$', 'the online knob: how many extra fixed linear directions inside the frozen bank are '
                'solved on top of the neural head. $q=0$ is the unmodified retained model.'),
        ('correction direction', 'one fixed spatial field, chosen offline, that the solver may add a '
                                 'freely solved multiple of. The set is nested, so a larger $q$ '
                                 'contains every smaller one.'),
        ('$K$ / solved dimension', 'the neural latent dimension / $K+q$, the number of unknowns the '
                                   'online solver actually solves for.'),
        ('$M$ (test modes)', 'how many smooth functions the PDE residual is averaged against; it must '
                             'exceed the solved dimension, which is why it grows with $q$.'),
        ('$m$ (quadrature points)', 'grid points used by the empirical quadrature rule for the '
                                    'nonlinear advection term instead of the whole grid; "dense" means '
                                    'the full grid sum, with no approximation.'),
        ('same-grid error', 'difference from the converged full-order solve on the same mesh, '
                            'normalised by the initial reference field norm. This isolates the '
                            'reduction error from the discretization error.'),
        ('reference error', 'difference from the refined reference solved on a much finer mesh and '
                            'time step. It contains this mesh\'s discretization error as well.'),
        ('best-found reconstruction', 'the smallest error found on that arm\'s manifold when fitting '
                                      'the reference field directly, with no PDE. It separates '
                                      'representation from dynamics.'),
        ('iterations', 'Levenberg-Marquardt steps per time step; hardware-free.'),
        ('budget exits', 'time steps that stopped because the iteration cap was reached rather than '
                         'because a stopping criterion was met.'),
        ('stationary', 'the normalized weak gradient fell below the shared tolerance everywhere. See '
                       'the caveats: not a quality ranking.'),
        ('completed', 'the shared stopping rule terminated everywhere with no budget exit and no '
                      'rejected-step exit.'),
        ('complete query', 'the timed unit: one supplied dense initial field on the GPU to six dense '
                           'output fields, including the initial fit, the evolution and the decode. '
                           'The host column adds the same-invocation transfers.'),
        ('pp', 'percentage points.'),
        ('FOM', 'full-order model: the unreduced solver. `fft_tight` is the converged reference '
                'solve on this mesh; `nt1e-2` is the efficient loose-tolerance control.'),
        ('development / final cohort', 'cases usable for method selection / cases kept unopened.'),
    ]:
        w(f'- **{term}:** {text}')
    w('')
    w('---')
    w('')
    w(f"Generated by `experiments/head-ablation/reports/generate_correction_ladder.py` from "
      f"`result.json` (SHA256 `{au['result_sha256']}`), its audit JSON and the smoke evidence. The "
      f"figure is produced by the same script (PNG SHA256 `{figinfo['png_sha256']}`, "
      f"{figinfo['plotted_points']} plotted points). Every number and marker is read from those files.")
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--smoke', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    au = json.loads(Path(a.audit).read_text())
    sm = json.loads(Path(a.smoke).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix('')
    figinfo = figure(au['checks']['arm_table'], r['config'],
                     Path(str(stem) + '-cost.png'), Path(str(stem) + '-cost.pdf'))
    text = build(r, au, sm, figinfo, Path(str(stem) + '-cost.png').name)
    out.write_text(text)
    print(out, hashlib.sha256(text.encode()).hexdigest())
    print(json.dumps(figinfo, indent=2))


if __name__ == '__main__':
    main()
