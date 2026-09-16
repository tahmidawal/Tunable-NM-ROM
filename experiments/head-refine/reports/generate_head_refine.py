"""Generate the head-refinement report and its error-versus-cost figure.

Every number and every plotted point is read from the raw result and audit JSON.
Nothing in the prose is typed by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHAPE = {'baseline': 'D', 'v1': 'o', 'v2': 's'}
COLOR = {'baseline': '#7a5ba6', 'v1': '#3b6ea5', 'v2': '#4f8a3d'}


def fmt(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def rom_rows(rows, key):
    return [x for x in rows if x['kind'] == 'rom' and x.get(key) is not None]


def ladder(rows, variant, mu_key, cost, err, quad=None):
    """One (variant, anchor) ladder in increasing n, with n = 0 shared as its foot."""
    out = [x for x in rows if x['kind'] == 'rom' and x['n'] == 0
           and (quad is None or x.get('quadrature') == quad)]
    out += [x for x in rows if x['kind'] == 'rom' and x['n'] > 0
            and x['variant'] == variant and x['mu_key'] == mu_key
            and (quad is None or x.get('quadrature') == quad)]
    return sorted([x for x in out if x[cost] is not None and x[err] is not None],
                  key=lambda x: x['n'])


def nondominated(points, cost, err):
    keep = []
    for x in points:
        if not any(y is not x and y[cost] <= x[cost] and y[err] <= x[err]
                   and (y[cost] < x[cost] or y[err] < x[err]) for y in points):
            keep.append(x)
    return sorted(keep, key=lambda x: x[cost])


def verdict(rows, variant, mu_key, cost, err, quad=None):
    lad = ladder(rows, variant, mu_key, cost, err, quad)
    e = [x[err] for x in lad]
    c = [x[cost] for x in lad]
    mono = all(b <= a * (1 + 1e-12) for a, b in zip(e, e[1:]))
    nd = nondominated(lad, cost, err)
    span_cost = (max(x[cost] for x in nd) / min(x[cost] for x in nd)) if nd else 0.
    span_err = (max(x[err] for x in nd) / max(min(x[err] for x in nd), 1e-300)) if nd else 0.
    conv = all(x['all_completed'] for x in nd) if nd else False
    breaks = [dict(n=lad[i + 1]['n'], previous=e[i], here=e[i + 1])
              for i in range(len(e) - 1) if e[i + 1] > e[i] * (1 + 1e-12)]
    return dict(variant=variant, mu_key=mu_key, n=[x['n'] for x in lad], errors=e, costs=c,
                monotone=bool(mono), monotonicity_breaks=breaks,
                nondominated=[x['arm'] for x in nd],
                nondominated_count=len(nd), cost_span=float(span_cost), error_span=float(span_err),
                all_nondominated_converged=bool(conv),
                early_stopped=[x['arm'] for x in lad if not x['all_completed']],
                accepted=bool(mono and len(nd) >= 3 and span_cost >= 2. and span_err >= 2. and conv))


def panel(ax, rows, cost, err, title, quad=None):
    pts = [x for x in rows if x['kind'] == 'rom' and x[err] is not None
           and (quad is None or x.get('quadrature') == quad)]
    for x in pts:
        v = x['variant']
        filled = bool(x['all_completed'])
        ax.plot(x[cost], x[err], SHAPE.get(v, 'o'), markersize=8,
                markerfacecolor=(COLOR.get(v, '#333') if filled else 'none'),
                markeredgecolor=COLOR.get(v, '#333'), markeredgewidth=1.4, zorder=3)
        ax.annotate(f"{x['n']}", (x[cost], x[err]), textcoords='offset points',
                    xytext=(6, 4), fontsize=7, color=COLOR.get(v, '#333'))
    for variant in ('v1', 'v2'):
        for mu_key, ls in (('loose', '-'), ('tight', '--')):
            lad = ladder(rows, variant, mu_key, cost, err, quad)
            if len(lad) > 1:
                ax.plot([x[cost] for x in lad], [x[err] for x in lad], ls, lw=1.1,
                        color=COLOR[variant], alpha=.6, zorder=2,
                        label=f'{variant.upper()}, {mu_key} anchor')
    nd = nondominated(pts, cost, err)
    if nd:
        ax.plot([x[cost] for x in nd], [x[err] for x in nd], ':', lw=2.0, color='#b03a3a',
                zorder=1, label='non-dominated frontier')
    for x in rows:
        if x['kind'] == 'fom':
            if x[err] is not None and x[err] > 0:
                ax.plot(x[cost], x[err], 'X', markersize=11, color='#b03a3a', zorder=4,
                        label=f"full-order `{x['arm']}`")
            else:
                ax.axvline(x[cost], color='#b03a3a', ls='-.', lw=1.2, alpha=.7, zorder=0,
                           label=f"full-order `{x['arm']}` cost (same-grid error 0 by definition)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=10)
    ax.grid(True, which='both', alpha=.25)
    ax.legend(fontsize=6.5, loc='best')


def figure(brows, prows, bL, pL, out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    panel(axes[0], brows, 'median_gpu_ms', 'worst_same_grid_percent',
          f'Burgers 2D, {bL} intervals (marker: D n=0, o V1, s V2; open = early-stopped)',
          quad='eq')
    axes[0].set_xlabel('median complete-query GPU time (ms)')
    axes[0].set_ylabel('worst same-grid error vs converged FOM (%)')
    panel(axes[1], prows, 'median_query_ms', 'worst_same_grid_percent',
          f'Poisson 2D, {pL} intervals (V1 only; V2 collapses onto it)')
    axes[1].set_xlabel('median complete-query time (ms)')
    axes[1].set_ylabel('worst same-grid error vs direct solve (%)')
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    return dict(png=str(out_png), pdf=str(out_pdf),
                png_sha256=hashlib.sha256(Path(out_png).read_bytes()).hexdigest())


def three_layer(w, rows, cost, err, label, quad=None):
    w(f'| arm | variant | $n$ | $\\mu$ | bank floor % | best-found (refined) % | worst {label} % | '
      f'median {label} % | worst reference % | drift | median {cost.replace("_", " ")} | '
      f'latent solves | budget exits | converged |')
    w('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for x in rows:
        w('| ' + ' | '.join([
            f"`{x['arm']}`" if x['kind'] == 'rom' else f"FOM `{x['arm']}`",
            x['variant'] or '—',
            '—' if x['n'] is None else str(x['n']),
            '—' if x.get('mu') in (None,) else f"{x['mu']:g}",
            fmt(x['worst_bank_projection_percent'], 4),
            fmt(x['worst_best_found_percent'], 4),
            fmt(x['worst_same_grid_percent'], 4),
            fmt(x['median_same_grid_percent'], 4),
            fmt(x.get('worst_reference_percent', x.get('worst_error_percent')), 4),
            ('—' if x.get('worst_drift') is None else f"{x['worst_drift']:.3e}"),
            fmt(x[cost], 3),
            ('—' if x.get('latent_solves') is None else str(x['latent_solves'])),
            ('—' if x.get('total_budget_exits') is None else str(x['total_budget_exits'])),
            yn(x['all_completed'])]) + ' |')


def build(b, ba, p, pa, sm, figname):
    brows, prows = ba['checks']['arm_table'], pa['checks']['arm_table']
    out = []
    w = out.append
    w('# Per-query head refinement as an inference-time knob')
    w('')
    w('Is refining the frozen head\'s own weights at query time, anchored to the trained weights, '
      'a usable accuracy/cost knob? Two pre-registered variants on two PDEs, at one frozen '
      'checkpoint each, against pre-registered acceptance criteria. **Final for the opened '
      'development cohorts and provisional as a paper claim**: one checkpoint per PDE, one mesh, '
      'one training seed, final cohorts sealed. The predeclared protocol is '
      '[DESIGN.md](../DESIGN.md).')
    w('')
    w(f"Burgers job `{b.get('job_id')}` on `{b.get('gpu')}` (source `{b.get('commit')}`, elapsed "
      f"{b['elapsed_seconds']:.1f} s, {b['intervals']} intervals) and Poisson job "
      f"`{p.get('job_id')}` on `{p.get('gpu')}` (source `{p.get('commit')}`, elapsed "
      f"{p['elapsed_seconds']:.1f} s, {p['intervals']} intervals). JAX {b['jax_version']}, backend "
      f"`{b['backend']}`, float64, matmul precision `{b['matmul_precision']}` in both.")
    w('')
    w('## The knob')
    w('')
    w('Everything except the refinement is the head-ablation arm (a) contract. The reduced state '
      'is $u(z) = G\\,h_\\theta(z)$ with the bank $G$ frozen; $\\theta$ is the head\'s own weights, '
      'flattened into one vector, and the anchor uses the plain Euclidean metric on that vector, '
      'made dimensionless by $\\|\\theta_0\\|_2$:')
    w('')
    w(r'$$\Omega(\theta) = \frac{\|\theta-\theta_0\|_2^2}{\|\theta_0\|_2^2},'
      r'\qquad \sqrt{\Omega}\ \text{is the relative drift reported below.}$$')
    w('')
    w(f"On Burgers $\\theta$ has {b['theta_count']:,} entries and $\\|\\theta_0\\|_2 = "
      f"{b['theta_norm']:.4f}$; on Poisson {p['theta_count']:,} entries and "
      f"$\\|\\theta_0\\|_2 = {p['theta_norm']:.4f}$.")
    w('')
    w('**V1, initial-only.** Refine once against the supplied initial field, on the same sampled '
      'node set the initializer uses, then evolve with the refined head and the unchanged solver:')
    w('')
    w(r'$$F_1(z,\theta) = \frac{\|\mathrm R\,h_\theta(z)-y\|_2^2}{\|u_0^{(w)}\|_2^2}'
      r' + \mu\,\Omega(\theta),$$')
    w('')
    w('with $\\mathrm R$ the thin-QR factor of the weighted bank at the Gauss nodes and '
      '$y = Q^\\top u_0^{(w)}$. This attacks the initial-compression loss the head-ablation job '
      'recorded as comparable to the whole trajectory error.')
    w('')
    w('**V2, per-step.** After each time step\'s Levenberg-Marquardt solve converges on $z$, take '
      '$n$ Adam steps on $\\theta$ against that step\'s own weak residual, anchored, then re-solve '
      '$z$ under the identical rule:')
    w('')
    w(r'$$F_2(\theta) = \frac{\|r_w(z;\theta,p)\|_2^2}{\|p\|_2^2} + \mu\,\Omega(\theta).$$')
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  U0["supplied u0"] --> IC["cold start + LM on z"]')
    w('  IC --> R1{"n > 0 and V1?"}')
    w('  R1 -- yes --> A1["n x (Adam on theta, re-solve z)"] --> EV')
    w('  R1 -- no --> EV["evolve: 50 backward-Euler steps"]')
    w('  EV --> S1["LM on z at this step"]')
    w('  S1 --> R2{"n > 0 and V2?"}')
    w('  R2 -- yes --> A2["n x Adam on theta"] --> S2["re-solve z"] --> NX')
    w('  R2 -- no --> NX["next step"]')
    w('  NX --> OUT["six dense output fields"]')
    w('  classDef frozen fill:#eee,stroke:#888;')
    w('  classDef trained fill:#dbe8f5,stroke:#3b6ea5;')
    w('  classDef solved fill:#dff0d8,stroke:#4f8a3d;')
    w('  class A1,A2 trained; class IC,S1,S2 solved; class U0,OUT frozen;')
    w('```')
    w('')
    w('On Poisson the query is a **single static solve** with residual exactly $Bh(z)-f_m$, so V2 '
      'has no per-step structure to exploit and collapses onto V1. Only V1 is defined and run '
      'there; the Poisson refinement objective is that same weak residual against the '
      'source-projected data, which is simultaneously the initial fit and the solve.')
    w('')
    w('## Fidelity gates')
    w('')
    w(f"**(i) Local smoke.** Through the new code path with $\\theta$ as a traced runtime operand, "
      f"$n=0$ reproduces the consolidated saved Burgers case to "
      f"{sm['n0_vs_saved_case']['relative_l2']:.3e} relative (latents "
      f"{sm['n0_vs_saved_case']['latent_relative_l2']:.3e}) and the incumbent "
      f"`accuracy_paths.make_rom` to {sm['n0_vs_incumbent']:.3e}, both inside the declared "
      f"$10^{{-12}}$. The reconstructed head is bit-identical to the retained head "
      f"({sm['head_roundtrip_max_abs']:.1e} maximum absolute difference).")
    w('')
    g = b['gate_in_job']
    w(f"**(ii) In-job Burgers.** `n0` reproduces the head-ablation job's `a_neural_eq` on all "
      f"{g['compared']} cases to a worst relative difference of "
      f"{g['worst_relative_delta']:.3e} against a declared $10^{{-9}}$, with "
      f"{g['bitwise_identical_fields']} of {g['compared']} output fields bitwise identical across "
      f"the two jobs (reference job `{g['reference_job']}` on `{g['reference_gpu']}`, this job on "
      f"`{g['this_gpu']}`).")
    w('')
    gp = p['gate_in_job']
    w(f"**(iii) In-job Poisson.** `n0` reproduces the head-ablation Poisson job's `a_neural` at "
      f"{p['intervals']} intervals on all {gp['compared']} sources to "
      f"{gp['worst_relative_delta']:.3e}, against the declared cross-job tolerance "
      f"{gp['tolerance']:g}, with {gp['bitwise_identical_fields']} of {gp['compared']} fields "
      f"bitwise identical.")
    w('')
    bd, pd = ba['checks']['redecode'], pa['checks']['redecode']
    w(f"**(iv) Independent audit.** A NumPy-only audit that imports neither driver nor JAX "
      f"recomputed every reported error from the retained output fields: worst relative "
      f"difference {ba['checks']['recorded_errors_recomputed_from_saved_fields']['detail']:.2e} on "
      f"Burgers and "
      f"{pa['checks']['recorded_errors_recomputed_from_saved_fields']['detail']:.2e} on Poisson.")
    w('')
    w(f"**(v) Re-decode.** The refined weights are saved per arm and case. The same audit rebuilt "
      f"the head in pure NumPy from those weights and recomputed $G\\,h_\\theta(z)$ at a fixed "
      f"{bd['nodes']}-node sample of the frozen bank: {bd['compared']} Burgers "
      f"(arm, case) pairs agree with the saved output field to {bd['worst_relative']:.2e} and "
      f"{pd['compared']} Poisson pairs to {pd['worst_relative']:.2e}, against a declared "
      f"{bd['tolerance']:g}.")
    w('')
    w('## The step size and the anchor')
    w('')
    for tag, r in (('Burgers', b), ('Poisson', p)):
        c = r['calibration']
        w(f"**{tag}.** {c['rule']}. On the single training-family calibration case the sweep gave "
          + ', '.join(f"$\\alpha={t['alpha']:g}$ &rarr; {t['final_data_term']:.6e}"
                      for t in c['trace'])
          + f", so $\\alpha = {c['selected_alpha']:g}$ was selected and frozen.")
        grid = r['config']['alpha_grid']
        if c['selected_alpha'] in (min(grid), max(grid)):
            w('')
            w(f"**Recorded limitation.** The selected step size sits at the "
              f"{'upper' if c['selected_alpha'] == max(grid) else 'lower'} edge of the "
              f"pre-registered grid $[{min(grid):g}, {max(grid):g}]$, so the calibration did not "
              f"bracket an interior optimum: a "
              f"{'larger' if c['selected_alpha'] == max(grid) else 'smaller'} step was never "
              f"tested. The grid was fixed before the run and was not widened afterwards, so every "
              f"{tag} number below is for that step size and may understate what refinement could "
              f"do with a better-chosen one.")
        w('')
        d = r['anchor_diagnostic']
        w(f"Anchor diagnostic on the same calibration case at $n={d['n']}$ "
          f"(diagnostic only; it selects nothing): "
          + ', '.join(f"$\\mu={x['mu']:g}$ drift {x['drift']:.3e}, anchor/data gradient "
                      f"{x['anchor_to_data_gradient']:.2e}" for x in d['rows']) + '.')
        w('')
    w('## Burgers 2D — the three layers')
    w('')
    tight = next((x for x in brows if x['arm'] == 'fft_tight'), None)
    w(f"The primary metric is the same-grid discrepancy against the converged full-order solve on "
      f"this mesh, because the refined-reference metric also contains this mesh's discretisation "
      f"error: that full-order solve itself carries {tight['worst_reference_percent']:.4f}% worst "
      f"against the refined reference. Both are reported. The bank projection floor is the best "
      f"any coefficients at all could do in the frozen bank and is unchanged by refinement.")
    w('')
    w('**How the refined best-found column is computed, and one structural caveat.** For each arm '
      'and case it is the worst over the six output times of a seeded multistart fit on that arm\'s '
      'own moved manifold, using the weights that arm actually produced. For V1 those weights are '
      'one refined $\\theta$ used at every output time. For V2 the weights in force at $t=0$ are '
      '$\\theta_0$ by construction — no time step has happened yet — so a V2 arm\'s worst-over-times '
      'best-found can never fall below the $n=0$ value even when its later times improve. Read the '
      'V2 best-found column as an upper bound pinned at $t=0$, not as evidence that per-step '
      'refinement does not move the manifold; the drift column and the per-output-time table show '
      'that it does.')
    w('')
    three_layer(w, brows, 'median_gpu_ms', 'worst_same_grid_percent', 'same-grid')
    w('')
    w('Full-order rows are **context only**; no speed claim is made against them here. '
      '`fft_tight` defines the same-grid metric, so its own same-grid value is zero by '
      'construction and it enters the figure as a cost line rather than a point.')
    w('')
    w('## Poisson 2D — the three layers')
    w('')
    three_layer(w, prows, 'median_query_ms', 'worst_same_grid_percent', 'same-grid')
    w('')
    w(f'![worst same-grid error versus median query cost]({figname})')
    w('')
    w('## Is $n$ a knob? The pre-registered acceptance test')
    w('')
    w('All three must hold, per variant: the worst same-grid error non-increasing along '
      '$n = 0,1,2,4,8,16,32$; at least three non-dominated points spanning $\\ge 2\\times$ in cost '
      'and $\\ge 2\\times$ in error; and none of those points early-stopped.')
    w('')
    w('| PDE | variant | anchor | monotone | non-dominated | cost span | error span | none '
      'early-stopped | **verdict** |')
    w('|---|---|---|---|---:|---:|---:|---|---|')
    verdicts = []
    for pde, rows, cost, quad, variants in (
            ('Burgers', brows, 'median_gpu_ms', 'eq', ('v1', 'v2')),
            ('Poisson', prows, 'median_query_ms', None, ('v1',))):
        for variant in variants:
            for mu_key in ('loose', 'tight'):
                v = verdict(rows, variant, mu_key, cost, 'worst_same_grid_percent', quad)
                v['pde'] = pde
                verdicts.append(v)
                w(f"| {pde} | {variant.upper()} | {mu_key} | {yn(v['monotone'])} | "
                  f"{v['nondominated_count']} | {v['cost_span']:.2f}x | {v['error_span']:.2f}x | "
                  f"{yn(v['all_nondominated_converged'])} | "
                  f"**{'A KNOB' if v['accepted'] else 'NOT a knob'}** |")
    w('')
    fails = {}
    for v in verdicts:
        why = []
        if not v['monotone']:
            why.append('monotonicity')
        if v['nondominated_count'] < 3:
            why.append('fewer than three non-dominated points')
        if v['cost_span'] < 2.:
            why.append('cost span below 2x')
        if v['error_span'] < 2.:
            why.append('error span below 2x')
        if not v['all_nondominated_converged']:
            why.append('an early-stopped point on the frontier')
        fails[(v['pde'], v['variant'], v['mu_key'])] = why
    passed = [v for v in verdicts if v['accepted']]
    w('**Reading.** '
      + ('Every ladder fails at least one criterion, so refinement is **not** a usable knob as '
         'pre-registered, on either PDE and under either anchor. ' if not passed
         else f"{len(passed)} of {len(verdicts)} ladders meet all three criteria. ")
      + 'What each ladder failed on: '
      + '; '.join(f"{v['pde']} {v['variant'].upper()}/{v['mu_key']} — "
                  + (', '.join(fails[(v['pde'], v['variant'], v['mu_key'])])
                     if fails[(v['pde'], v['variant'], v['mu_key'])] else 'nothing')
                  for v in verdicts) + '.')
    w('')
    for v in verdicts:
        w(f"- {v['pde']} {v['variant'].upper()} / {v['mu_key']}: worst same-grid along "
          f"$n={v['n']}$ is {[round(x, 4) for x in v['errors']]} percent at "
          f"{[round(x, 3) for x in v['costs']]} ms."
          + (f" Early-stopped arms on this ladder: {', '.join('`' + a + '`' for a in v['early_stopped'])}."
             if v['early_stopped'] else ' No arm on this ladder is early-stopped.')
          + (' Monotonicity breaks at ' + ', '.join(
              f"$n={x['n']}$ ({x['previous']:.4f} &rarr; {x['here']:.4f} %)"
              for x in v['monotonicity_breaks']) + '.' if v['monotonicity_breaks'] else ''))
    w('')
    w('### The non-dominated set')
    w('')
    for pde, rows, cost, quad in (('Burgers', brows, 'median_gpu_ms', 'eq'),
                                  ('Poisson', prows, 'median_query_ms', None)):
        pts = [x for x in rows if x['kind'] == 'rom'
               and (quad is None or x.get('quadrature') == quad)]
        nd = nondominated(pts, cost, 'worst_same_grid_percent')
        w(f'**{pde}**, over every reduced arm in the primary quadrature:')
        w('')
        w('| arm | variant | $n$ | $\\mu$ | worst same-grid % | median cost (ms) | drift | converged |')
        w('|---|---|---:|---:|---:|---:|---:|---|')
        for x in nd:
            w(f"| `{x['arm']}` | {x['variant']} | {x['n']} | "
              f"{'—' if x.get('mu') is None else format(x['mu'], 'g')} | "
              f"{x['worst_same_grid_percent']:.4f} | {x[cost]:.3f} | "
              f"{'—' if x.get('worst_drift') is None else format(x['worst_drift'], '.3e')} | "
              f"{yn(x['all_completed'])} |")
        w('')
    w('## The held-out generalisation risk: error per output time')
    w('')
    w('Refining on the initial field can overfit it and worsen later times. Worst same-grid error '
      'over the six development cases, by output time, on Burgers:')
    w('')
    times = b['output_times']
    w('| arm | ' + ' | '.join(f'$t={t:g}$' for t in times) + ' |')
    w('|---|' + '---:|' * len(times))
    for x in brows:
        if x['kind'] == 'rom' and x['worst_same_grid_per_time_percent']:
            w(f"| `{x['arm']}` | "
              + ' | '.join(f'{v:.4f}' for v in x['worst_same_grid_per_time_percent']) + ' |')
    w('')
    bt = base_per_time = next((y['worst_same_grid_per_time_percent'] for y in brows
                               if y['arm'] == 'n0'), None)
    if bt:
        at0 = int(np.argmax(bt))
        w(f"**Where the worst-over-times error lives.** For the unrefined head the maximum over the "
          f"six output times is attained at $t={times[at0]:g}$ "
          f"({bt[at0]:.4f}% against {max(bt[1:]):.4f}% over the evolved times). "
          + ("Because the $t=0$ output is the model's own compression of the supplied initial "
             "field, and V2 has not taken a single refinement step by the time that field is "
             "decoded, **no V2 arm can move the worst-over-times number at all** once its evolved "
             "times fall below the $t=0$ value: several V2 rows sit at exactly the $n=0$ value for "
             "that reason, not because refinement did nothing. V1 does move it, because it refines "
             "before $t=0$ is decoded. The per-time table above is the honest reading for V2."
             if at0 == 0 else
             "The maximum is not at $t=0$, so the per-step variant is not structurally pinned "
             "here."))
        w('')
    risky = []
    for x in brows:
        v = x.get('worst_same_grid_per_time_percent')
        base = next((y['worst_same_grid_per_time_percent'] for y in brows
                     if y['arm'] == 'n0'), None)
        if v and base and x['n'] and v[1] < base[1] * (1 - 1e-12) and v[-1] > base[-1] * (1 + 1e-12):
            risky.append(x['arm'])
    w('Arms that improve the first evolved output time and worsen the last, relative to `n0` — the '
      'signature of overfitting the supplied initial field: '
      + (', '.join(f'`{a}`' for a in risky) + '.' if risky else '**none**.'))
    w('')
    w('## Plain-language glossary')
    w('')
    for line in [
        '**Arm** — one configuration under test; everything except the named difference is held '
        'fixed.',
        '**Bank $G$** — the fixed set of spatial fields the reduced state is built from. It is '
        'frozen here: refinement never touches it.',
        '**Head $h_\\theta$** — the small network turning the few solved coordinates into bank '
        'coefficients. Its weights $\\theta$ are the object being refined.',
        '**$n$** — the number of gradient steps taken on $\\theta$ inside the timed query. $n=0$ '
        'is the retained frozen-weight model exactly.',
        '**V1 / V2** — initial-only refinement (once, against the supplied initial field) and '
        'per-step refinement (after every time step, against that step\'s weak residual).',
        '**Anchor weight $\\mu$** — how hard the refined weights are pulled back towards the '
        'trained weights. "Loose" permits drift, "tight" restrains it.',
        '**Drift** — $\\sqrt{\\Omega}$, how far the refined weights moved, relative to the size '
        'of the trained weights. Zero at $n=0$.',
        '**Step size $\\alpha$** — the Adam learning rate, chosen once on one training-family '
        'case and then frozen for every evaluation case.',
        '**Latent solve** — one run of the Levenberg-Marquardt iteration for the coordinates $z$: '
        'the initial fit, each time step, and each refinement re-solve.',
        '**Exit reason** — why a latent solve stopped: 1 residual below tolerance, 2 step too '
        'small to matter, 3 step rejected repeatedly, 4 normalized gradient below tolerance, '
        '0 iteration budget exhausted. Only 1, 2 and 4 count as converged.',
        '**Early-stopped** — a solve that ran out of iteration budget or kept rejecting steps. '
        'Such an arm is a legitimate point on an error/cost curve but is never relabelled '
        'stationary or converged.',
        '**Bank projection floor** — the best any coefficients at all could do in the frozen '
        'bank. No head, refined or not, can beat it.',
        '**Best-found reconstruction (refined)** — the best that arm\'s own moved manifold can do '
        'on the reference field with no PDE involved, using the weights that arm actually '
        'produced. It separates representation from dynamics.',
        '**Same-grid error** — the discrepancy against the converged full-order solve on the '
        'same mesh. It isolates reduction error from this mesh\'s discretisation error, so it is '
        'the discriminator.',
        '**Reference error** — the discrepancy against a much finer refined solve. It contains '
        'this mesh\'s discretisation error as well.',
        '**Empirical quadrature (eq) / dense** — a learned weighted subset of grid points '
        'standing in for the full grid sum, versus the exact full sum.',
        '**Non-dominated** — a point no other point beats on both cost and error at once. The '
        'set of them is the usable frontier.',
        '**Development / final cohort** — cases usable for method selection / cases kept '
        'unopened. No final-cohort case was opened here.',
        '**Calibration case** — the single training-family case used to choose the step size. It '
        'is not an evaluation case.',
        '**Full-order (FOM) rows** — the conventional solver, shown as context only. No speed '
        'claim is made against them in this report.',
    ]:
        w(f'- {line}')
    w('')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--burgers-result', required=True)
    ap.add_argument('--burgers-audit', required=True)
    ap.add_argument('--poisson-result', required=True)
    ap.add_argument('--poisson-audit', required=True)
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    b = json.loads(Path(a.burgers_result).read_text())
    ba = json.loads(Path(a.burgers_audit).read_text())
    p = json.loads(Path(a.poisson_result).read_text())
    pa = json.loads(Path(a.poisson_audit).read_text())
    sm = json.loads(Path(a.smoke).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out.with_suffix(''))
    fig = figure(ba['checks']['arm_table'], pa['checks']['arm_table'],
                 b['intervals'], p['intervals'],
                 Path(stem + '-cost.png'), Path(stem + '-cost.pdf'))
    text = build(b, ba, p, pa, sm, Path(stem + '-cost.png').name)
    out.write_text(text)
    print(out, hashlib.sha256(text.encode()).hexdigest())
    print(json.dumps(fig, indent=2))


if __name__ == '__main__':
    main()
