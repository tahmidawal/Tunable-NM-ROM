"""Generate the prior-dial report and its error-versus-cost figure.

Every number and every plotted point is read from the raw result JSONs, the
independent audit JSONs and the retained smoke evidence. Nothing is typed by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLOCK_COLOR = {(64, 'eq'): '#3b6ea5', (64, 'dense'): '#4f8a3d',
               (256, 'dense'): '#b07b32', (1024, 'dense'): '#7a5ba6'}
FOM_COLOR = '#b03a3a'


def fmt(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def lam_label(x):
    return r'$\infty$' if x is None else f'{x:g}'


def lam_plain(x):
    return 'inf' if x is None else f'{x:g}'


def lam_key(x):
    return np.inf if x is None else x


def primary(rows, M=64, quad='eq'):
    """The primary block: arm (a)'s own test count and quadrature, trust free."""
    out = [x for x in rows if x['kind'] == 'rom' and x['M'] == M and x['quadrature'] == quad
           and x['trust_y_mode'] == 'free']
    return sorted(out, key=lambda x: -lam_key(x['lambda_rel']))


def blocks(rows):
    keys = []
    for x in rows:
        if x['kind'] == 'rom' and x['trust_y_mode'] == 'free':
            k = (x['M'], x['quadrature'])
            if k not in keys:
                keys.append(k)
    return [(k, sorted([x for x in rows if x['kind'] == 'rom' and x['trust_y_mode'] == 'free'
                        and (x['M'], x['quadrature']) == k],
                       key=lambda x: -lam_key(x['lambda_rel']))) for k in sorted(keys)]


def nondominated(points, err_key, cost_key='median_gpu_ms'):
    """Pareto-optimal in (error, cost); both minimized. Ties keep the cheaper."""
    out = []
    for p in points:
        e, c = p[err_key], p[cost_key]
        if e is None:
            continue
        dominated = any(q[err_key] is not None and q is not p
                        and q[err_key] <= e and q[cost_key] <= c
                        and (q[err_key] < e or q[cost_key] < c) for q in points)
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda p: p[cost_key])


def verdict(block, err_key, cost_key='median_gpu_ms'):
    """The three pre-registered acceptance criteria, evaluated from the table."""
    errs = [x[err_key] for x in block]
    mono = all(b <= a * (1 + 1e-12) for a, b in zip(errs, errs[1:]))
    nd = nondominated(block, err_key, cost_key)
    cost_span = (max(p[cost_key] for p in nd) / min(p[cost_key] for p in nd)) if nd else 0.
    err_span = (max(p[err_key] for p in nd) / max(min(p[err_key] for p in nd), 1e-300)) if nd else 0.
    honest = bool(nd) and all(p['all_completed'] for p in nd)
    return dict(metric=err_key, errors=errs, monotone=bool(mono),
                nondominated=[p['arm'] for p in nd], count=len(nd),
                cost_span=cost_span, error_span=err_span, none_early_stopped=honest,
                criterion_1_monotone=bool(mono),
                criterion_2_frontier=bool(len(nd) >= 3 and cost_span >= 2 and err_span >= 2),
                criterion_3_honest=honest,
                passed=bool(mono and len(nd) >= 3 and cost_span >= 2 and err_span >= 2 and honest))


# ------------------------------------------------------------------ figure ---

def figure(rows, prows, out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    for ax, key, title in [
            (axes[0], 'worst_same_grid_percent', 'all six output times'),
            (axes[1], 'worst_same_grid_evolved_percent', 'evolved times only ($t>0$)')]:
        for (M, quad), blk in blocks(rows):
            xs = [x['median_gpu_ms'] for x in blk]
            ys = [x[key] for x in blk]
            col = BLOCK_COLOR.get((M, quad), '#555555')
            ax.plot(xs, ys, '-', color=col, lw=1.2, zorder=2,
                    label=f'$M={M}$, {quad}')
            for x in blk:
                done = x['all_completed']
                ax.plot(x['median_gpu_ms'], x[key], 'o' if done else 'x',
                        ms=7 if done else 9, mfc=col if done else 'none',
                        mec=col, color=col, mew=1.8, zorder=4)
                ax.annotate(lam_plain(x['lambda_rel']), (x['median_gpu_ms'], x[key]),
                            textcoords='offset points', xytext=(5, 5), fontsize=7, color=col)
        for x in [r for r in rows if r['kind'] == 'fom']:
            v = x[key]
            if v is not None and v > 0:
                ax.plot(x['median_gpu_ms'], v, 'X', ms=12, color=FOM_COLOR, zorder=5,
                        label=f"full-order `{x['arm']}`")
            else:
                ax.axvline(x['median_gpu_ms'], color=FOM_COLOR, ls='--', lw=1.3, zorder=1,
                           label=f"full-order `{x['arm']}` cost (same-grid error 0 by definition)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('median complete-query GPU time (ms), same job')
        ax.set_ylabel('worst same-grid error vs converged FOM (%)')
        ax.set_title(f'Burgers 2D, 256 intervals: worst over {title}')
        ax.grid(True, which='both', alpha=.25)
    axes[0].legend(fontsize=7, loc='best')
    fig.text(.5, -.02, 'filled circle = completed under the shared stopping rule; '
                       'cross = early-stopped (iteration-budget or rejected-step exit). '
                       r'Labels are $\lambda_{\rm rel}$.', ha='center', fontsize=8)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)

    pfig, pax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    meshes = sorted({x['intervals'] for x in prows})
    for i, n in enumerate(meshes):
        blk = sorted([x for x in prows if x['kind'] == 'rom' and x['intervals'] == n],
                     key=lambda x: -lam_key(x['lambda_rel']))
        col = ['#3b6ea5', '#7a5ba6'][i % 2]
        pax.plot([x['median_query_ms'] for x in blk],
                 [x['worst_same_grid_percent'] for x in blk], '-', color=col, lw=1.2)
        for x in blk:
            done = x['all_completed']
            pax.plot(x['median_query_ms'], x['worst_same_grid_percent'], 'o' if done else 'x',
                     ms=7 if done else 9, mfc=col if done else 'none', mec=col, mew=1.8)
            pax.annotate(lam_plain(x['lambda_rel']),
                         (x['median_query_ms'], x['worst_same_grid_percent']),
                         textcoords='offset points', xytext=(5, 5), fontsize=7, color=col)
        pax.plot([], [], 'o-', color=col, label=f'{n} intervals')
        for x in [r for r in prows if r['kind'] == 'fom' and r['intervals'] == n]:
            pax.axvline(x['median_query_ms'], color=FOM_COLOR, ls='--', lw=1.2,
                        label=f"direct DST, {n} intervals")
    pax.set_xscale('log')
    pax.set_yscale('log')
    pax.set_xlabel('median complete host query (ms), same job')
    pax.set_ylabel('worst same-grid error vs the direct solve (%)')
    pax.set_title('Poisson 2D: the same dial')
    pax.grid(True, which='both', alpha=.25)
    pax.legend(fontsize=8, loc='best')
    ppng = Path(str(out_png).replace('-cost.png', '-poisson.png'))
    pfig.savefig(ppng, dpi=200)
    pfig.savefig(str(ppng).replace('.png', '.pdf'))
    plt.close(pfig)
    return dict(png=str(out_png), pdf=str(out_pdf), poisson_png=str(ppng),
                png_sha256=hashlib.sha256(Path(out_png).read_bytes()).hexdigest(),
                poisson_png_sha256=hashlib.sha256(ppng.read_bytes()).hexdigest(),
                plotted_points=2 * len([x for x in rows if x['kind'] == 'rom'])
                + len([x for x in prows if x['kind'] == 'rom']))


# ------------------------------------------------------------------- report --

def build(r, au, sm, pr, pau, figinfo, figname, pfigname):
    rows = au['checks']['arm_table']
    prows = pau['checks']['arm_table']
    cfg = r['config']
    out = []
    w = out.append
    prim = primary(rows)
    v_all = verdict(prim, 'worst_same_grid_percent')
    v_evo = verdict(prim, 'worst_same_grid_evolved_percent')

    w('# The prior dial: is trust in the neural prior a usable inference-time knob?')
    w('')
    w('One frozen checkpoint, one knob. Instead of solving only for the latent code with the '
      'bank coefficients pinned to the neural head, this cell solves for the **full** bank '
      'coefficient vector under a penalty $\\lambda$ pulling it back towards the head, and '
      'prices the whole range of $\\lambda$ from "the head is law" to "the head is a '
      'suggestion". **Final for the six opened Burgers development cases and the twelve '
      'opened Poisson development sources, and provisional as a paper claim**: one checkpoint '
      'per PDE, one training seed, final cohorts sealed.')
    w('')
    w(f"Burgers job `{r.get('job_id')}` on `{r.get('gpu')}` and Poisson job `{pr.get('job_id')}` "
      f"on `{pr.get('gpu')}`, source commit `{r.get('commit')}`, JAX {r['jax_version']}, backend "
      f"`{r['backend']}`, float64, matmul precision `{r['matmul_precision']}`; elapsed "
      f"{r['elapsed_seconds']:.1f} s and {pr['elapsed_seconds']:.1f} s.")
    w('')

    # ------------------------------------------------------------- the knob --
    w('## The knob')
    w('')
    w('Arm (a) of the head ablation pins the bank coefficients to the head and solves')
    w('')
    w(r'$$\min_{z\in\mathbb R^{K}}\ \big\|r_w\big(h_\theta(z)\big)\big\|_2^2 .$$')
    w('')
    w('This cell keeps every network weight, the bank $G$, $K=16$, the initializer policy, '
      '$\\Delta t$, the stopping rule and the output contract, and solves instead')
    w('')
    w(r'$$\min_{z\in\mathbb R^{K},\,c\in\mathbb R^{R}}\ \big\|r_w(c)\big\|_2^2'
      r'\;+\;\lambda\,\big\|R_G\,(c-h_\theta(z))\big\|_2^2 ,$$')
    w('')
    w('with $G=Q_GR_G$ the thin QR of the bank, so $\\|R_G\\delta\\|_2=\\|G\\delta\\|_2$ and the '
      'penalty is the squared **field-norm** distance between the solved state and the state '
      'the head would have produced. Writing $y=R_G(c-h_\\theta(z))$ gives '
      '$u=G\\,h_\\theta(z)+Q_G\\,y$, so the solver never applies $R_G^{-1}$ to a solved vector, '
      'and the residual it sees is')
    w('')
    w(r'$$F(z,y)=\begin{bmatrix} r_w\big(h_\theta(z)+R_G^{-1}y\big)\\[2pt]'
      r'\sqrt{\lambda}\,y\end{bmatrix}\in\mathbb R^{M+R}.$$')
    w('')
    sig = next((x['sigma'] for x in rows if x['sigma']), None)
    w(f"**The scaling of $\\lambda$.** $\\lambda=\\lambda_{{\\rm rel}}\\,\\sigma^2$ with "
      f"$\\sigma=\\|A R_G^{{-1}}\\|_2=\\|\\Phi^\\top Q_G\\|_2$, the exact linear part of "
      f"$\\partial r_w/\\partial y$. That part is state independent because the "
      f"$1/(1+\\Delta t\\,\\nu\\lambda_j)$ row scaling of the weak residual cancels the "
      f"diffusion term exactly, and because $\\Phi$ and $Q_G$ both have orthonormal columns "
      f"$\\sigma$ is the largest principal cosine between the test-mode span and the bank "
      f"span. Measured here: $\\sigma={sig:.10f}$ at $M=64$ — the frozen bank contains the "
      f"lowest sine modes almost exactly. $\\lambda_{{\\rm rel}}=1$ therefore means one unit of "
      f"field-norm departure from the head is penalized as strongly as the strongest linear "
      f"response of the weak residual to it. Every table carries both $\\lambda_{{\\rm rel}}$ "
      f"and $\\sigma$, so the absolute $\\lambda$ can be recovered.")
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  U["supplied dense field u0"] --> IC["arm (a) initializer:<br/>nearest training code,<br/>'
      'K-dim Gauss state fit"]')
    w('  IC -->|"z0, y = 0"| LM["damped Levenberg-Marquardt<br/>on (z, y)"]')
    w('  G["frozen bank G = Q_G R_G"] --> LM')
    w('  H["frozen head h_theta"] --> LM')
    w('  LM --> RW["weak residual r_w(c)<br/>exact linear + EQ or dense advection"]')
    w('  LM --> PEN["prior penalty<br/>sqrt(lambda) y"]')
    w('  RW --> LM')
    w('  PEN --> LM')
    w('  LM --> DEC["decode u = G h(z) + Q_G y"]')
    w('  DEC --> OUT["six dense output fields"]')
    w('  classDef frozen fill:#e8eef6,stroke:#3b6ea5,stroke-width:2px;')
    w('  classDef solved fill:#eef6e8,stroke:#4f8a3d,stroke-width:2px;')
    w('  classDef knob fill:#f6efe4,stroke:#b07b32,stroke-width:3px;')
    w('  class G,H,IC frozen;')
    w('  class LM,DEC solved;')
    w('  class PEN knob;')
    w('```')
    w('')
    w(f"$\\lambda=\\infty$ is implemented as the **exact** elimination $c=h_\\theta(z)$ — the "
      f"$y$ block has zero width and is dropped at trace time — not as a large number, which "
      f"is what makes the first gate a bitwise statement. $\\lambda_{{\\rm rel}}="
      f"{max(x for x in cfg['lambda_rel'] if x is not None):g}$ is the "
      f"numerically-infinite end of the *finite* code path and exists to show that path "
      f"returning to arm (a).")
    w('')

    # ---------------------------------------------------------------- gates --
    w('## Gates')
    w('')
    g2 = au['checks'].get('laminf_reproduces_head_ablation_arm_a', {})
    g3 = pau['checks'].get('laminf_reproduces_pabl01_arm_a', {})
    g0 = au['checks'].get('initial_output_is_lambda_independent', {})
    w('| gate | result |')
    w('| --- | --- |')
    w(f"| (i) local, $\\lambda=\\infty$ through the new path vs the consolidated saved Burgers "
      f"case | {sm['laminf_vs_saved_case']['relative_l2']:.3e} relative "
      f"({sm['laminf_vs_saved_case']['latent_relative_l2']:.3e} on internal latents), "
      f"tolerance {sm['laminf_vs_saved_case']['tolerance']:g} |")
    w(f"| (i) local, same vs the incumbent `accuracy_paths.make_rom` | "
      f"{sm['laminf_vs_incumbent']:.1e} — "
      f"{'**bit-identical**' if sm['laminf_bitwise_identical_to_incumbent'] else 'not bitwise'} |")
    w(f"| (i) local, the *finite* path at $\\lambda_{{\\rm rel}}="
      f"{sm['finite_path_large_lambda']['lambda_rel']:g}$ vs that limit | "
      f"{sm['finite_path_large_lambda']['relative_to_laminf']:.3e} relative, realised "
      f"correction {sm['finite_path_large_lambda']['max_relative_correction']:.2e} of the "
      f"state norm |")
    if g2.get('detail'):
        d = g2['detail']
        w(f"| (ii) in job, `M64_eq_laminf` vs `abl01` `a_neural_eq` on {d['compared']} cases | "
          f"worst {d['worst_relative_delta']:.3e} relative, tolerance {d['tolerance']:g}, "
          f"{d['bitwise_identical_fields']}/{d['compared']} fields bitwise identical across "
          f"jobs — **{'pass' if g2['passed'] else 'FAIL'}** |")
    if g3.get('detail'):
        d = g3['detail']
        w(f"| (iii) in job, Poisson `laminf` vs `pabl01` `a_neural` | worst "
          f"{d['worst_relative_delta']:.3e} relative, tolerance {d['tolerance']:g} — "
          f"**{'pass' if g3['passed'] else 'FAIL'}** |")
    w(f"| (iv) independent NumPy audit recomputing every error from saved fields | worst "
      f"relative disagreement "
      f"{au['checks']['recorded_errors_recomputed_from_saved_fields']['detail']:.2e}; Burgers "
      f"audit {'passed' if au['passed'] else 'FAILED'}, Poisson audit "
      f"{'passed' if pau['passed'] else 'FAILED'} |")
    if g0:
        w(f"| structural, the $t=0$ output is $\\lambda$-independent | "
          f"{'confirmed' if g0['passed'] else 'VIOLATED'} — see below |")
    w('')

    # ---------------------------------------------- the structural finding ---
    t0note = au['checks'].get('all_times_worst_is_at_t0', {}).get('detail')
    w('## What the initializer contract already decides')
    w('')
    w(f"The contract fixes the initializer as arm (a)'s: the nearest training code, a "
      f"$K$-dimensional Gauss state fit, and $y=0$. So the $t=0$ output of every arm is the "
      f"head's compression of the supplied field and **cannot depend on $\\lambda$** — the "
      f"audit asserts this and it holds "
      f"{'exactly' if g0.get('passed') else 'NOT'}. On these cases that compression error is "
      f"also the largest of the six output times"
      + ('' if not t0note else f" (exceptions, arm and argmax index: {t0note})")
      + ", so the worst same-grid error over **all** output times is pinned by construction "
        "and no setting of $\\lambda$ can move it.")
    w('')
    w('That is a real result about this checkpoint, not a defect of the dial, and it is '
      'reported as the pre-registered primary metric below. Because it is uninformative '
      'about $\\lambda$ itself, DESIGN.md amendment 2 — recorded **before** submission, from '
      'the local probe — added the worst same-grid error over the **evolved** times as a '
      'declared secondary metric, where $\\lambda$ can act. Both are reported, and the three '
      'acceptance criteria are evaluated against both.')
    w('')

    # ------------------------------------------------------- the main table --
    w('## Burgers: three layers, every setting')
    w('')
    tight = next((x for x in rows if x['arm'] == 'fft_tight'), None)
    w(f"Layer 1 is the bank projection floor; layer 2 is the best-found reconstruction on that "
      f"arm's own reachable set; layer 3 is the solved error. For every **finite** $\\lambda$ "
      f"layers 1 and 2 coincide, because the penalty restricts the *solver*, not the reachable "
      f"set: the whole bank is still reachable. Only $\\lambda=\\infty$ has a manifold of its "
      f"own. Whatever $\\lambda$ does, it does entirely in the reduction/solver layer.")
    w('')
    w(f"The same-grid metric is the discrepancy from the converged full-order solve on this "
      f"mesh, which is what isolates reduction error from discretization error; the full-order "
      f"model itself carries {tight['worst_reference_percent']:.4f}% worst against the refined "
      f"reference, and both are reported.")
    w('')
    w('| arm | $M$ | $m$ | quad. | $\\lambda_{\\rm rel}$ | solved dim | L1 bank floor % | '
      'L2 best-found % | L3 worst same-grid % | L3 worst evolved % | median evolved % | '
      'worst reference % | realised $\\|y\\|/\\|u\\|$ % | median iters/step | budget exits | '
      'completed | median GPU ms | median host ms |')
    w('|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|')
    for x in rows:
        w('| ' + ' | '.join([
            (f"`{x['arm']}`" if x['kind'] == 'rom' else f"FOM `{x['arm']}`"),
            str(x['M'] or '—'), str(x['m'] or '—'), x['quadrature'] or '—',
            (lam_label(x['lambda_rel']) if x['kind'] == 'rom' else '—'),
            str(x['solved_dimension'] or '—'),
            fmt(x['worst_bank_projection_percent'], 4), fmt(x['worst_best_found_percent'], 4),
            fmt(x['worst_same_grid_percent'], 4), fmt(x['worst_same_grid_evolved_percent'], 4),
            fmt(x['median_same_grid_evolved_percent'], 4), fmt(x['worst_reference_percent'], 4),
            fmt(x['max_relative_correction_percent'], 4),
            fmt(x['median_iterations'], 1),
            ('—' if x['total_budget_exits'] is None else str(x['total_budget_exits'])),
            {True: 'yes', False: 'no', None: '—'}[x['all_completed']],
            fmt(x['median_gpu_ms'], 3), fmt(x['median_host_ms'], 3)]) + ' |')
    w('')
    w('The two full-order rows are **context, not competitors**. `fft_tight` is the converged '
      'solve that *defines* the same-grid metric, so its own same-grid value is zero by '
      'construction and it enters the figure as a cost line. `fft_loose` is the efficient '
      'loose-tolerance control. Arms with $M<R=' + str(r['R']) + '$ at small $\\lambda$ are '
      '**regularized underdetermined solves, not the free bank**: there are fewer weak '
      'equations than bank coefficients, and what the solver returns is set jointly by '
      '$\\lambda$ and by the Levenberg-Marquardt damping. Only $M=1024>R$ reaches the free-bank '
      'limit as $\\lambda\\to0$.')
    w('')
    w(f'![error versus cost]({figname})')
    w('')

    # ---------------------------------------------------- acceptance verdict --
    w('## The verdict against the pre-registered criteria')
    w('')
    w('The criteria were fixed in `DESIGN.md` before any implementation: (1) error monotone in '
      '$\\lambda$; (2) at least three non-dominated points spanning $\\ge2\\times$ in cost '
      '**and** $\\ge2\\times$ in error; (3) none of them early-stopped. They are evaluated on '
      f"the primary block ($M=64$, EQ — arm (a)'s own test count and quadrature).")
    w('')
    w('| metric | (1) monotone | (2) non-dominated points / cost span / error span | '
      '(3) none early-stopped | **verdict** |')
    w('| --- | --- | --- | --- | --- |')
    for v, label in [(v_all, 'primary: worst same-grid, all six output times'),
                     (v_evo, 'secondary: worst same-grid, evolved times only')]:
        w(f"| {label} | {'yes' if v['criterion_1_monotone'] else 'NO'} | "
          f"{v['count']} / {v['cost_span']:.2f}$\\times$ / {v['error_span']:.2f}$\\times$ "
          f"{'(pass)' if v['criterion_2_frontier'] else '(fail)'} | "
          f"{'yes' if v['criterion_3_honest'] else 'NO'} | "
          f"**{'KNOB' if v['passed'] else 'NOT A KNOB'}** |")
    w('')
    w(f"Primary-metric errors along $\\lambda_{{\\rm rel}}="
      f"[{', '.join(lam_plain(x['lambda_rel']) for x in prim)}]$: "
      f"{[round(e, 6) for e in v_all['errors']]} percent. "
      f"Secondary-metric errors on the same arms: {[round(e, 6) for e in v_evo['errors']]} "
      f"percent.")
    w('')
    w('### The non-dominated set')
    w('')
    for key, label in [('worst_same_grid_percent', 'all six output times'),
                       ('worst_same_grid_evolved_percent', 'evolved times only')]:
        nd = nondominated([x for x in rows if x['kind'] == 'rom'], key)
        w(f'Over **every** Burgers arm, worst same-grid error over {label}:')
        w('')
        w('| arm | $M$ | quad. | $\\lambda_{\\rm rel}$ | worst same-grid % | median GPU ms | '
          'completed |')
        w('|---|---:|---|---:|---:|---:|---|')
        for x in nd:
            w('| ' + ' | '.join([f"`{x['arm']}`", str(x['M']), x['quadrature'],
                                 lam_label(x['lambda_rel']), fmt(x[key], 4),
                                 fmt(x['median_gpu_ms'], 3),
                                 'yes' if x['all_completed'] else '**no**']) + ' |')
        w('')
    loose = next((x for x in rows if x['arm'] == 'fft_loose'), None)
    roms = [x for x in rows if x['kind'] == 'rom']
    for fom in [x for x in [loose, tight] if x]:
        for key, label in [('worst_same_grid_percent', 'all times'),
                           ('worst_same_grid_evolved_percent', 'evolved times')]:
            beat = [x for x in roms if x[key] is not None and fom[key] is not None
                    and x[key] <= fom[key] and x['median_gpu_ms'] <= fom['median_gpu_ms']]
            w(f"Against full-order `{fom['arm']}` ({fmt(fom[key], 4)}% same-grid over {label}, "
              f"{fom['median_gpu_ms']:.3f} ms median GPU, {fom['median_host_ms']:.3f} ms "
              f"complete host query): "
              + ((', '.join(f"`{x['arm']}`" for x in beat) + ' dominate it on both axes.')
                 if beat else '**no arm dominates it on both axes.**'))
        w('')

    # ---------------------------------------------------------- the controls --
    w('## Controls')
    w('')
    w('### Does the test count decide what $\\lambda$ can do?')
    w('')
    for (M, quad), blk in blocks(rows):
        if quad != 'dense' and M != 64:
            continue
        inf = blk[0]
        best = min((x for x in blk if x['worst_same_grid_evolved_percent'] is not None),
                   key=lambda x: x['worst_same_grid_evolved_percent'])
        w(f"- $M={M}$, {quad} ({'overdetermined' if M > r['R'] else 'underdetermined'} in the "
          f"bank, $R={r['R']}$): evolved-time error moves from "
          f"{inf['worst_same_grid_evolved_percent']:.4f}% at $\\lambda=\\infty$ to "
          f"{best['worst_same_grid_evolved_percent']:.4f}% at $\\lambda_{{\\rm rel}}="
          f"{lam_plain(best['lambda_rel'])}$, a factor "
          f"{inf['worst_same_grid_evolved_percent'] / max(best['worst_same_grid_evolved_percent'], 1e-300):.3f}, "
          f"for a cost factor {best['median_gpu_ms'] / inf['median_gpu_ms']:.3f}; that best point "
          f"{'completes' if best['all_completed'] else '**does not complete**'} under the "
          f"shared stopping rule, and the realised correction is "
          f"{best['max_relative_correction_percent']:.4f}% of the state norm.")
    w('')
    w('### Is the empirical-quadrature rule still valid off the head manifold?')
    w('')
    w('The EQ rule is fitted on **decoder-output** advection snapshots, so once $c$ leaves the '
      'head manifold it is extrapolating. The paired dense block at the same $M$ is what '
      'measures that.')
    w('')
    w('| $\\lambda_{\\rm rel}$ | EQ worst evolved % | dense worst evolved % | difference (pp) | '
      'EQ median GPU ms | dense median GPU ms | cost factor |')
    w('|---:|---:|---:|---:|---:|---:|---:|')
    eqb = {x['lambda_rel']: x for x in primary(rows, 64, 'eq')}
    for x in primary(rows, 64, 'dense'):
        e = eqb.get(x['lambda_rel'])
        if not e:
            continue
        w('| ' + ' | '.join([
            lam_label(x['lambda_rel']),
            fmt(e['worst_same_grid_evolved_percent'], 4),
            fmt(x['worst_same_grid_evolved_percent'], 4),
            fmt(e['worst_same_grid_evolved_percent'] - x['worst_same_grid_evolved_percent'], 4),
            fmt(e['median_gpu_ms'], 3), fmt(x['median_gpu_ms'], 3),
            fmt(x['median_gpu_ms'] / e['median_gpu_ms'], 3)]) + ' |')
    w('')
    w('### Does the trust radius decide the answer instead of $\\lambda$?')
    w('')
    ty = [x for x in rows if x['kind'] == 'rom' and x['trust_y_mode'] == 'latent']
    if ty:
        w("Arm (a)'s trust radius is 1% of the radius of the training code cloud, "
          f"{r['trust_radius']:.9g} — a *latent-space* number. The correction $y$ is in "
          "field-norm units, so the primary sweep bounds $\\|\\delta z\\|$ by it and leaves "
          "$\\|\\delta y\\|$ free. These rows repeat two $\\lambda$ values with "
          "$\\|\\delta y\\|$ bounded by the same number, the correction-ladder convention.")
        w('')
        w('| $\\lambda_{\\rm rel}$ | free $\\delta y$: worst evolved % / GPU ms / completed | '
          'bounded $\\delta y$: worst evolved % / GPU ms / completed |')
        w('|---:|---|---|')
        for x in sorted(ty, key=lambda z: -lam_key(z['lambda_rel'])):
            f_ = eqb.get(x['lambda_rel'])
            w(f"| {lam_label(x['lambda_rel'])} | "
              f"{fmt(f_['worst_same_grid_evolved_percent'], 4)} / {fmt(f_['median_gpu_ms'], 3)} / "
              f"{'yes' if f_['all_completed'] else 'no'} | "
              f"{fmt(x['worst_same_grid_evolved_percent'], 4)} / {fmt(x['median_gpu_ms'], 3)} / "
              f"{'yes' if x['all_completed'] else 'no'} |")
    w('')

    # ------------------------------------------------------------- Poisson ---
    w('## Poisson: the same dial, with $c$ eliminated exactly')
    w('')
    w('The Poisson weak residual is exactly $r(c)=Bc-f_m$, linear in $c$, so for fixed $z$')
    w('')
    w(r'$$\big(B_y^\top B_y+\lambda I\big)\,y=B_y^\top\big(f_m-B\,h_\theta(z)\big),'
      r'\qquad B_y=B\,R_G^{-1},$$')
    w('')
    w('is closed form. It is evaluated from one thin SVD of $B_y$ built at setup — never from '
      'the Gram, which would square the condition number — so the sweep shares one '
      'factorization and the inner solve is two $R\\times R$ matvecs inside the timed query. '
      'The outer Levenberg-Marquardt therefore runs in $z$ only and differentiates through the '
      'closed form, the solved dimension stays $K$, and the trust radius keeps its original '
      'latent meaning. Here $M=' + str(next((x['M'] for x in prows if x['M']), None))
      + '>R=' + str(pr['R']) + '$, so $\\lambda\\to0$ **does** reach the free bank.')
    w('')
    w('| intervals | $\\lambda_{\\rm rel}$ | L1 bank floor % | L2 best-found % | '
      'L3 worst same-grid % | median same-grid % | worst physical % | '
      'realised $\\|y\\|/\\|u\\|$ % | median iters | completed | median host ms | '
      'median device ms |')
    w('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|')
    for x in prows:
        w('| ' + ' | '.join([
            str(x['intervals']),
            (lam_label(x['lambda_rel']) if x['kind'] == 'rom' else f"FOM `{x['arm']}`"),
            fmt(x['worst_bank_projection_percent'], 4), fmt(x['worst_best_found_percent'], 4),
            fmt(x['worst_same_grid_percent'], 4), fmt(x['median_same_grid_percent'], 4),
            fmt(x['worst_error_percent'], 4), fmt(x['max_relative_correction_percent'], 4),
            fmt(x['median_iterations'], 1),
            {True: 'yes', False: 'no', None: '—'}[x['all_completed']],
            fmt(x['median_query_ms'], 4), fmt(x['median_device_ms'], 4)]) + ' |')
    w('')
    w(f'![Poisson]({pfigname})')
    w('')
    for n in sorted({x['intervals'] for x in prows}):
        blk = sorted([x for x in prows if x['kind'] == 'rom' and x['intervals'] == n],
                     key=lambda z: -lam_key(z['lambda_rel']))
        errs = [x['worst_same_grid_percent'] for x in blk]
        mono = all(b <= a * (1 + 1e-12) for a, b in zip(errs, errs[1:]))
        nd = nondominated(blk, 'worst_same_grid_percent', 'median_query_ms')
        cs = (max(p['median_query_ms'] for p in nd) / min(p['median_query_ms'] for p in nd)) if nd else 0
        es = (max(p['worst_same_grid_percent'] for p in nd)
              / max(min(p['worst_same_grid_percent'] for p in nd), 1e-300)) if nd else 0
        fom = next((x for x in prows if x['kind'] == 'fom' and x['intervals'] == n), None)
        w(f"- **{n} intervals.** Worst same-grid error is "
          f"{'monotone decreasing' if mono else '**NOT monotone**'} in $\\lambda$: "
          f"{[round(e, 4) for e in errs]} percent from $\\lambda=\\infty$ down. "
          f"{len(nd)} non-dominated points spanning {cs:.2f}$\\times$ in cost and "
          f"{es:.2f}$\\times$ in error; "
          f"{'all complete' if all(p['all_completed'] for p in nd) else 'some are early-stopped'}. "
          + (f"The direct DST solve costs {fom['median_query_ms']:.4f} ms with zero same-grid "
             f"error by definition." if fom else ''))
    w('')

    # ---------------------------------------------- deviations and caveats ---
    w('## Recorded deviations and caveats')
    w('')
    w(f"1. **The trust radius applies to the latent block only** in the primary sweep. Arm "
      f"(a)'s radius {r['trust_radius']:.9g} is 1% of the training code cloud's radius, a "
      f"latent-space quantity; carrying it to $y$ would cap the field correction at that same "
      f"number and would regularize the small-$\\lambda$ end with something that is not "
      f"$\\lambda$. The trust-control rows above measure the size of that choice. At "
      f"$\\lambda=\\infty$ there is no $y$ block and the two are identical.")
    w(f"2. **The residual tolerance now includes the penalty.** The solver stops on $\\|F\\|$, "
      f"and $F$ carries the $\\sqrt\\lambda\\,y$ block, so the absolute residual exit is "
      f"harder to reach at large $\\lambda$ than it is for arm (a) at the same $z$. The "
      f"stationarity and small-step exits are unaffected; every row reports its budget exits "
      f"and completion status.")
    w(f"3. **Arms above {cfg['gauss_jordan_max']} unknowns use a pivoted dense step solve** "
      f"instead of the incumbent unrolled Gauss-Jordan. More accurate, not weaker.")
    w(f"4. **Empirical quadrature is fitted only at $M=64$** ($m={cfg['quadrature_multiplier']}M$), "
      f"which is arm (a)'s own rule, refit with the identical seed, candidate cap, fit-state "
      f"count and code table. At $M=256$ and $M=1024$ a nonnegative-least-squares rule at "
      f"$m=4M$ is not constructible inside the job budget, so those blocks use the exact dense "
      f"grid sum, stated per row.")
    w(f"5. **Layers 1 and 2 coincide at every finite $\\lambda$**, as explained above. The "
      f"reported layer 2 for a finite-$\\lambda$ row is therefore the bank projection floor, "
      f"not a separate measurement.")
    w(f"6. **Completion, not stationarity, is the honest status.** The normalized gradient is "
      f"scale invariant and stays of order one for an arm whose reduced fit is attainable, "
      f"which then exits by the small-step rule with a better fit.")
    w(f"7. One checkpoint per PDE, one mesh for Burgers, one training seed, development cases "
      f"only. These are within-job comparisons on one GPU with burn-in before every timed "
      f"block and all repetitions retained; no cross-job timing ratio is used anywhere.")
    w('')

    # -------------------------------------------------------------- glossary --
    w('## Glossary')
    w('')
    for term, text in [
        ('$\\lambda$ / $\\lambda_{\\rm rel}$', 'the knob: how strongly the solver is pulled back '
         'towards the neural head. $\\lambda_{\\rm rel}$ is the dimensionless version, '
         '$\\lambda=\\lambda_{\\rm rel}\\sigma^2$. Large means "trust the head"; small means '
         '"trust the equations".'),
        ('$\\sigma$', 'the scale that makes $\\lambda$ dimensionless: how strongly the weak '
         'residual responds, through its exact linear terms, to one unit of field-norm '
         'departure from the head. Geometrically, the largest principal cosine between the '
         'test-mode span and the bank span.'),
        ('bank / $G$', 'the fixed set of spatial fields the reduced state is built from. Frozen '
         'here; nothing is retrained.'),
        ('head / $h_\\theta$', 'the trained network turning the few solved coordinates into bank '
         'coefficients. This is the "prior" the dial trusts or distrusts.'),
        ('$K$ / $R$ / solved dimension', 'the latent dimension (16) / the number of bank '
         'coefficients (512 on Burgers, 128 on Poisson) / how many numbers the online solver '
         'actually solves for.'),
        ('$M$ (test modes)', 'how many smooth functions the PDE residual is averaged against. '
         'With the bank free, $M<R$ means fewer equations than unknowns.'),
        ('underdetermined', 'fewer weak equations than free bank coefficients ($M<R$). The '
         'answer is then decided jointly by $\\lambda$ and by the solver damping, and it is '
         'NOT the free bank however small $\\lambda$ gets.'),
        ('$m$ (quadrature points) / EQ / dense', 'grid points used by the learned quadrature rule '
         'for the nonlinear advection term / that rule / the exact full grid sum with no '
         'approximation.'),
        ('same-grid error', 'difference from the converged full-order solve on the same mesh, '
         'normalised by the initial reference field norm. Isolates reduction error from '
         'discretization error.'),
        ('all times / evolved times', 'worst over the six output times $t=0,0.05,\\dots,0.25$ / '
         'worst over the five with $t>0$. The $t=0$ output is the head\'s compression of the '
         'supplied field and is the same at every $\\lambda$, which is why both are reported.'),
        ('reference error', 'difference from the refined reference solved on a much finer mesh '
         'and time step; it contains this mesh\'s discretization error as well.'),
        ('L1 bank projection floor', 'the smallest error any bank coefficients at all could '
         'reach on the reference field. A floor no solver can beat.'),
        ('L2 best-found reconstruction', 'the smallest error that arm\'s own reachable set can '
         'reach on the reference field with no PDE involved. Separates representation from '
         'dynamics.'),
        ('L3 solved error', 'what the real online solve actually produced.'),
        ('realised $\\|y\\|/\\|u\\|$', 'how far the solved state actually departed from the '
         'head\'s prediction, as a fraction of the state norm. If this is near zero the prior '
         'was never binding.'),
        ('iterations / budget exits', 'Levenberg-Marquardt steps per time step, hardware-free / '
         'time steps that stopped because the iteration cap was reached rather than because a '
         'stopping criterion was met.'),
        ('completed / early-stopped', 'the shared stopping rule terminated everywhere with no '
         'budget exit and no rejected-step exit / it did not. An early-stopped point is a '
         'legitimate point on a cost curve but is not a converged solution and is never '
         'relabelled as one.'),
        ('non-dominated (Pareto)', 'a setting no other setting beats on both error and cost at '
         'once. A knob worth having produces several of them, well spread.'),
        ('complete query', 'the timed unit: one supplied dense field on the GPU to the dense '
         'output fields, including the initial fit, the solve and the decode. The host column '
         'adds the same-invocation transfers.'),
        ('FOM', 'full-order model: the unreduced solver. `fft_tight` / the direct DST solve is '
         'the converged reference on this mesh; `fft_loose` is the efficient loose-tolerance '
         'control. They are context, not competitors.'),
        ('pp', 'percentage points.'),
        ('development / final cohort', 'cases usable for method selection / cases kept '
         'unopened.'),
        ('variable projection (Golub-Pereyra)', 'eliminating the coefficients that enter '
         'linearly, then solving only the remaining nonlinear ones, differentiating through '
         'the elimination. Used on Poisson, impossible on Burgers because advection makes the '
         'residual quadratic in the coefficients.'),
    ]:
        w(f'- **{term}:** {text}')
    w('')
    w('---')
    w('')
    w(f"Generated by `experiments/prior-dial/reports/generate_prior_dial.py` from the Burgers "
      f"`result.json` (SHA256 `{au['result_sha256']}`), the Poisson `result.json` (SHA256 "
      f"`{pau['result_sha256']}`), both independent audit JSONs and the retained smoke "
      f"evidence. Figures are produced by the same script (Burgers PNG SHA256 "
      f"`{figinfo['png_sha256']}`, Poisson PNG SHA256 `{figinfo['poisson_png_sha256']}`, "
      f"{figinfo['plotted_points']} plotted points). Every number and marker is read from "
      f"those files.")
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--smoke', required=True)
    p.add_argument('--poisson-result', required=True)
    p.add_argument('--poisson-audit', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    au = json.loads(Path(a.audit).read_text())
    sm = json.loads(Path(a.smoke).read_text())
    pr = json.loads(Path(a.poisson_result).read_text())
    pau = json.loads(Path(a.poisson_audit).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out.with_suffix(''))
    figinfo = figure(au['checks']['arm_table'], pau['checks']['arm_table'],
                     Path(stem + '-cost.png'), Path(stem + '-cost.pdf'))
    text = build(r, au, sm, pr, pau, figinfo, Path(stem + '-cost.png').name,
                 Path(stem + '-poisson.png').name)
    out.write_text(text)
    print(out, hashlib.sha256(text.encode()).hexdigest())
    print(json.dumps(figinfo, indent=2))


if __name__ == '__main__':
    main()
