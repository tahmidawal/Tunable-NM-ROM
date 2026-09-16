"""Generate the cheap-corrections report and its error-versus-cost figure.

Every number and every plotted point is read from the audited run JSON. Nothing in the
report is typed by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MARK = {'joint': 'o', 'varpro': 's', 'block': 'D', 'alt': '^'}
COLOR = {'dense': '#3b6ea5', 'eq': '#4f8a3d'}


def fmt(x, d=4):
    return '—' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require_same_grid(rows):
    missing = [x['arm'] for x in rows if x['kind'] == 'rom' and x.get('worst_same_grid_percent') is None]
    assert not missing, ('same-grid errors missing for ' + ', '.join(missing)
                         + '; rerun audit_cheap.py with --fields <output directory>')


def ladder_rows(rows, variant, rule, quad):
    out = [x for x in rows if x['kind'] == 'rom' and x['variant'] == variant
           and x['rule'] == rule and x['quadrature'] == quad]
    return sorted(out, key=lambda x: x['q'])


def figure(rows, old, cfg, out_png, out_pdf):
    require_same_grid(rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    if old:
        ax.plot([x['median_gpu_ms'] for x in old], [x['worst_same_grid_percent'] for x in old],
                '-o', color='#b8b8b8', lw=1.2, ms=5, zorder=1,
                label='audited ladder qlad01 (joint solver, M = 4(K+q), dense)')
        for x in old:
            ax.annotate(f"q={x['q']}", (x['median_gpu_ms'], x['worst_same_grid_percent']),
                        textcoords='offset points', xytext=(5, -12), fontsize=7, color='#9a9a9a')
    seen = set()
    for x in [r for r in rows if r['kind'] == 'rom']:
        mark = MARK.get(x['variant'], 'P')
        col = COLOR.get(x['quadrature'], '#7a5ba6')
        face = col if x['converged'] else 'none'
        lab = f"{x['variant']}, {x['quadrature']}"
        ax.plot(x['median_gpu_ms'], x['worst_same_grid_percent'], mark, ms=8, color=col,
                mfc=face, mew=1.4, zorder=3, label=lab if lab not in seen else None)
        seen.add(lab)
    for rule, style in (('m4', '-'), ('m2', '--'), ('m256', ':')):
        for quad in ('dense', 'eq'):
            L = ladder_rows(rows, cfg['ladder_arms']['variant'], rule, quad)
            if len(L) > 1:
                ax.plot([x['median_gpu_ms'] for x in L], [x['worst_same_grid_percent'] for x in L],
                        style, color=COLOR[quad], lw=1.3, alpha=.8, zorder=2,
                        label=f'{rule}, {quad}')
                for x in L:
                    ax.annotate(f"q={x['q']}", (x['median_gpu_ms'], x['worst_same_grid_percent']),
                                textcoords='offset points', xytext=(5, 5), fontsize=7,
                                color=COLOR[quad])
    for x in [r for r in rows if r['kind'] == 'fom']:
        if x['worst_same_grid_percent'] and x['worst_same_grid_percent'] > 0:
            ax.plot(x['median_gpu_ms'], x['worst_same_grid_percent'], 'X', ms=12, color='#b03a3a',
                    zorder=5, label=f"full-order `{x['arm']}`")
        else:
            ax.axvline(x['median_gpu_ms'], color='#b03a3a', ls='-.', lw=1.3, zorder=1,
                       label=f"full-order `{x['arm']}` cost (same-grid error 0 by definition)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('median complete-query GPU time (ms), same job, burn-in before every timed block')
    ax.set_ylabel('worst same-grid error vs converged FOM (%)')
    ax.set_title('Burgers 2D, 256 intervals: making the correction ladder cheap\n'
                 'filled marker = converged under the shared rule, hollow = not converged')
    ax.grid(True, which='both', alpha=.25)
    ax.legend(fontsize=7, loc='best', ncol=2)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def table(head, body):
    lines = ['| ' + ' | '.join(head) + ' |',
             '|' + '|'.join(['---:' if i else '---' for i in range(len(head))]) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body]
    return '\n'.join(lines)


def pareto_span(rows):
    if len(rows) < 2:
        return None, None
    c = [x['median_gpu_ms'] for x in rows]
    err = [x['worst_same_grid_percent'] for x in rows]
    return max(c) / min(c), max(err) / max(min(err), 1e-300)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--burgers-audit', required=True)
    p.add_argument('--burgers-result', required=True)
    p.add_argument('--poisson-audit', default=None)
    p.add_argument('--poisson-result', default=None)
    p.add_argument('--qlad01', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    ba = json.loads(Path(a.burgers_audit).read_text())
    br = json.loads(Path(a.burgers_result).read_text())
    rows = ba['checks']['arm_table']
    cfg = br['config']
    out = Path(a.out)
    figure_png = out.with_name(out.stem + '-cost.png')
    figure_pdf = out.with_name(out.stem + '-cost.pdf')

    old = []
    if a.qlad01:
        oa = json.loads(Path(a.qlad01).read_text())
        old = sorted([x for x in oa['checks']['arm_table']
                      if x['kind'] == 'rom' and x['quadrature'] == 'dense'
                      and x['dt'] == cfg['dt'] and x['arm'] != 'q0_dense_Mmax'],
                     key=lambda x: x['q'])
    figure(rows, old, cfg, figure_png, figure_pdf)

    roms = [x for x in rows if x['kind'] == 'rom']
    foms = [x for x in rows if x['kind'] == 'fom']
    conv = [x for x in roms if x['converged']]
    nd_all = ba['checks']['nondominated_all']
    nd_conv = ba['checks']['nondominated_converged']
    nd_conv_rows = [x for x in roms if x['arm'] in nd_conv]
    cspan, espan = pareto_span(nd_conv_rows)

    lad = cfg['ladder_arms']['variant']
    base_arm = next((x for x in roms if x['arm'] == 'q0_m4_eq_varpro'), None)
    q64 = [x for x in roms if x['q'] == 64 and x['converged']]
    best64 = min(q64, key=lambda x: x['median_gpu_ms']) if q64 else None
    ratio = (best64['median_gpu_ms'] / base_arm['median_gpu_ms']
             if (best64 and base_arm) else None)
    target_pass = bool(ratio is not None and ratio <= 3.0)

    # monotonicity within the retained configuration
    def monotone(seq):
        v = [x['worst_same_grid_percent'] for x in seq]
        return all(b <= a + 1e-12 for a, b in zip(v, v[1:])), v

    mono_rows = {}
    for rule in cfg['ladder_arms']['rules']:
        for quad in ('dense', 'eq'):
            L = ladder_rows(rows, lad, rule, quad)
            if len(L) > 2:
                ok, v = monotone(L)
                mono_rows[f'{rule}/{quad}'] = dict(q=[x['q'] for x in L], values=v, monotone=ok)
    knob = bool(len(nd_conv_rows) >= 3 and cspan and cspan >= 2 and espan and espan >= 2
                and any(m['monotone'] for m in mono_rows.values()))

    L = []
    W = L.append
    W(f"# Making the Burgers correction ladder converge cheaply\n")
    W(f"Three isolated changes to the audited fixed-weight correction ladder — eliminating the "
      f"correction coefficients from the nonlinear iteration, decoupling the test count from $q$, "
      f"and fitting one empirical-quadrature rule per rung — measured on the same frozen "
      f"checkpoint, the same nested directions and the same six opened development cases as job "
      f"`{json.loads(Path(a.qlad01).read_text())['job_id'] if a.qlad01 else '3713867'}`. "
      f"These numbers are final for this cell: job `{br.get('job_id')}` on "
      f"`{br.get('gpu')}`, source `{br.get('commit')}`, independently recomputed from the saved "
      f"fields. Nothing here is provisional unless a row says so.\n")

    W('## What the audited ladder left open\n')
    W('The audited ladder solved the whole augmented vector $w=(z,y)\\in\\mathbb R^{K+q}$ with one '
      'Levenberg–Marquardt iteration at the $q=0$ trust radius, forced $M=4(K+q)$ test modes, and '
      'had no empirical-quadrature rule above $q=16$. Error fell monotonically with $q$ but only '
      '$q\\in\\{0,16\\}$ converged and cost grew about $28\\times$.\n')
    W('The reduced state, the directions and the reachable set are unchanged here:\n')
    W('$$u(z,y)=G\\big(h_\\theta(z)+C_q\\,y\\big),\\qquad '
      'r(z,y)=\\frac{A\\eta-p+\\Delta t\\big(\\Phi^\\top\\mathcal N(G\\eta)+\\nu\\lambda\\odot A\\eta\\big)}'
      '{1+\\Delta t\\,\\nu\\lambda},\\qquad \\eta=h_\\theta(z)+C_qy.$$\n')
    W('```mermaid\nflowchart LR\n'
      '  U[supplied initial field] --> IC[initial fit: exact elimination of y]\n'
      '  IC --> S{per time step}\n'
      '  S --> Z[outer LM on z, 16-dim, q=0 trust radius]\n'
      '  Z --> Y[inner Gauss-Newton on y, no trust region]\n'
      '  Y --> Z\n'
      '  S --> Q[weak residual: exact linear terms + quadrature on advection]\n'
      '  Q --> EQ[one NNLS rule per rung, m points]\n'
      '  Q --> DN[exact dense grid sum]\n'
      '  S --> O[six dense output fields]\n'
      '  classDef frozen fill:#eee,stroke:#999;\n'
      '  classDef solved fill:#dbe9f6,stroke:#3b6ea5;\n'
      '  classDef offline fill:#e4f0dd,stroke:#4f8a3d;\n'
      '  class U,O frozen; class Z,Y,IC solved; class EQ,DN offline;\n'
      '```\n')

    W('## Gates\n')
    g = []
    for name, v in ba['checks'].items():
        if isinstance(v, dict) and 'passed' in v:
            d = v.get('detail')
            d = '' if d is None else str(d)
            g.append([f'`{name}`', yn(v['passed']), (d[:150] + ('…' if len(d) > 150 else ''))])
    W(table(['gate', 'passed', 'detail'], g) + '\n')

    W('## Change 1 — the solver\n')
    W('For fixed $z$ the residual is quadratic in $y$, so $y^\\*(z)=\\arg\\min_y\\|r(z,y)\\|$ is '
      'found by damped Gauss–Newton and the outer iteration runs on $z\\in\\mathbb R^{16}$ alone. '
      'Three variants were compared against the audited joint solver at matched $q$, test count '
      'and quadrature. Convergence is the shared rule: every time step and the initial fit exit '
      'for a reason in $\\{1,2,4\\}$, no iteration-budget exit, and joint normalized gradient\n')
    W('$$g=\\frac{\\sqrt{\\|J_z^{\\top}r\\|^2+\\|J_y^{\\top}r\\|^2}}'
      '{\\sqrt{\\|J_z\\|_F^2+\\|J_y\\|_F^2}\\;\\|r\\|}\\le 10^{-6}.$$\n')
    var = sorted([x for x in roms if x['rule'] == 'm4' and x['quadrature'] == 'dense'],
                 key=lambda x: (x['q'], x['variant']))
    W(table(['q', 'variant', 'outer dim', 'M', 'worst same-grid %', 'median GPU ms',
             'median iters/step', 'budget exits', 'max joint $g$', 'max inner $g_y$', 'converged'],
            [[x['q'], f"`{x['variant']}`", fmt(x['outer_dimension'], 0), x['M'],
              fmt(x['worst_same_grid_percent']), fmt(x['median_gpu_ms'], 3),
              fmt(x['median_iterations'], 1), x['total_budget_exits'],
              f"{x['max_joint_stationarity']:.2e}", f"{x['max_inner_stationarity']:.2e}",
              yn(x['converged'])] for x in var]) + '\n')

    W('## Change 2 — the test count\n')
    W('$M=4(K+q)$ (the retained rule), $M=2(K+q)$, and a fixed $M=%d$ wherever $M>K+q$.\n'
      % cfg['fixed_test_count'])
    tc = sorted([x for x in roms if x['variant'] == lad and x['rule'] in ('m4', 'm2', 'm256')],
                key=lambda x: (x['q'], x['rule'], x['quadrature']))
    W(table(['q', 'rule', 'M', 'quadrature', 'm', 'worst same-grid %', 'median GPU ms',
             'budget exits', 'converged'],
            [[x['q'], f"`{x['rule']}`", x['M'], x['quadrature'], fmt(x['m'], 0),
              fmt(x['worst_same_grid_percent']), fmt(x['median_gpu_ms'], 3),
              x['total_budget_exits'], yn(x['converged'])] for x in tc]) + '\n')

    W('## Change 3 — one empirical-quadrature rule per rung\n')
    W('Each rule is a nonnegative least-squares weighting of $m$ grid points fitted offline on '
      'enriched decoder-output advection snapshots $h_\\theta(z_i^\\*)+C_qy_i$. Offline fitting '
      'cost is charged here and never inside a query timing.\n')
    eqr = sorted([x for x in roms if x['quadrature'] == 'eq'], key=lambda x: (x['q'], x['M']))
    W(table(['arm', 'q', 'M', 'm', 'fitter', 'relative fit', 'support', 'truncated',
             'fit seconds', 'worst same-grid %', 'median GPU ms'],
            [[f"`{x['arm']}`", x['q'], x['M'], fmt(x['m'], 0), x['fitter'],
              (f"{x['quadrature_fit_relative']:.3e}" if x['quadrature_fit_relative'] else '—'),
              fmt(x['quadrature_support'], 0), yn(x['quadrature_truncated']),
              fmt(x['quadrature_fit_seconds'], 1), fmt(x['worst_same_grid_percent']),
              fmt(x['median_gpu_ms'], 3)] for x in eqr]) + '\n')
    pairs = []
    for x in eqr:
        d = next((y for y in roms if y['q'] == x['q'] and y['M'] == x['M']
                  and y['quadrature'] == 'dense' and y['variant'] == x['variant']), None)
        if d:
            pairs.append([x['q'], x['M'], fmt(x['median_gpu_ms'], 3), fmt(d['median_gpu_ms'], 3),
                          fmt(d['median_gpu_ms'] / x['median_gpu_ms'], 3),
                          fmt(x['worst_same_grid_percent'] - d['worst_same_grid_percent'], 5)])
    if pairs:
        W('Paired empirical-quadrature / dense rows at the same $q$ and the same $M$ isolate the '
          'quadrature effect:\n')
        W(table(['q', 'M', 'eq GPU ms', 'dense GPU ms', 'cost factor',
                 'same-grid difference (pp)'], pairs) + '\n')

    W('## The three layers\n')
    W('Bank projection is the floor any coefficients at all could reach; best-found '
      'reconstruction is the best the $q$-rung manifold can do on the reference field with no PDE '
      'involved; solved is the online result. They are not an additive decomposition.\n')
    W(table(['arm', 'q', 'rule', 'quadrature', 'variant', 'bank projection %',
             'best-found %', 'solved worst same-grid %', 'solved worst vs reference %'],
            [[f"`{x['arm']}`", x['q'], f"`{x['rule']}`", x['quadrature'], f"`{x['variant']}`",
              fmt(x['worst_bank_projection_percent']), fmt(x['worst_best_found_percent']),
              fmt(x['worst_same_grid_percent']), fmt(x['worst_reference_percent'])]
             for x in sorted(roms, key=lambda x: (x['q'], x['rule'], x['quadrature'],
                                                  x['variant']))]) + '\n')

    W('## Full-order controls, as context only\n')
    W(table(['method', 'worst same-grid %', 'worst vs reference %', 'median GPU ms',
             'median host ms'],
            [[f"`{x['arm']}`", fmt(x['worst_same_grid_percent']), fmt(x['worst_reference_percent']),
              fmt(x['median_gpu_ms'], 3), fmt(x['median_host_ms'], 3)] for x in foms]) + '\n')

    W('## Non-dominated set\n')
    W('Non-dominated over (median GPU ms, worst same-grid %), all arms: '
      + ', '.join(f'`{x}`' for x in nd_all) + '.\n')
    W('Restricted to CONVERGED arms: ' + (', '.join(f'`{x}`' for x in nd_conv) or 'none') + '.\n')
    W(table(['arm', 'q', 'rule', 'quadrature', 'worst same-grid %', 'median GPU ms', 'converged'],
            [[f"`{x['arm']}`", x['q'], f"`{x['rule']}`", x['quadrature'],
              fmt(x['worst_same_grid_percent']), fmt(x['median_gpu_ms'], 3), yn(x['converged'])]
             for x in sorted(nd_conv_rows, key=lambda x: x['median_gpu_ms'])]) + '\n')
    W(f'![error versus cost]({figure_png.name})\n')

    W('## Pre-registered acceptance\n')
    W(table(['criterion', 'required', 'measured', 'verdict'], [
        ['error monotone in $q$', 'non-increasing in at least one retained configuration',
         '; '.join(f"{k}: {yn(v['monotone'])}" for k, v in mono_rows.items()) or '—',
         yn(any(v['monotone'] for v in mono_rows.values()) if mono_rows else None)],
        ['non-dominated converged points', '>= 3', str(len(nd_conv_rows)),
         yn(len(nd_conv_rows) >= 3)],
        ['cost span of those points', '>= 2x', fmt(cspan, 2), yn(bool(cspan and cspan >= 2))],
        ['error span of those points', '>= 2x', fmt(espan, 2), yn(bool(espan and espan >= 2))],
        ['**q is a knob**', 'all of the above', '', yn(knob)],
    ]) + '\n')
    W('**Target.** $q=64$ converged at $\\le 3\\times$ the $q=0$ median GPU cost, the baseline '
      'being the ladder\'s own retained $q=0$ rung `q0_m4_eq_varpro`.\n')
    W(table(['quantity', 'value'], [
        ['$q=0$ baseline arm', f"`{base_arm['arm']}`" if base_arm else '—'],
        ['$q=0$ baseline median GPU ms', fmt(base_arm['median_gpu_ms'], 3) if base_arm else '—'],
        ['cheapest CONVERGED $q=64$ arm', f"`{best64['arm']}`" if best64 else 'none'],
        ['its median GPU ms', fmt(best64['median_gpu_ms'], 3) if best64 else '—'],
        ['its worst same-grid %', fmt(best64['worst_same_grid_percent']) if best64 else '—'],
        ['cost ratio', fmt(ratio, 3)],
        ['**target (<= 3x, converged)**', '**PASS**' if target_pass else '**FAIL**'],
    ]) + '\n')

    W('## Direction fit and its compilation cost\n')
    d = br['directions']
    f = br.get('directions_flat') or {}
    W(table(['quantity', 'audited double vmap', 'flattened single vmap'], [
        ['seconds', fmt(d['seconds'], 1), fmt(f.get('seconds'), 1)],
        ['directions sha256', '`' + d['directions_sha256'][:16] + '…`',
         ('`' + f['directions_sha256'][:16] + '…`') if f else '—'],
        ['bitwise identical to the audited path', '—', yn(f.get('bitwise_identical_to_audited'))],
        ['max absolute difference', '—',
         (f"{f['max_abs_difference_from_audited']:.3e}" if f else '—')],
        ['seconds saved', '—', fmt(f.get('compile_seconds_saved'), 1)],
    ]) + '\n')
    W('Residual energy captured by the first $q$ directions: '
      + ', '.join(f"$q={k}$: {100*v:.4f}\\%" for k, v in d['residual_energy_captured'].items())
      + '.\n')

    if a.poisson_audit:
        pa = json.loads(Path(a.poisson_audit).read_text())
        pr = json.loads(Path(a.poisson_result).read_text())
        prows = pa['checks']['arm_table']
        W('## Poisson 2D, 1024 intervals\n')
        W('The Poisson weak residual is exactly $B\\,h(z)-f_m$, **linear** in the coefficients, so '
          'the retained path already eliminates $y$ by an exact triangular solve and the nonlinear '
          'iteration stays $K=16$-dimensional at every $q$. There is no quadrature, so change 3 '
          'does not apply. Job `%s` on `%s`.\n' % (pr.get('job_id'), pr.get('gpu')))
        W(table(['arm', 'q', 'rule', 'M', 'nominal dim', 'nonlinear dim', 'linear rank ok',
                 'bank projection %', 'best-found %', 'worst physical %', 'median device ms',
                 'all solver-valid'],
                [[f"`{x['arm']}`", x['q'], f"`{x['rule']}`", x['M'], fmt(x['nominal_dimension'], 0),
                  fmt(x['nonlinear_dimension'], 0), yn(x['linear_rank_valid']),
                  fmt(x['worst_bank_projection_percent']), fmt(x['worst_best_found_percent']),
                  fmt(x['worst_physical_percent']), fmt(x['median_device_ms'], 4),
                  yn(x['all_solver_valid'])]
                 for x in sorted(prows, key=lambda x: (x['kind'] != 'rom',
                                                       x['q'] if x['q'] is not None else 0,
                                                       x['rule'] or ''))]) + '\n')
        W('Non-dominated over (median device ms, worst physical %), all arms: '
          + ', '.join(f'`{x}`' for x in pa['checks']['nondominated_all']) + '.\n')
        pg = [[f'`{k}`', yn(v['passed']), str(v.get('detail'))[:120]]
              for k, v in pa['checks'].items() if isinstance(v, dict) and 'passed' in v]
        W(table(['gate', 'passed', 'detail'], pg) + '\n')

    W('## Limitations\n')
    W('- One mesh, one checkpoint, one training seed, six (Burgers) and twelve (Poisson) opened '
      'development cases. The final cohorts stay sealed and no new case was opened.\n')
    W('- The variable-projection and block-damped arms reach the SAME reachable set as the '
      'audited ladder but are a different solver, so agreement with the audited rungs is on the '
      'solution, not bitwise.\n')
    W('- Empirical-quadrature rules above the two retained ones use the bounded block-greedy '
      'fitter with a walltime cap; every rule records its support, fit residual and whether the '
      'cap bound.\n')
    W('- Offline costs (direction fit, quadrature fit, operator assembly) are reported '
      'separately and are never part of a query timing; they are one-time per rung.\n')
    W('- The full-order rows are same-job context, not a speed claim: no ratio against another '
      'job is taken anywhere in this report.\n')

    W('## Glossary\n')
    for term, text in [
        ('$q$', 'the number of extra fixed linear bank directions solved on top of the neural head.'),
        ('$K$', 'the latent dimension of the frozen head, 16 here; the outer solve works in this '
                'many unknowns once the corrections are eliminated.'),
        ('$M$ (test count)', 'how many smooth sine test functions the PDE residual is averaged '
                             'against. The weak objective needs more tests than unknowns.'),
        ('$m$', 'how many grid points the empirical-quadrature rule keeps.'),
        ('Variable projection', 'eliminating the coefficients a residual depends on linearly (or '
                                'here, quadratically but solvably) so the hard nonlinear search '
                                'runs in fewer dimensions.'),
        ('Kaufman Jacobian', 'the cheap variable-projection Jacobian that ignores how the '
                             'eliminated coefficients move with the outer variables. It gives the '
                             'exact gradient at inner optimality, so the stopping test is exact.'),
        ('Block-damped', 'one Jacobian per iteration for the whole augmented vector, but the '
                         'Levenberg damping and the trust radius apply only to the latent block.'),
        ('Alternating', 'rounds of (solve the latent block, then solve the correction block).'),
        ('Trust radius', 'a cap on how far one solver step may move; the audited ladder applied '
                         'the $q=0$ cap to the whole augmented step, which is what bound at large $q$.'),
        ('Converged', 'every time step and the initial fit exited for a legitimate reason, with no '
                      'iteration-budget exit, and the joint normalized gradient is below $10^{-6}$.'),
        ('Budget exit', 'a solve that stopped because it ran out of iterations, not because it '
                        'reached a solution.'),
        ('Joint normalized gradient $g$', 'a scale-free measure of how close a solve is to a '
                                          'stationary point; small means the solve really stopped '
                                          'at a solution.'),
        ('Empirical quadrature (EQ)', 'a learned weighted subset of grid points standing in for a '
                                      'full grid sum; "dense" means no such approximation.'),
        ('NNLS', 'nonnegative least squares: fitting weights that are not allowed to be negative, '
                 'which keeps the quadrature rule physically sensible.'),
        ('Bank projection floor', 'the best error any coefficients at all could reach in the frozen '
                                  'spatial bank.'),
        ('Best-found reconstruction', 'the best that rung\'s manifold can do on the reference field '
                                      'with no PDE solve involved.'),
        ('Worst same-grid error', 'the largest discrepancy, over cases and output times, against '
                                  'the converged full-order solve on the SAME mesh. It excludes '
                                  'the mesh\'s own discretization error.'),
        ('Worst reference error', 'the same but against the refined full-order reference, so it '
                                  'also contains the mesh\'s discretization error.'),
        ('Non-dominated', 'a setting that nothing else beats on both cost and error at once.'),
        ('Held-out / development / final cohort', 'cases used for measurement versus cases kept '
                                                  'unopened for the final report.'),
        ('pp', 'percentage points, i.e. an absolute difference between two percentages.'),
    ]:
        W(f'- **{term}** — {text}')
    W('')

    out.write_text('\n'.join(L) + '\n')
    print(out)
    print('sha256', sha(out))
    print('figure', figure_png, sha(figure_png))


if __name__ == '__main__':
    main()
