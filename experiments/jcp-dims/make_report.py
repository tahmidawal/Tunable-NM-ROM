"""jcp-dims report generator (DESIGN 9): report.md + PNG plots + runs/EXPORT.json from checks/analysis-2d.json,
checks/analysis-3d.json, checks/smoke-accept-*.json, runs/GPU-LEDGER.json and the audit folder. No number is typed by hand.

    JAX_PLATFORMS=cpu python make_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']   # categorical, fixed order
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#fcfcfb'


def load(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return '—' if x is None else f'{100 * x:.3g} %'


def sci(x):
    return '—' if x is None else f'{x:.3g}'


def lab(d):
    return d.get('label') if isinstance(d, dict) else str(d)


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': .6, 'figure.facecolor': SURF,
                         'axes.facecolor': SURF, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


# ------------------------------------------------------------------------------------------------ 2D sections
def sec_2d(a2, L, plots):
    if a2 is None:
        L.append('## 2D results\n\nNot available (analysis-2d.json missing).\n')
        return {}
    exp = {}
    cfg = a2['configurations']
    L.append('## A4 (2D): smallest qualifying Gauss rule $m^\\*(r,M)$, LSPG-CN-R $\\Delta t_0$, $1024^2$\n')
    rows = []
    for lid, v in sorted(a2['mstar'].items()):
        if lid.startswith('F2|'):
            p = lid.split('|')
            rows.append([p[1][1:], p[2][1:], v['mstar']['outcome'], ' '.join(x[0].upper() if x != 'missing' else '·' for x in v['labels'])])
    rows.sort(key=lambda r: (int(r[0]), int(r[1])))
    L.append(table(['r', 'M', 'm* outcome', 'labels per candidate (Q qualified, N not, · missing)'], rows) + '\n')
    exp['mstar_2d'] = {lid: v['mstar'] for lid, v in a2['mstar'].items()}
    L.append('\n## B3 (2D): rule families\n')
    for cell, f in a2['families'].items():
        L.append(f'### {cell}\n')
        L.append(table(['family', 'family m*', 'per-draw (median / range / censored)'],
                       [[k, v['family']['outcome'], f"{v.get('median_draw', '—')} / {v.get('range', '—')} / {v.get('censored', '—')}"]
                        for k, v in sorted(f['mstar'].items())]) + '\n')
        L.append(table(['comparison', 'verdict'], sorted(f['point_efficiency'].items())) + '\n')
        g = a2['gates'][cell]
        L.append(table(['family | draw', 'cross-family gate', 'worst distance to gauss384'],
                       [[k, v.get('gate', v.get('fired')), sci(v.get('worst_dist_to_target'))] for k, v in sorted(g.items())]) + '\n')
    exp['families_2d'] = a2['families']
    L.append('\n## B6 (2D): test space $M/r$, M = r control, Galerkin vs LSPG (primary rung gauss256)\n')
    rows = []
    for k, v in sorted(a2['B6'].items()):
        if 'transitions' in v:
            rows.append([k, v['control'], ', '.join(f"{t}: {lab(x)}" for t, x in v['transitions'].items()), v.get('range_2_4')])
    L.append(table(['r | scheme', 'M = r control', 'transitions', 'range 2–4'], rows) + '\n')
    rows = [[k, nm, lab(x)] for k, v in sorted(a2['B6'].items()) if 'GAL-vs-LSPG' in k for nm, x in v.items()]
    L.append(table(['r | kappa', 'comparison', 'verdict'], rows) + '\n')
    exp['B6_2d'] = a2['B6']
    L.append('\n## A3 (2D): representation size at fixed $(M,m)$ and fixed $m$\n')
    for nm, v in a2['A3'].items():
        L.append(f"- **{nm}**: " + '; '.join(f"{t['X'].split('|')[0]} → {t['Y'].split('|')[0]}: {t['label']}" for t in v['transitions'])
                 + f" — range: **{v['range']}**\n")
    if a2.get('elasticity'):
        e = a2['elasticity']
        if e['status'] == 'ok':
            L.append(f"\nElasticities of the full-query cost (descriptive, {e.get('label', '')}): "
                     f"β_r = {e['beta'][1]:.3g} [{e['interval'][0][1]:.3g}, {e['interval'][1][1]:.3g}], "
                     f"β_M = {e['beta'][2]:.3g} [{e['interval'][0][2]:.3g}, {e['interval'][1][2]:.3g}], "
                     f"β_m = {e['beta'][3]:.3g} [{e['interval'][0][3]:.3g}, {e['interval'][1][3]:.3g}].\n")
        else:
            L.append(f"\nElasticity fit: {e['status']}.\n")
    if a2.get('shapley'):
        L.append('\n### Deployed-path decomposition (Shapley; selection-conditioned description, not a decomposition of the error norm)\n')
        rows = []
        for k, v in a2['shapley'].items():
            if 'acc' in v:
                rows.append([k, v['population'], ', '.join(pct(x) for x in v['acc']), ', '.join(v['acc_labels']),
                             ', '.join(sci(x) for x in v.get('cost', [])) or '—', ', '.join(v.get('cost_labels', [])) or '—'])
            else:
                rows.append([k, '—', v.get('outcome'), '', '', ''])
        L.append(table(['r_a → r_b', 'cases', 'Δ median error (r, M, m)', 'labels', 'Δ cost (r, M, m)', 'labels'], rows) + '\n')
    lm = a2.get('LMF', {})
    L.append('\n## B6-LMF: low mixed-frequency (hyperbolic-cross) test set at (384, 1536)\n')
    L.append(f"Stability checks: {lm.get('stability')}. " + (f"Outcome: {lm['outcome']}." if lm.get('outcome') else '') + '\n')
    for f in ('gauss', 'smolyak'):
        if f in lm:
            L.append(f"- {f}: LMF {lm[f]['lmf']['outcome']} vs default {lm[f]['default']['outcome']} → **{lm[f]['efficiency']}**\n")
    if lm.get('accuracy_gauss256'):
        L.append(f"- accuracy at gauss256: **{lm['accuracy_gauss256']['label']}**\n")
    exp['LMF'] = lm
    # plots
    plt = plots
    if plt is not None:
        Rs = (128, 256, 384, 512)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for i, r in enumerate(Rs):
            xs, ys = [], []
            for k, v in cfg.items():
                if v.get('rule') == 'gauss256' and v.get('form') == 'LSPG' and v.get('scheme') == 'CNR' and v['r'] == r and v.get('median') is not None:
                    xs.append(v['M'] / r)
                    ys.append(v['median'])
            o = np.argsort(xs)
            if xs:
                ax.plot(np.array(xs)[o], 100 * np.array(ys)[o], '-o', color=SERIES[i], lw=2, ms=4, label=f'r = {r}')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('test-space ratio M / r')
        ax.set_ylabel('median E0 error (%)')
        ax.set_title('2D, LSPG-CN-R, gauss256: error against M / r', color=INK, loc='left')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS / 'b6_2d_kappa.png', dpi=160)
        plt.close(fig)
    return exp


# ------------------------------------------------------------------------------------------------ 3D sections
def sec_3d(a3, L):
    if a3 is None:
        L.append('## 3D results\n\nNot available (analysis-3d.json missing).\n')
        return {}
    exp = {}
    L.append('## A4 (3D): front completion and per-width $m^\\*$\n')
    r = a3['rounds']
    rows = [[k, v['label'], sci(v.get('worst_dist_qp')), sci(v.get('rho_max')), v.get('round')] for k, v in sorted(r.items())]
    L.append('### Q2 rounds (qualified subset of the tested candidates; never a completeness claim)\n')
    L.append(table(['configuration', 'Q2-3D', 'worst q vs q+', 'ρ max', 'round'], rows) + '\n')
    L.append('\n### Width $m^\\*$ (intersection over fs-BE, fs-CN, qn-BDF2)\n')
    rows = [[k, v['width']['outcome']] + [v['per_spec'].get(s_, {}) and v['per_spec'][s_]['outcome'] for s_ in
                                          ('fs|BE|0.01|K1', 'fs|CN|0.01|K1', 'qn|BDF2|0.025|k4')] for k, v in sorted(a3['width_mstar'].items())]
    L.append(table(['width | family', 'width m*', 'fs-BE', 'fs-CN', 'qn-BDF2'], rows) + '\n')
    L.append('\n### Coverage: numerically qualified configurations per width\n')
    L.append(table(['width', 'n qualified', 'examples'], [[w, len(v), ', '.join(v[:3])] for w, v in a3['coverage'].items()]) + '\n')
    exp['coverage'] = {w: len(v) for w, v in a3['coverage'].items()}
    if a3.get('h2q'):
        L.append('\n## Qualified 3D re-timing (H2q; full-field $65^3$ mesh-path query, H200; not a comparison with the best FOM — K2 already fired)\n')
        rows = [[c, v.get('comparator'), v.get('comparator_q2'), v.get('winner'), sci(v.get('S')), v.get('verdict')] for c, v in a3['h2q'].items()]
        L.append(table(['comparator', 'configuration', 'its Q2', 'qualified winner', 'S', 'verdict'], rows) + '\n')
        exp['h2q'] = a3['h2q']
    L.append('\n## B6 (3D, W512) and A3 (3D)\n')
    for k, v in a3['B6'].items():
        if 'transitions' in v:
            L.append(f"- **{k}**: control {v['control']}; " + '; '.join(f"{t}: {lab(x)}" for t, x in v['transitions'].items())
                     + f"; range 2–4: {v.get('range_2_4')}\n")
        else:
            L.append(f"- **{k}**: " + '; '.join(f"{kk}: {lab(x)}" for kk, x in v.items()) + '\n')
    for k, v in a3['A3'].items():
        L.append(f"- **{k}**: " + '; '.join(lab(t) for t in v['transitions']) + f" — range **{v['range']}**\n")
    fam = a3.get('families', {})
    if fam:
        L.append('\n## B3 (3D, W512 κ = 4, fs-CN)\n')
        L.append(table(['family', 'family m*'], [[k, v['family']['outcome']] for k, v in sorted(fam['mstar'].items())]) + '\n')
        L.append(table(['comparison', 'verdict'], sorted(fam['point_efficiency'].items())) + '\n')
    exp['B6_3d'], exp['A3_3d'], exp['families_3d'] = a3['B6'], a3['A3'], fam
    return exp


def main():
    PLOTS.mkdir(exist_ok=True)
    a2, a3 = load('checks/analysis-2d.json'), load('checks/analysis-3d.json')
    led = load('runs/GPU-LEDGER.json')
    try:
        plt = setup_plot()
    except Exception:  # noqa: BLE001
        plt = None
    L = ['# jcp-dims: representation size, test space and quadrature — accuracy, cost, point counts and a quadrature-qualified 3D re-timing (viscous Burgers 2D/3D, off-mesh ROM)\n',
         '**Status:** generated by `make_report.py` from the lane\'s analysis files; every accuracy number is scored against E0 '
         '$R_{ST}$, a conditional Richardson estimate (E0 A10; combined A10, A11 and the retracted A15 carried). Numbers are '
         'provisional in that sense.\n',
         '## Jobs and budget\n']
    if led:
        L.append(table(['attempt', 'job id', 'state', 'GPU-h'], [[j['attempt'], j['job_id'], j['state'], sci(j.get('elapsed_hours'))]
                                                                for j in led['jobs']]) + '\n')
        L.append(f"Spent **{led['spent_gpu_hours']:.3g}** of {led['cap_gpu_hours']:.3g} GPU-h.\n")
    exp = dict(lane='jcp-dims', label='E0 R_ST: conditional Richardson estimate (E0 A10; combined A10, A11, retracted A15 carried)')
    exp.update(sec_2d(a2, L, plt))
    exp.update(sec_3d(a3, L))
    sm = {p: load(f'checks/smoke-accept-{p}.json') for p in ('smk2', 'smk3')}
    L.append('\n## Controls\n')
    for p, v in sm.items():
        if v:
            n = sum(1 for x in v['results'].values() if x['required'])
            L.append(f"- {p}: {n - len(v['failures'])} of {n} required controls behave as registered (accepted: {v['accepted']}).\n")
    L.append('\n## Glossary\n')
    for t, d in GLOSSARY:
        L.append(f'- **{t}** — {d}\n')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=str) + '\n')
    print('report written')


GLOSSARY = [
    ('r', 'bank width: the number of ordered bank coefficients the reduced model solves for.'),
    ('M', 'the number of sine test functions the residual is tested against; M/r is the test-space ratio.'),
    ('m', 'the number of quadrature points of the off-mesh rule used to integrate the nonlinear (advection) term.'),
    ('m*', 'the smallest point count on a pre-declared list at which a configuration is numerically qualified; "none" means no listed rule qualified.'),
    ('Q1', 'solver qualification: every run valid, at most 1 % of steps not certified, distance to a tight solve within 2.5e-4 on every case.'),
    ('Q2', 'quadrature qualification: the next finer rule changes the trajectory by at most 2.5e-4 on every case and the tested advection by at most 11.6 % at the reached states; the next-but-one rule must agree too.'),
    ('numerically qualified', 'Q1 and Q2 both pass.'),
    ('ladder top', 'the two largest rules of a ladder; they have no finer rules to check against and are never qualified.'),
    ('E0 / R_ST', 'the second-order space-time reference solutions the errors are measured against.'),
    ('ε_ref', 'the per-case uncertainty of the reference; two results differing by less than 2 ε_ref are a tie.'),
    ('transition', 'a paired per-case comparison of two cells differing in one axis: "improves", "does not improve" or "inconclusive".'),
    ('M = r control', 'the deliberately weak square test space; it must be worse than M = 4r, or test-space conclusions at that r are blocked.'),
    ('GAL / LSPG', 'Galerkin (residual projected onto the trial space) versus least-squares Petrov–Galerkin (residual minimised over M tests).'),
    ('LSPG-CN-R', 'Crank–Nicolson with two backward-Euler start-up steps, solved by LSPG.'),
    ('family', 'a kind of quadrature rule: Gauss tensor, rank-1 lattice, Sobol, Halton, Smolyak sparse grid, Monte Carlo.'),
    ('draw', 'one random realisation (shift or scramble seed) of a randomised family; five draws per family.'),
    ('cross-family gate', 'a family is flagged when its finest rule disagrees with an independent Gauss target (tested advection or rollout).'),
    ('bad-vector control', 'a deliberately degenerate lattice that the gate must flag.'),
    ('LMF', 'low mixed-frequency (hyperbolic-cross) test set: modes ordered by kx·ky.'),
    ('Shapley shares', 'the average over the six axis orderings of each axis\'s one-step change between two deployed configurations.'),
    ('elasticity β', 'the exponent of cost against r, M or m in a log-linear fit across the factorial (descriptive).'),
    ('H2q', 'speed-up of the cheapest quadrature-qualified 3D configuration that ties or beats a comparator, confirmed in a paired timing panel; pass if ≥ 1/0.6.'),
    ('C_old, C_W768, C_W1024', 'the deployed backward-Euler 3D configurations used as speed comparators.'),
    ('width m*', 'the smallest rule qualified for all three registered 3D solver specs of a bank width.'),
    ('coverage', 'whether a 3D bank width has at least one numerically qualified configuration.'),
]

if __name__ == '__main__':
    main()
