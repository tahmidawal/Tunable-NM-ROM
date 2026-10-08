"""jcp-mechanism: generate report.md and plots/*.png from the pulled job outputs (never hand-typed numbers).

Inputs (runs/<job>/code/output/...): a2q (A2), a1d3 (A1 3D, n65 / n129), a1d2 (A1 2D, L256 / L1024); and, read-only,
the 2026-10-01 lane outputs for the historical-reproducibility comparison (DESIGN amendment A1-5 / A2-4).
Every decision rule is the one pre-registered in DESIGN.md (amendments 1-3). Missing jobs are reported as missing.
Usage: /home/tahmid/Dev/.venv/bin/python make_report.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
PLOTS = HERE / 'plots'
WT = HERE.parents[2]
Q3 = WT / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d'
Q2 = WT / '2026-10-01-quadrature-study/experiments/quadrature-study'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
MARK = ['o', 's', 'D', '^', 'v', 'P', 'X']


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def e(x, d=1):
    return 'n/a' if x is None else f'{x:.{d}e}'


def pc(x, d=2):
    return 'n/a' if x is None else f'{100 * x:.{d}f} %'


def fit_slope(h, g):
    h, g = np.asarray(h, float), np.asarray(g, float)
    return float(np.polyfit(np.log(h), np.log(g), 1)[0])


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


# ====================================================================================================== A2
def a2_analysis(out):
    r = jload(RUNS / 'a2q/code/output/result.json')
    if r is None:
        return None, ['A2 job output missing.']
    z3 = np.load(RUNS / 'a2q/code/output/gaps_3d.npz')
    z2 = np.load(RUNS / 'a2q/code/output/gaps_2d.npz')
    cfg = r['config']
    res = dict(job=r, panels=[])
    for dim, z, groups, meshes, win, hof in (
            ('3D', z3, [f'R{x}' for x in cfg['Rps3d']], cfg['meshes3d'], cfg['fit_window3d'], lambda m: 1 / (m - 1)),
            ('2D', z2, cfg['settings2d'], cfg['meshes2d'], cfg['fit_window2d'], lambda m: 1 / m)):
        for g in groups:
            key = (lambda m, st: f'{g}_fixed_n{m}_{st}') if dim == '3D' else (lambda m, st: f'{g}_fixed_L{m}_{st}')
            chk = z[f'{g}_fixed_check_rho']
            ind = z[f'{g}_fixed_lat32768_rho'] if dim == '3D' else z[f'{g}_fixed_fib121393_rho']
            pan = dict(dim=dim, group=g, meshes=meshes, h=[hof(m) for m in meshes], window=win, check_worst=float(chk.max()),
                       independent_worst=float(ind.max()), independent_ok=bool(ind.max() <= 1e-2), stencils={})
            hw = [hof(m) for m in win]
            for st in ('upwind', 'central', 'nodes'):
                G = np.stack([z[key(m, st)] for m in meshes])                       # (meshes, states)
                Gw = np.stack([z[key(m, st)] for m in win])
                ok = np.all((Gw > 100 * chk[None]) & (Gw > 1e-12), axis=0)
                ent = dict(median=[float(np.median(x)) for x in G], worst=[float(x.max()) for x in G],
                           p10=[float(np.quantile(x, .1)) for x in G], p90=[float(np.quantile(x, .9)) for x in G],
                           screened=int(ok.sum()), states=int(G.shape[1]))
                if ok.sum() >= max(1, 0.25 * G.shape[1]) and len(win) >= 3:
                    Gs = Gw[:, ok]
                    ent['slope_median'] = fit_slope(hw, np.median(Gs, axis=1))
                    ent['slope_worst'] = fit_slope(hw, Gs.max(axis=1))
                    ent['slope_perstate_median'] = float(np.median([fit_slope(hw, Gs[:, i]) for i in range(Gs.shape[1])]))
                else:
                    ent['slope_median'] = ent['slope_worst'] = ent['slope_perstate_median'] = None
                bar = dict(upwind=(0.8, 1.2), central=(1.7, 2.3)).get(st)
                ent['bar'] = bar
                ent['consistent'] = (None if bar is None or ent['slope_median'] is None
                                     else bool(bar[0] <= ent['slope_median'] <= bar[1]))
                pan['stencils'][st] = ent
            # own-mesh states (context)
            own = {}
            for m in meshes:
                k_ = f'{g}_own{m}_n{m}_upwind' if dim == '3D' else f'{g}_own{m}_L{m}_upwind'
                if k_ in z.files:
                    own[m] = {st: float(np.median(z[k_.replace('_upwind', '_' + st)])) for st in ('upwind', 'central', 'nodes')}
                    own[m]['upwind_worst'] = float(z[k_].max())
            pan['own'] = own
            for k_ in z.files:                                  # independent-family check on every state set
                if k_.startswith(f'{g}_own') and (k_.endswith('_lat32768_rho') or k_.endswith('_fib121393_rho')):
                    pan.setdefault('own_independent_worst', {})[k_] = float(z[k_].max())
                    if z[k_].max() > 1e-2:
                        pan['independent_ok'] = False
            res['panels'].append(pan)
    # manufactured controls
    ctl = {}
    for dim, key, win, hof in (('3D', 'manufactured_3d', cfg['fit_window3d'], lambda m: 1 / (m - 1)),
                               ('2D', 'manufactured_2d', cfg['fit_window2d'], lambda m: 1 / m)):
        m_ = r.get(key)
        if not m_:
            continue
        hw = [hof(m) for m in win]
        gv = lambda st: [m_['meshes'][str(m)][st] for m in win]
        fin = str(win[-1])
        c = dict(slope_upwind=fit_slope(hw, gv('upwind')), slope_central=fit_slope(hw, gv('central')),
                 slope_nodes=fit_slope(hw, gv('nodes')), slope_neg_a=fit_slope(hw, gv('neg_a')),
                 slope_central101=fit_slope(hw, gv('central101')),
                 lead_up_rel=m_['lead_upwind_rel'], lead_ce_rel=m_['lead_central_rel'],
                 up_vs_lead=m_['meshes'][fin]['upwind_vs_lead'], ce_vs_lead=m_['meshes'][fin]['central_vs_lead'],
                 meshes=m_['meshes'])
        c['pos_pass'] = bool(abs(c['slope_upwind'] - 1) <= .15 and abs(c['slope_central'] - 2) <= .15 and
                             c['up_vs_lead'] <= .1 and c['ce_vs_lead'] <= .1 and c['lead_up_rel'] >= 1e-3 and
                             c['lead_ce_rel'] >= 1e-3)
        c['neg_a_pass'] = bool(abs(c['slope_neg_a']) < .05)
        ctl[dim] = c
    res['controls'] = ctl
    res['void'] = (not all(c['pos_pass'] and c['neg_a_pass'] for c in ctl.values()) or len(ctl) < 2
                   or not all(p['independent_ok'] for p in res['panels']))
    return res, []


def a2_plot(a2):
    fig, axs = plt.subplots(1, len(a2['panels']), figsize=(3.4 * len(a2['panels']), 3.6), sharey=False)
    for ax, p in zip(np.atleast_1d(axs), a2['panels']):
        h = np.array(p['h'])
        for i, st in enumerate(('upwind', 'central', 'nodes')):
            s = p['stencils'][st]
            ax.fill_between(h, s['p10'], s['p90'], color=SERIES[i], alpha=.12, lw=0)
            ax.plot(h, s['median'], color=SERIES[i], lw=2, marker=MARK[i], ms=6, mec='white', mew=1,
                    label=f"{st} (slope {s['slope_median']:.2f})" if s['slope_median'] is not None else f'{st} (unresolved)')
        c = a2['controls'].get(p['dim'])
        if c is not None:
            mh = sorted(c['meshes'], key=lambda k: -int(k))
            hh = [c['meshes'][k]['h'] for k in mh]
            ax.plot(hh, [c['meshes'][k]['upwind'] for k in mh], color=SERIES[0], lw=1, ls='--')
            ax.plot(hh, [c['meshes'][k]['central'] for k in mh], color=SERIES[1], lw=1, ls='--')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('mesh spacing h', color=INK2, fontsize=9)
        title = f"{p['dim']}, {p['group'].replace('R', 'R′=')}"
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        style(ax)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc='lower left')
    np.atleast_1d(axs)[0].set_ylabel('gap to continuum target (median over states)', color=INK2, fontsize=9)
    fig.text(0.01, 0.005, 'Bands: 10th–90th percentile over the fixed states. Dashed: manufactured control state.',
             color=INK2, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / 'a2_gap_vs_h.png', dpi=180, facecolor='#fcfcfb')
    plt.close(fig)


# ====================================================================================================== A1 3D
def bootstrap_low(f, seed=0, B=2000):
    rng = np.random.default_rng(seed)
    f = np.asarray(f)
    meds = [np.median(f[rng.integers(0, len(f), len(f))]) for _ in range(B)]
    return float(np.quantile(meds, .025))


def label(keys, s, dn, invalid, provisional, expected):
    """DESIGN amendment 2 (A2-1) decision order. keys: case ids; s, dn: per-case d(incumbent, conv), d(nodes, conv).
    `expected` is the registered case list: any missing case makes the result INCOMPLETE (amendment A2-6)."""
    missing = [k for k in expected if k not in keys]
    out = dict(missing_cases=missing, boot_low=None, why=invalid or None, provisional_solver=provisional)
    if missing:
        out.update(label='INCOMPLETE')
        return out
    if invalid:
        out.update(label='X')
        return out
    if any(v is None or not np.isfinite(v) for v in list(s) + list(dn)):
        out.update(label='X', why='non-finite distance')
        return out
    s, dn = np.asarray(s, float), np.asarray(dn, float)
    out.update(median_separation=float(np.median(s)),
               excluded_cases=[keys[i] for i in np.nonzero(s < 1e-3)[0]])
    if np.median(s) < 1e-2:
        out.update(label='X0', why='median separation below 1e-2')
        return out
    keep = s >= 1e-3
    f = 1 - dn[keep] / s[keep]
    med, lo, frac = float(np.median(f)), bootstrap_low(f), float(np.mean(f >= .75))
    out.update(median_f=med, frac_f_075=frac, min_f=float(f.min()))
    if med >= .9 and lo >= .8 and frac >= .9:
        out.update(label='R', boot_low=lo)
    elif med <= .5:
        out.update(label='N', boot_low=lo)
    else:
        out.update(label='X', boot_low=None, boot_low_info=lo)
    if out['label'] not in ('R', 'N'):
        out['provisional_solver'] = None
    return out


def a1d3_analysis():
    res = {}
    for n in (65, 129):
        r = jload(RUNS / f'a1d3/code/output/n{n}/result.json')
        if r is None:
            continue
        res[n] = r
    return res


def a1d3_labels(res):
    labs = {}
    for n, r in res.items():
        for Rp in r['config']['Rps']:
            A = r['arms']
            need = [f'tensor_R{Rp}', f'lat32768_R{Rp}', f'nodes_R{Rp}']
            if any(nm not in A or 'distance' not in A[nm] for nm in need):
                labs[(n, Rp)] = dict(label='INCOMPLETE', missing_cases=['arm output missing'], boot_low=None, why=None,
                                     provisional_solver=None)
                continue
            keys = [f'c{c["case"]}' for c in A[f'nodes_R{Rp}']['cases']]
            s = A[f'tensor_R{Rp}']['distance']['lat32768']['per_case']
            dn = A[f'nodes_R{Rp}']['distance']['lat32768']['per_case']
            invalid, prov, r0s = [], [], {}
            for nm in need:
                a = A[nm]
                if not a['all_finite'] or a['reason3_total'] > 0:
                    invalid.append(f'{nm}: non-finite or reason-3')
                r0 = sum(c['reasons']['0'] for c in a['cases'])
                steps = sum(sum(c['reasons'].values()) for c in a['cases'])
                r0s[nm.rsplit('_R', 1)[0]] = (r0, steps)
                if r0 > .01 * steps:
                    prov.append(f'{nm} reason-0 {r0}/{steps}')
            if not r['rho'][str(Rp)].get('continuum_target_valid', False):
                invalid.append('continuum target check failed on certification states')
            nr_ = (r.get('rho_nodes_reached') or {}).get(str(Rp))
            if nr_ is None or nr_.get('continuum_check_worst') is None or nr_['continuum_check_worst'] > 1e-6:
                invalid.append('continuum target check failed or missing on nodes-reached states')
            sens = (r.get('adaptive_sensitivity') or {}).get(str(Rp))
            if sens:
                ms = float(np.median([v for v in s if v is not None])) if any(v is not None for v in s) else 0.
                for k in ('nodes', 'lat32768'):
                    w_ = sens[k]['worst']
                    if w_ is None or w_ > .1 * ms or sens[k]['reason3'] > 0:
                        invalid.append(f'adaptive sensitivity of {k}: {w_} vs 0.1 x median separation {0.1 * ms:.2e}')
            else:
                prov.append('no sensitivity rerun at this mesh')
            labs[(n, Rp)] = label(keys, s, dn, '; '.join(invalid), '; '.join(prov) or None,
                                  [f'c{j}' for j in range(r['config']['cohort_count'])])
            labs[(n, Rp)]['reason0'] = r0s
    return labs


def a1d2_load():
    out = {}
    for L in (256, 1024):
        r = jload(RUNS / f'a1d2/code/output/L{L}/result.json')
        if r is not None:
            out[L] = r
    return out


def rows_of(r, s, arm):
    return [x for x in r['rows'] if x['setting'] == s and x['arm'] == arm]


def expected_2d(r):
    return [f'{coh}{c}' for coh in r['config']['cohorts'] for c in range(r['cohorts'][coh]['n'])
            if r['config'].get('case_subset', {}).get(coh) is None or c in r['config']['case_subset'][coh]]


def a1d2_labels(res):
    labs = {}
    for L, r in res.items():
        for s in r['config']['settings']:
            rows = {nm: {x['cohort'] + str(x['case']): x for x in rows_of(r, s, nm)} for nm in ('dense', 'gref', 'nodes')}
            keys = [k for k in expected_2d(r) if all(k in rows[nm] for nm in rows)]
            sep = [rows['dense'][k].get('vs_gref_restricted_evolved') for k in keys]
            dn = [rows['nodes'][k].get('vs_gref_restricted_evolved') for k in keys]
            invalid, prov, r0s = [], ['no sensitivity rerun (2D)'], {}
            for nm, rr in rows.items():
                r0, steps = 0, 0
                for x in rr.values():
                    if not x['finite'] or x['exits']['damping_limit'] > 0:
                        invalid.append(f'{nm} {x["cohort"]}{x["case"]} non-finite or damping exit')
                    r0 += x['exits']['budget']
                    steps += sum(x['exits'].values())
                r0s[nm] = (r0, steps)
                if r0 > .01 * steps:
                    prov.append(f'{nm} budget exits {r0}/{steps}')
            if not r.get('complete'):
                invalid.append('job not complete')
            if not r.get('targets_valid', False):
                invalid.append('continuum target check failed (targets_valid false)')
            labs[(L, s)] = label(keys, sep, dn, '; '.join(invalid), '; '.join(prov), expected_2d(r))
            labs[(L, s)]['reason0'] = r0s
            labs[(L, s)]['cases'] = len(keys)
    return labs


def a1_plot(r3, r2):
    panels = []
    for Rp in (512, 256):
        if r3:
            panels.append(('3D', Rp))
    for s in ('acc', 'fast'):
        if r2:
            panels.append(('2D', s))
    if not panels:
        return
    fig, axs = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 6.2))
    axs = np.array(axs).reshape(2, -1)
    for j, (dim, g) in enumerate(panels):
        if dim == '3D':
            sel = 'gl24' if g == 512 else 'lat4096'
            arms = [('tensor', 'tensor (backward diff.)'), (sel, f'{sel} (selected off-mesh)'),
                    ('lat32768', 'lat32768 (converged)'), ('nodes', 'nodes + analytic gradient')]
            xs = sorted(r3)
            xl = [f'{n - 1}³' for n in xs]
            err = {a: [r3[n]['arms'][f'{a}_R{g}'].get('worst_refined') for n in xs] for a, _ in arms}
            dist = {a: [r3[n]['arms'][f'{a}_R{g}']['distance'].get('lat32768', {}).get('worst') if a != 'lat32768' else None
                        for n in xs] for a, _ in arms}
            t1, t2 = f'3D, R′={g}', 'refined error (worst, PROVISIONAL)'
        else:
            sel = 'gauss96' if g == 'acc' else 'fib1597'
            arms = [('dense', 'dense (sign-upwind)'), ('lat64', 'lat64 (mesh lattice)'), (sel, f'{sel} (selected)'),
                    ('gref', 'Gauss 640² (converged)'), ('nodes', 'nodes + analytic gradient')]
            xs = sorted(r2)
            xl = [f'{L}²' for L in xs]
            err = {a: [max(x['ref_ST_evolved'] for x in rows_of(r2[L], g, a)) for L in xs] for a, _ in arms}
            dist = {a: [max(x['vs_gref_restricted_evolved'] for x in rows_of(r2[L], g, a)) if a != 'gref' else None
                        for L in xs] for a, _ in arms}
            t1, t2 = f'2D, {g}', 'ST error (worst, PROVISIONAL)'
        for i, (a, lab) in enumerate(arms):
            ax = axs[0, j]
            ax.plot(range(len(xs)), [100 * v for v in err[a]], color=SERIES[i], lw=2, marker=MARK[i], ms=7, mec='white',
                    mew=1, label=lab)
            if any(v is not None for v in dist[a]):
                axs[1, j].plot(range(len(xs)), [100 * v for v in dist[a]], color=SERIES[i], lw=2, marker=MARK[i], ms=7,
                               mec='white', mew=1, label=lab)
        for k, yl in ((0, t2), (1, 'distance to converged rollout (worst, %)')):
            ax = axs[k, j]
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xl)
            ax.set_xlim(-.3, len(xs) - .7)
            style(ax)
            if k == 1:
                ax.set_yscale('log')
            if j == 0:
                ax.set_ylabel(yl, color=INK2, fontsize=8)
        axs[0, j].set_title(t1, color=INK, fontsize=10, loc='left')
        axs[0, j].legend(fontsize=6.5, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / 'a1_error_vs_mesh.png', dpi=180, facecolor='#fcfcfb')
    plt.close(fig)


def cell_name(k):
    return f'3D {k[0] - 1}³ R′={k[1]}' if isinstance(k[1], int) else f'2D {k[0]}² {k[1]}'


def label_text(l):
    t = f"**{l['label']}**"
    if l.get('median_f') is not None:
        t += f" (median recovered fraction {l['median_f']:.3f}"
        t += f", bootstrap bound {l['boot_low']:.3f}" if l['boot_low'] is not None else ', bootstrap bound N/A'
        t += f", median separation {pc(l.get('median_separation'))}"
        if l.get('provisional_solver'):
            t += f"; provisional (solver): {l['provisional_solver']}"
        t += ')'
    if l.get('why'):
        t += f" [{l['why']}]"
    if l.get('missing_cases'):
        t += f" [missing: {', '.join(l['missing_cases'])}]"
    return t


# ====================================================================================================== historical
def g65_rows():
    """Bank rows on the 63^3 lattice x = k/64 by the vendor NumPy evaluation (common.features_np), cached in the
    lane's runs/ directory (not committed)."""
    cache = RUNS / 'G65_cache.npy'
    if cache.exists():
        return np.load(cache)
    sys.path.insert(0, str(HERE / 'vendor/quad3d/vendor/burgers3d-span'))
    import common as C3
    b = pickle.loads((HERE / 'vendor/quad3d/inputs/model_M2/bank.pkl').read_bytes())
    ax = np.arange(1, 64) / 64.0
    X = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    G = np.concatenate([C3.features_np(b['params'], X[i:i + 65536]) for i in range(0, len(X), 65536)]) @ np.asarray(b['rotation'])
    np.save(cache, G.T)
    return G.T


def g4_3d(r3):
    """Per arm, worst over cases of the evolved-time field distance on the 63^3 lattice (normalised by the refined
    reference's initial norm, as the errors) between this job's rollouts and the 2026-10-01 validation job's."""
    out = {}
    if not r3:
        return out
    G = g65_rows()
    ref = np.load(Q3 / 'runs/ref1/code/output/ref_923801.npz')
    n0 = {j: float(np.linalg.norm(ref[f'c{j}'][0])) for j in range(64)}
    for n in r3:
        old = np.load(Q3 / f'runs/val{n}/code/output/fields/coefficients.npz')
        for Rp in (512, 256):
            p = RUNS / f'a1d3/code/output/n{n}/fields/coefficients_R{Rp}.npz'
            if not p.exists():
                continue
            new = np.load(p)
            for arm in ('tensor', 'gl24', 'lat4096', 'lat32768'):
                d = []
                for j in range(64):
                    k = f'{arm}_R{Rp}__c{j}'
                    if k in new.files and k in old.files:
                        d.append(float((np.linalg.norm((new[k] - old[k]) @ G[:Rp], axis=1) / n0[j])[1:].max()))
                if d:
                    out[(n, Rp, arm)] = max(d)
    return out


def g4_2d(r2):
    out = {}
    for L, r in r2.items():
        old = jload(Q2 / f'runs/dv{L}/archive/output/result.json')
        if old is None:
            continue
        for s in r['config']['settings']:
            for arm in ('dense', 'lat64', 'gauss96' if s == 'acc' else 'fib1597', 'gref'):
                a = {(x['cohort'], x['case']): x['ref_ST_evolved'] for x in rows_of(r, s, arm)}
                b = {(x['cohort'], x['case']): x['ref_ST_evolved'] for x in rows_of(old, s, arm)}
                ks = sorted(set(a) & set(b))
                if ks:
                    out[(L, s, arm)] = max(abs(a[k] - b[k]) / b[k] for k in ks)
                fd = []
                for c in (0, 2):                     # field distance on the saved audit cases (257^2 shared nodes)
                    pn = RUNS / f'a1d2/code/output/L{L}/audit_{s}_{arm}_dev6{c}.npz'
                    po = Q2 / f'runs/dv{L}/archive/output/audit_{s}_{arm}_dev6{c}.npz'
                    if pn.exists() and po.exists():
                        fn, fo = np.load(pn)['f257'], np.load(po)['f257']
                        fd.append(float((np.linalg.norm((fn - fo).reshape(6, -1), axis=1) /
                                         np.linalg.norm(fo[0]))[1:].max()))
                if fd:
                    out[(L, s, arm + ' (field, audit cases)')] = max(fd)
    return out


# ====================================================================================================== report
def main():
    lines = []
    w = lines.append
    a2, notes2 = a2_analysis(None)
    r3 = a1d3_analysis()
    r2 = a1d2_load()
    lab3 = a1d3_labels(r3) if r3 else {}
    lab2 = a1d2_labels(r2) if r2 else {}
    if a2:
        a2_plot(a2)
    a1_plot(r3, r2)
    w('# Mechanism of the off-mesh gain: mesh nodes with the exact gradient (A1) and the stencil gap (A2)')
    w('')
    w('This report covers the two Group A experiments of the lane `jcp-mechanism`: **A1**, a reduced-solve arm that keeps '
      'the mesh nodes but uses the bank\'s analytic gradient (`nodes`), run beside the mesh incumbents and the off-mesh rules; '
      'and **A2**, the consistency gap of first-order upwind and second-order central mesh sums against the continuum '
      'advection as the mesh is refined. **Status: PRELIMINARY** (2026-10-08; validation/development cohorts only; every '
      'error against a refined reference is **PROVISIONAL** because the references are first-order solutions). Every number '
      'is generated by `experiments/jcp-mechanism/make_report.py` from the job outputs; the pre-registration is `DESIGN.md` '
      '(amendments 1–3). A3 and A4 are deferred (DESIGN §8).')
    w('')
    # ---------------------------------------------------------------- answers
    w('## Answers')
    w('')
    if lab3 or lab2:
        parts = [f"{cell_name(k)}: {label_text(l)}" for k, l in list(sorted(lab3.items())) + list(sorted(lab2.items()))]
        w('1. **A1 — does the exact gradient on the mesh nodes reproduce the off-mesh (continuum) solve?** Outcome labels '
          '(DESIGN amendment 2: R = `nodes` recovers the converged off-mesh rollout within the declared margins; '
          'N = it does not; X = intermediate or invalid; X0 = nothing to explain): ' + '; '.join(parts) + '.')
    else:
        w('1. **A1:** no job output yet.')
    if a2:
        parts = []
        for p in a2['panels']:
            s = p['stencils']
            fs = lambda x: 'unresolved' if x is None else f'{x:.2f}'
            parts.append(f"{p['dim']} {p['group']}: upwind {fs(s['upwind']['slope_median'])}, central "
                         f"{fs(s['central']['slope_median'])}, nodes {fs(s['nodes']['slope_median'])}")
        cons = all(p['stencils'][st]['consistent'] is True for p in a2['panels'] for st in ('upwind', 'central'))
        w(f"2. **A2 — stencil gap slopes** (median-state statistic over the pre-registered window): " + '; '.join(parts)
          + f". Upwind ≈ 1 and central ≈ 2 on every panel: **{'yes' if cons else 'no'}**"
          + (' (controls passed).' if not a2['void'] else ' — **controls failed: A2 verdicts void**.'))
    w('')
    # ---------------------------------------------------------------- A1 3D
    w('## 1. A1 in 3D (Burgers 3D, validation cohort 923801 × 64)')
    w('')
    if not r3:
        w('Job output missing.')
    for n, r in sorted(r3.items()):
        w(f"### {n - 1}³ (job {r.get('job_id')}, {r.get('gpu')}, commit `{(r.get('commit') or '')[:9]}`, "
          f"JAX {r.get('jax_version')}, backend {r.get('backend')}, precision {r.get('precision')})")
        w('')
        gates = r['gates']
        w('Gates: ' + ', '.join(f'{k} {v:.1e}' if isinstance(v, float) else f'{k} {v}' for k, v in gates.items()) + '.')
        w('')
        for Rp in r['config']['Rps']:
            if str(Rp) not in r['rho']:
                continue
            rho = r['rho'][str(Rp)]
            w(f"**R′ = {Rp}** (M = {rho['M']}). $\\rho$ on {rho['states_evolved']} evolved tensor-reached states of the "
              f"24 certification draws; continuum check (Gauss 64³ vs 80³) worst {e(rho['continuum_check']['worst'])}.")
            w('')
            w('| arm | m | ρ cont. worst (median) | ρ mesh worst | refined worst (median), PROVISIONAL | same-grid worst | '
              'dist. converged worst (median) | dist. tensor worst | LM its (median) | exits 0/3 |')
            w('|---|---|---|---|---|---|---|---|---|---|')
            names = [f'tensor_R{Rp}', f'gl24_R{Rp}', f'lat4096_R{Rp}', f'lat32768_R{Rp}', f'nodes_R{Rp}']
            for nm in names:
                a = r['arms'].get(nm)
                if a is None or 'worst_refined' not in a:
                    continue
                k = nm.rsplit('_R', 1)[0]
                rr = rho['rules'][k]
                dc = a['distance'].get('lat32768')
                dt = a['distance'].get('tensor')
                r0 = sum(c['reasons']['0'] for c in a['cases'])
                w(f"| `{k}` | {a.get('m', 'mesh')} | {e(rr['cont']['worst'])} ({e(rr['cont']['median'])}) | "
                  f"{e(rr['mesh']['worst'])} | {pc(a['worst_refined'])} ({pc(a['median_refined'])}) | "
                  f"{pc(a['worst_same_grid65'])} | {pc(dc['worst'], 3) + ' (' + pc(dc['median'], 3) + ')' if dc else '–'} | "
                  f"{pc(dt['worst'], 3) if dt else '–'} | {a['iterations_median']:.0f} | {r0}/{a['reason3_total']} |")
            du = rho['rules']['dense_upwind']
            w(f"| `dense` sign-upwind (target only) | mesh | {e(du['cont']['worst'])} ({e(du['cont']['median'])}) | 0 | – | – | – | – | – | – |")
            w('')
            nr = (r.get('rho_nodes_reached') or {}).get(str(Rp))
            if nr:
                w(f"$\\rho$ on the {nr['states']} `nodes`-reached states (first 8 cases; continuum check worst "
                  f"{e(nr['continuum_check_worst'])}): " + ', '.join(f"`{k}` {e(v['cont_worst'])}" for k, v in nr['rules'].items()) + '.')
                w('')
            sens = (r.get('adaptive_sensitivity') or {}).get(str(Rp))
            if sens:
                w('Adaptive-LM sensitivity (every step adaptive, first 8 cases), worst field distance from the fixed-sweep '
                  'rollout: ' + ', '.join(f"`{k}` {e(v['worst'])} (reason-3 {v['reason3']})" for k, v in sens.items()) + '.')
                w('')
    if len(r3) == 2:
        w('**Mesh invariance** (worst refined error, PROVISIONAL, 64³ / 128³ and their ratio max/min):')
        w('')
        w('| arm | R′ | 64³ | 128³ | ratio |')
        w('|---|---|---|---|---|')
        for Rp in (512, 256):
            for k in ('tensor', 'gl24', 'lat4096', 'lat32768', 'nodes'):
                v = [r3[n]['arms'].get(f'{k}_R{Rp}', {}).get('worst_refined') for n in (65, 129)]
                if None in v:
                    continue
                w(f'| `{k}` | {Rp} | {pc(v[0])} | {pc(v[1])} | {max(v) / min(v):.3f} |')
        w('')
    # ---------------------------------------------------------------- A1 2D
    w('## 2. A1 in 2D (Burgers 2D, dev6 ∪ val32, 38 cases)')
    w('')
    if not r2:
        w('Job output missing.')
    for L, r in sorted(r2.items()):
        w(f"### {L}² (job {r.get('job_id')}, {r.get('gpu')}, commit `{(r.get('commit') or '')[:9]}`)")
        w('')
        ng = {k: v for k, v in r['gates'].items() if k.startswith('nodes_')}
        w('Nodes gates: ' + '; '.join(f"{k}: " + ', '.join(f'{a} {b:.1e}' for a, b in v.items() if a != 'm') for k, v in ng.items())
          + '. Continuum target check: ' + ', '.join(
              f"{k.split('_')[-1]} {'pass' if v['passed'] else 'FAIL'}" for k, v in r['gates'].items()
              if k.startswith('continuum_target_converged')) + '.')
        w('')
        for s in r['config']['settings']:
            rh = r['rho'].get(s)
            w(f"**{s}** ($R'$ = {r['setup'][s]['R_prime']}, M = {r['setup'][s]['M']}); $\\rho$ on {rh['states'] if rh else 'n/a'} "
              'lat64-reached states.')
            w('')
            w('| arm | m | ρ cont. worst (median) | ρ mesh worst | ST worst (median), PROVISIONAL | S worst | '
              'dist. converged worst (median) | dist. dense worst | LM its (median) | budget / damping exits |')
            w('|---|---|---|---|---|---|---|---|---|---|')
            for spec in r['config']['arms'][s]:
                rows = rows_of(r, s, spec['name'])
                if not rows:
                    continue
                st = [x['ref_ST_evolved'] for x in rows]
                ss = [x['ref_S_evolved'] for x in rows]
                dg = [x.get('vs_gref_restricted_evolved') for x in rows if x.get('vs_gref_restricted_evolved') is not None]
                dd = [x.get('vs_dense_restricted_evolved') for x in rows if x.get('vs_dense_restricted_evolved') is not None]
                rr = rh['rules'].get(spec['name']) if rh else None
                w(f"| `{spec['name']}` | {rows[0]['m']} | " + (f"{e(rr['cont']['max'])} ({e(rr['cont']['median'])}) | {e(rr['mesh']['max'])}"
                                                            if rr else '– | –')
                  + f" | {pc(max(st))} ({pc(float(np.median(st)))}) | {pc(max(ss))} | "
                  + (f"{pc(max(dg), 3)} ({pc(float(np.median(dg)), 3)})" if dg else '–') + ' | '
                  + (pc(max(dd), 3) if dd else '–') + f" | {np.median([x['iterations_total'] for x in rows]):.0f} | "
                  f"{sum(x['exits']['budget'] for x in rows)} / {sum(x['exits']['damping_limit'] for x in rows)} |")
            w('')
            nr = (r.get('rho_nodes_reached') or {}).get(s)
            if nr:
                w(f"$\\rho$ on {nr['states']} `nodes`-reached states (first {nr['cases']} cases; check {e(nr['check_rho_max'])}): "
                  + ', '.join(f"`{k}` {e(v['cont_max'])}" for k, v in nr['rules'].items()) + '.')
                w('')
    # ---------------------------------------------------------------- labels
    w('## 3. A1 outcome labels (DESIGN amendment 2)')
    w('')
    w('| cell | label | median separation $s_j$ | median $f_j$ | bootstrap 2.5 % bound | fraction $f_j\\ge0.75$ | min $f_j$ | '
      'excluded cases ($s_j<10^{-3}$) | non-stationary steps (incumbent / converged / nodes) | provisional (solver) | invalid because |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for key, l in list(sorted(lab3.items())) + list(sorted(lab2.items())):
        f3 = lambda x: 'n/a' if x is None else f'{x:.3f}'
        r0 = l.get('reason0') or {}
        r0t = ' / '.join(f'{v[0]}/{v[1]}' for v in r0.values()) if r0 else 'n/a'
        w(f"| {cell_name(key)} | **{l['label']}** | {pc(l.get('median_separation'), 3)} | {f3(l.get('median_f'))} | "
          f"{'N/A' if l['boot_low'] is None else f3(l['boot_low'])} | {f3(l.get('frac_f_075'))} | {f3(l.get('min_f'))} | "
          f"{', '.join(l.get('excluded_cases') or []) or 'none'} | {r0t} | {l.get('provisional_solver') or 'no'} | "
          f"{l.get('why') or '–'}{'; missing: ' + ', '.join(l['missing_cases']) if l.get('missing_cases') else ''} |")
    w('')
    # ---------------------------------------------------------------- A2
    w('## 4. A2: the stencil gap against $h$')
    w('')
    if not a2:
        w('Job output missing.')
    else:
        j = a2['job']
        w(f"Job {j.get('job_id')} on {j.get('gpu')}, commit `{(j.get('commit') or '')[:9]}`. Gates: "
          + ', '.join(f'{k} {v:.1e}' for k, v in j['gates'].items()) + '.')
        w('')
        w(f"Fixed states (3D: {a2['panels'][0]['stencils']['upwind']['states']} per width from the 64³ validation tensor "
          f"rollouts; 2D: {a2['panels'][-1]['stencils']['upwind']['states']} per setting from the 1024² dev "
          'population), tests frozen. Gap $g=\\lVert N_h-N\\rVert/\\lVert N\\rVert$, median (worst) over states; slopes fitted on the '
          'pre-registered window on the screened population (DESIGN amendment 2, A2-3).')
        w('')
        for p in a2['panels']:
            w(f"**{p['dim']}, {p['group']}** — window {p['window']}; continuum check worst {e(p['check_worst'])}; "
              f"independent family worst $\\rho$ {e(p['independent_worst'])} "
              f"({'passes' if p['independent_ok'] else 'FAILS'} the empirical $10^{{-2}}$ check).")
            w('')
            hdr = ' | '.join(f"{('n=' if p['dim'] == '3D' else 'L=')}{m}" for m in p['meshes'])
            w(f'| stencil | {hdr} | slope (median state) | slope (worst state) | median per-state slope | screened | bar | consistent |')
            w('|---|' + '---|' * len(p['meshes']) + '---|---|---|---|---|---|')
            for st, s in p['stencils'].items():
                cells = ' | '.join(f'{e(a)} ({e(b)})' for a, b in zip(s['median'], s['worst']))
                sl = lambda x: 'unresolved' if x is None else f'{x:.2f}'
                w(f"| {st} | {cells} | {sl(s['slope_median'])} | {sl(s['slope_worst'])} | {sl(s['slope_perstate_median'])} | "
                  f"{s['screened']}/{s['states']} | {s['bar'] or '–'} | {'–' if s['consistent'] is None else ('yes' if s['consistent'] else 'no')} |")
            w('')
            if p['own']:
                w('Own-mesh states (context, not fitted), median gap upwind / central / nodes: ' + '; '.join(
                    f"{m}: {e(v['upwind'])} / {e(v['central'])} / {e(v['nodes'])}" for m, v in p['own'].items()) + '.')
                w('')
        w('**Controls** (manufactured state $u_\\star$, DESIGN amendment 1 A1-3 and amendment 2 A2-2):')
        w('')
        w('| dim | upwind slope | central slope | nodes slope | upwind vs leading term (finest) | central vs leading term (finest) | '
          'leading norms (up / central) | C-neg-a slope | central×1.01 slope | C-pos | C-neg-a |')
        w('|---|---|---|---|---|---|---|---|---|---|---|')
        for dim, c in a2['controls'].items():
            w(f"| {dim} | {c['slope_upwind']:.3f} | {c['slope_central']:.3f} | {c['slope_nodes']:.2f} | {c['up_vs_lead']:.3f} | "
              f"{c['ce_vs_lead']:.3f} | {e(c['lead_up_rel'])} / {e(c['lead_ce_rel'])} | {c['slope_neg_a']:.1e} | "
              f"{c['slope_central101']:.2f} | {'pass' if c['pos_pass'] else 'FAIL'} | {'pass' if c['neg_a_pass'] else 'FAIL'} |")
        w('')
    # ---------------------------------------------------------------- historical
    w('## 5. Historical reproducibility (report-only, DESIGN amendment A1-5)')
    w('')
    h3 = g4_3d(r3) if r3 else {}
    h2 = g4_2d(r2) if r2 else {}
    if h3:
        w('3D: worst over cases of the evolved-time field distance (63³ lattice) from the 2026-10-01 validation rollouts: ' +
          ', '.join(f'{n - 1}³ R′={Rp} `{a}` {e(v)}' for (n, Rp, a), v in sorted(h3.items())) + '.')
        for n, r in sorted(r3.items()):
            w(f"3D {n - 1}³ ρ reproduction (worst continuum ρ, max relative difference): " + ', '.join(
                f"R′={k.split('R')[-1]} {v:.1e}" for k, v in r['gates'].items() if k.startswith('G4a')) + '.')
    if h2:
        w('')
        w('2D: worst over cases of the relative difference of the ST error from the 2026-10-01 dev jobs, and the field '
          'distance on the two saved audit cases: ' +
          ', '.join(f'{L}² {s} `{a}` {e(v)}' for (L, s, a), v in sorted(h2.items())) + '.')
    w('')
    # ---------------------------------------------------------------- glossary
    w('## Glossary')
    w('')
    for t, d in GLOSSARY:
        w(f'- **{t}**: {d}')
    (HERE / 'report.md').write_text('\n'.join(lines) + '\n')
    json.dump(dict(labels3d={f'{k[0]}_{k[1]}': v for k, v in lab3.items()},
                   labels2d={f'{k[0]}_{k[1]}': v for k, v in lab2.items()},
                   a2=None if not a2 else dict(panels=a2['panels'], controls=a2['controls'], void=a2['void'])),
              open(HERE / 'report_numbers.json', 'w'), indent=1, default=str)
    print('wrote report.md', file=sys.stderr)


GLOSSARY = [
    ('A1 / `nodes` arm', 'the reduced solve whose advection is the equal-weight sum over the interior mesh nodes of the '
     'continuum integrand, with the solution and its gradient taken from the bank analytically (the mesh-node trapezoid '
     'rule); it differs from the full-order model\'s own mesh advection only in the derivative.'),
    ('A2 / stencil gap', 'the relative difference between a mesh sum of the tested advection (with a difference stencil) and '
     'the continuum tested advection, on the same state; it measures the stencil\'s consistency error.'),
    ('tensor (3D incumbent)', 'the precomputed quadratic form that evaluates the mesh advection with a fixed backward '
     'difference exactly.'),
    ('dense (2D incumbent)', 'the full-order model\'s sign-upwind advection on every mesh node.'),
    ('lat64', 'the deployed 2D mesh rule: the upwind stencil on a 63×63 sub-lattice of nodes, equal weights.'),
    ('selected off-mesh rule', 'the frozen rule chosen on validation by the source lanes: 3D Gauss 24³ (R′=512) and the '
     '4096-point lattice (R′=256); 2D Gauss 96² (acc) and Fibonacci 1597 (fast).'),
    ('converged off-mesh rollout', 'the reduced solve with a rule fine enough that further refinement does not change it: '
     '3D the 32768-point lattice, 2D Gauss 640².'),
    ('upwind / central', 'first-order sign-upwind differences (the full-order model\'s) and second-order central differences.'),
    ('ρ', 'relative error of a rule\'s tested advection vector against a target on a given state.'),
    ('continuum target / mesh target', 'the tested continuum advection by a very fine Gauss rule (3D 80³, 2D 640²) / the '
     'sign-upwind stencil on every mesh node.'),
    ('refined error (PROVISIONAL)', 'evolved-time maximum relative field error against a first-order full-order reference '
     'on a finer mesh (3D: 513 nodes per axis, on the 63³ lattice x = k/64; 2D: 8192², ST = with a 16× smaller time '
     'step, S = same time step), normalised by the initial field.'),
    ('same-grid', 'error against a tightly converged full-order solution on the same mesh.'),
    ('dist. converged / tensor / dense', 'evolved-time maximum field distance between two reduced rollouts of the same case, '
     'normalised by the initial field (3D on the 63³ lattice; 2D on the 257² shared nodes).'),
    ('separation $s_j$', 'the per-case distance between the mesh incumbent and the converged off-mesh rollout: the effect to explain.'),
    ('recovered fraction $f_j$', '$1 - d_j(\\text{nodes}, \\text{converged})/s_j$: 1 means `nodes` lands on the converged '
     'off-mesh rollout, 0 means it is as far from it as the incumbent.'),
    ('labels R / N / X / X0', 'R: `nodes` recovers the converged off-mesh rollout within the declared margins (median '
     '$f_j\\ge0.9$, bootstrap bound $\\ge0.8$, $f_j\\ge0.75$ on 90 % of cases); N: median $f_j\\le0.5$; X: intermediate or '
     'invalid; X0: median separation below 1 %, nothing to explain.'),
    ('bootstrap 2.5 % bound', 'the 2.5th percentile of the median $f_j$ over 2000 resamples of cases.'),
    ('provisional (solver)', 'a label not backed by a solver-sensitivity rerun at that mesh, or with more than 1 % '
     'non-stationary LM steps.'),
    ('exits 0/3, budget/damping', 'Levenberg–Marquardt step outcomes: 0 = not stationary at the end of the fixed sweep '
     '(3D) / iteration budget exhausted (2D); 3 = non-finite or damping exhausted.'),
    ('R′, M, m', 'number of bank columns in the solve; number of sine test functions; number of quadrature points.'),
    ('fixed states', 'reached coefficient states held fixed while only the mesh changes, so the gap depends on h alone.'),
    ('screened population', 'states whose gap exceeds 100× their own continuum-target check and $10^{-12}$ at every window mesh.'),
    ('slope (median state)', 'least-squares slope of log(median gap) against log h over the window; 1 = first order, 2 = second.'),
    ('manufactured state / C-pos / C-neg-a', 'a known smooth positive field used as a control: its slopes and its gap '
     'against the predicted leading error term must come out right (C-pos); an injected constant 1 % error must give slope 0 (C-neg-a).'),
    ('validation / dev cohorts', '3D: 64 validation cases (seed 923801) and 24 certification draws; 2D: dev6 and val32.'),
]

if __name__ == '__main__':
    main()
