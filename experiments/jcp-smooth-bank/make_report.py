"""Generate report.md and plots/*.png for jcp-smooth-bank from the pulled run JSONs (never hand-typed numbers).

Inputs (committed): results/<train job>/<arm>/train.json and results/<eval job>/{eval,timing}/.
Decision rules: DESIGN.md A1.5 as amended by A2.4 and A4 (implemented in `verdicts`).

    python make_report.py --train tr1 --eval ev1 [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS = ('acc', 'fast')
BARS = (0.116, 0.06, 0.01)
GAUSS = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256)
FIB = (987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368)
MMAX = dict(G=256 ** 2, F=46368)
LABEL = dict(frozen_deployed='frozen bank, deployed rotation', frozen_lane='frozen bank, lane rotation',
             base='base (retrain, σ=4)', sob01='sob01 (σ=4, λ=0.1)', sig2='sig2 (σ=2)', sig1='sig1 (σ=1)',
             sob001='sob001 (σ=4, λ=0.01)', sob1='sob1 (σ=4, λ=1)', coarse='coarse (129-node data)',
             base_s1='base_s1 (seed 1)', comb='comb')
ORDER = ['frozen_deployed', 'frozen_lane', 'base', 'base_ev2r', 'base_s1', 'sob001', 'sob01', 'sob1', 'sig2', 'sig1', 'coarse', 'comb']


def fmt(x, p=2):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    if x == 0:
        return '0'
    return f'{x:.{p}e}' if (abs(x) < 1e-2 or abs(x) >= 1e4) else f'{x:.{p + 1}g}'


def pct(x):
    return '—' if x is None else f'{100 * x:.3f}'


def mfmt(m, fam):
    return f'>{MMAX[fam]}' if m is None else str(int(m))


def load(train, ev):
    """train / ev: comma-separated job names. A label evaluated again in a later job is kept as <label>_<job>."""
    T = {}
    for tj in train.split(','):
        for p in sorted((HERE / 'results' / tj).glob('*/train.json')):
            T[p.parent.name] = json.loads(p.read_text())
    E, TMS, meta = {}, {}, []
    for ej in ev.split(','):
        for p in sorted((HERE / 'results' / ej / 'eval').glob('eval_*.json')):
            d = json.loads(p.read_text())
            lab = d['label'] if d['label'] not in E else f"{d['label']}_{ej}"
            d['label'], d['eval_job'] = lab, ej
            E[lab] = d
        tp = HERE / 'results' / ej / 'timing/timing.json'
        if tp.exists():
            TMS[ej] = json.loads(tp.read_text())
        meta += [dict(json.loads(p.read_text()), eval_job=ej) for p in sorted((HERE / 'results' / ej / 'eval').glob('meta_task*.json'))]
    TM = TMS.get(ev.split(',')[0])
    TM_extra = {k: v for k, v in TMS.items() if k != ev.split(',')[0]}
    return T, E, TM, meta, TM_extra


# --------------------------------------------------------------------- metrics ----

def rows(E, b, s, arm):
    return [r for r in E[b]['settings'][s]['rows'] if r['arm'] == arm]


def worst(E, b, s, arm, key):
    rs = rows(E, b, s, arm)
    v = [r[key] for r in rs if r.get(key) is not None]
    return max(v) if v and len(v) == len(rs) else None


def metrics(E, b, s):
    S = E[b]['settings'][s]
    P = S['projection']['S']
    g = rows(E, b, s, 'gref')
    out = dict(
        E_S=worst(E, b, s, 'gref', 'ref_S_evolved'), E_ST=worst(E, b, s, 'gref', 'ref_ST_evolved'),
        e_val_med=P['value']['median'], e_val_max=P['value']['max'],
        e_grad_med=P['grad_D2']['median'], e_grad4_med=P['grad_D4']['median'], e_grad_max=P['grad_D2']['max'],
        e_grad2c_med=P['grad_D2_common']['median'],
        fd_unc_med=P['fd_target_uncertainty']['median'],
        mstar={k: v for k, v in S['mstar'].items()}, tail=S.get('tail', {}), C4=S.get('C4', {}),
        spectra=S.get('spectra', {}).get('aggregate', {}), trust=S['trust'], A_sv=S.get('A_singular_values'),
        cases=len(g))
    # C6 (A4): every required rollout finite with <= 2 non-accepted exits; tight-solver sensitivity on dev6
    bad = [f"{r['arm']}:{r['cohort']}{r['case']}" for r in S['rows']
           if r['arm'] not in ('gauss8',) and (not r['finite'] or r.get('nonaccepted_exits', 0) > 2)]
    tight = [r.get('vs_gref_evolved') for r in S['rows'] if r['arm'] == 'gref_tight']
    out['C6_failures'] = bad
    failed_cases = {(r['cohort'], r['case']) for r in S['rows'] if r['arm'] != 'gauss8' and
                    (not r['finite'] or r.get('nonaccepted_exits', 0) > 2)}       # union over every arm
    keep = [r for r in g if (r['cohort'], r['case']) not in failed_cases]
    out['E_S_without_C6_failed'] = max(r['ref_S_evolved'] for r in keep) if keep else None
    out['E_ST_without_C6_failed'] = max(r['ref_ST_evolved'] for r in keep) if keep else None
    out['C6_failed_cases'] = sorted(f'{a}{b}' for a, b in failed_cases)
    out['target_norm_ok'] = all(v['zero_or_nonfinite'] == 0 for v in S.get('target_norm', {}).values()) and bool(S.get('target_norm'))
    out['tight_max'] = max(tight) if tight else None
    c4 = S.get('C4', {})
    out['C4_pass'] = bool(c4) and all(v['check_rho_max'] <= 1e-5 and v['flux_rho_max'] <= 1e-5 for v in c4.values())
    g768 = [r.get('vs_gref_evolved') for r in S['rows'] if r['arm'] == 'gauss768']
    out['g768_max'] = max(g768) if g768 else None
    return out


N_CASES, N_STATES, N_COMMON = 38, 38 * 50, 38 * 6


def complete_inputs(E, b, s):
    """Completeness (A4 eligibility): evaluation complete, every case of every required arm, full populations."""
    why = []
    if not E[b].get('complete'):
        why.append('evaluation incomplete')
    S = E[b]['settings'].get(s)
    if not S:
        return why + ['setting missing']
    allc = {(c, i) for c, n in (('dev6', 6), ('val32', 32)) for i in range(n)}
    dev6 = {('dev6', i) for i in range(6)}
    req = ['gref', 'lat64', 'gauss32', 'gauss48', 'gauss64', 'gauss96', 'fib1597', 'fib4181', 'fib6765']
    for bb in (0.116, 0.06):                                     # P4 rollouts of the selected rules
        ms = S['mstar'].get(f'gref|G|{bb}')
        if ms is not None:
            req.append(f'gauss{int(round(math.sqrt(ms)))}')
    for arm in req:
        got = [(r['cohort'], r['case']) for r in rows(E, b, s, arm)]
        if len(got) != len(set(got)) or set(got) != allc:
            why.append(f'{arm} cases')
    for arm in ('gauss768', 'gref_tight'):
        got = [(r['cohort'], r['case']) for r in rows(E, b, s, arm)]
        if len(got) != len(set(got)) or set(got) != dev6:
            why.append(f'{arm} dev6 cases')
    for r in S['rows']:
        if not all(math.isfinite(r.get(k) if r.get(k) is not None else float('nan')) for k in ('ref_S_evolved', 'ref_ST_evolved')):
            why.append(f"non-finite error {r['arm']}"); break
    tn = S.get('target_norm', {})
    for p_, n_ in (('gref', N_STATES), ('lat64', N_STATES), ('common', N_COMMON)):
        if tn.get(p_, {}).get('states') != n_ or tn.get(p_, {}).get('zero_or_nonfinite_320', 1) != 0:
            why.append(f'{p_} population')
    return why


def eligible(E, T, b, s, M):
    m = M[b][s]
    why = complete_inputs(E, b, s)
    if not m['C4_pass']:
        why.append('C4')
    if m['g768_max'] is None or m['g768_max'] > 1e-6:
        why.append('gref-vs-768 rollout')
    if m['C6_failures']:
        why.append(f"C6 ({len(m['C6_failures'])} rollouts)")
    if m['tight_max'] is None or m['tight_max'] > 1e-3:
        why.append('solver-sensitive')
    for k in ('E_S', 'E_ST', 'e_val_med', 'e_grad_med', 'e_grad4_med'):
        if m[k] is None or not math.isfinite(m[k]) or m[k] <= 0:
            why.append(f'{k} missing')
    if not m['target_norm_ok']:
        why.append('zero/non-finite target norm')
    if b != 'frozen_deployed':
        tr = T['base'].get('rotation_frozen') if b == 'frozen_lane' and 'base' in T else T.get(b, {}).get('rotation')
        if not tr:
            why.append('rotation diagnostics missing')
        elif tr['R_G_condition_number'] > 1e12 or tr['rotated_bank_orthonormality_deviation_at_train_mesh'] > 1e-6:
            why.append('rotation conditioning')
    if not E[b]['C5a']['passed']:
        why.append('C5a')
    return why


def interp_m(E, b, s, bar, pop='gref'):
    """Descriptive (A1.4): log-linear interpolation of m between the confirmed Gauss rung m* and the rung below it."""
    R = E[b]['settings'][s]['rho']
    ms = E[b]['settings'][s]['mstar'].get(f'{pop}|G|{bar}')
    if ms is None:
        return None
    lad = [p * p for p in GAUSS]
    j = lad.index(ms)
    if j == 0:
        return float(ms)
    r0, r1 = R[f'gauss{GAUSS[j - 1]}'][pop]['max'], R[f'gauss{GAUSS[j]}'][pop]['max']
    if not (r0 > bar >= r1 > 0):
        return float(ms)
    t = (math.log(r0) - math.log(bar)) / (math.log(r0) - math.log(r1))
    return float(math.exp(math.log(lad[j - 1]) + t * (math.log(lad[j]) - math.log(lad[j - 1]))))


def red(m_base, m_t, fam):
    """A2.4 reduction factor of m*."""
    if m_t is None:
        return 1.0 if m_base is None else 0.0
    return (MMAX[fam] if m_base is None else m_base) / m_t


def verdicts(E, T, M):
    out = {}
    base = 'base'
    ref = 'frozen_lane'
    if base not in M:
        return out
    noise = {}
    for s in SETTINGS:
        for k in ('E_S', 'E_ST', 'e_val_med', 'e_grad_med', 'e_grad2c_med', 'e_grad4_med'):
            vals = [abs(M[base][s][k] / M[c_][s][k] - 1) for c_ in (ref, 'base_s1')
                    if c_ in M and M[c_][s][k] and M[base][s][k]]
            noise[(s, k)] = max(vals) if vals else float('inf')     # no comparator -> nothing is resolved
    out['noise'] = {f'{s}|{k}': v for (s, k), v in noise.items()}
    resolved = lambda s, k, d: d is not None and abs(d) > 2 * noise.get((s, k), float('inf'))
    dlt = lambda t, s, k: (M[t][s][k] / M[base][s][k] - 1) if (M[t][s][k] and M[base][s][k]) else None
    for t in M:
        if t in (base, ref, 'frozen_deployed', 'base_s1', 'base_ev2r', 'coarse'):
            continue
        v = dict(eligible={s: eligible(E, T, t, s, M) for s in SETTINGS})
        base_ok = {s: not eligible(E, T, base, s, M) for s in SETTINGS}
        ok = {s: base_ok[s] and not v['eligible'][s] for s in SETTINGS}
        d = {s: {k: dlt(t, s, k) for k in ('E_S', 'E_ST', 'e_val_med', 'e_grad_med', 'e_grad2c_med', 'e_grad4_med')} for s in SETTINGS}
        v['delta'] = d
        # H1 (A1.5 + A4)
        h1g = all(ok[s] and d[s]['e_grad2c_med'] is not None and d[s]['e_grad4_med'] is not None
                  and d[s]['e_grad2c_med'] <= -.2 and d[s]['e_grad4_med'] <= -.2
                  and resolved(s, 'e_grad2c_med', d[s]['e_grad2c_med']) and resolved(s, 'e_grad4_med', d[s]['e_grad4_med'])
                  for s in SETTINGS)
        h1v = all(ok[s] and d[s]['e_val_med'] <= .10 for s in SETTINGS)
        h1e = any(ok[s] and d[s]['E_S'] <= -.05 and resolved(s, 'E_S', d[s]['E_S']) and
                  all(d[o]['E_S'] <= .02 for o in SETTINGS if o != s) for s in SETTINGS)
        v['H1'] = dict(grad=h1g, value=h1v, rollout=h1e, passed=bool(h1g and h1v and h1e))
        # H2 descriptive (A1.5 + A2.4)
        h2 = {}
        for bb in (0.06, 0.01):
            for s in SETTINGS:
                mb, mt = M[base][s]['mstar'][f'gref|G|{bb}'], M[t][s]['mstar'][f'gref|G|{bb}']
                cb, ct = M[base][s]['mstar'][f'common|G|{bb}'], M[t][s]['mstar'][f'common|G|{bb}']
                mref = M[ref][s]['mstar'][f'gref|G|{bb}'] if ref in M else 'na'
                rf = red(mb, mt, 'G')
                other = [o for o in SETTINGS if o != s][0]
                rf_o = red(M[base][other]['mstar'][f'gref|G|{bb}'], M[t][other]['mstar'][f'gref|G|{bb}'], 'G')
                def no_worse(a_, b_):           # a_ (treatment) <= b_ (base), censored (None) larger than any finite
                    return (a_ is not None and (b_ is None or a_ <= b_)) or (a_ is None and b_ is None)
                same_dir = no_worse(ct, cb)
                cbo, cto = M[base][other]['mstar'][f'common|G|{bb}'], M[t][other]['mstar'][f'common|G|{bb}']
                mrefo = M[ref][other]['mstar'][f'gref|G|{bb}'] if ref in M else 'na'
                mbo = M[base][other]['mstar'][f'gref|G|{bb}']
                res_ = (mref == mb) and (mrefo == mbo) and all(
                    M[x][y]['mstar'][f'gref|G|{bb}'] == M[base][y]['mstar'][f'gref|G|{bb}'] for x in ('base_s1',) if x in M for y in SETTINGS)
                h2[f'{s}|{bb}'] = dict(reduction=rf, other_reduction=rf_o, common_same_direction=same_dir,
                                       common_same_direction_other=no_worse(cto, cbo), resolved=res_,
                                       passed=bool(ok[s] and ok[other] and rf >= 1.5 and rf_o >= 1.0 and same_dir
                                                   and no_worse(cto, cbo) and res_))
        v['H2'] = h2
        v['H2_passed'] = any(x['passed'] for x in h2.values())
        v['useful_winner'] = bool((v['H1']['passed'] or v['H2_passed']) and all(
            ok[s] and M[t][s]['E_S'] <= 1.1 * M[base][s]['E_S'] and M[t][s]['E_ST'] <= 1.1 * M[base][s]['E_ST']
            for s in SETTINGS))
        score = 0.
        for s in SETTINGS:
            mb, mt = M[base][s]['mstar']['gref|G|0.06'], M[t][s]['mstar']['gref|G|0.06']
            score += math.log2(((MMAX['G'] if mb is None else mb)) / (2 * MMAX['G'] if mt is None else mt))
        v['rank_score'] = score
        v['mean_E_S'] = float(np.mean([M[t][x]['E_S'] for x in SETTINGS])) if all(M[t][x]['E_S'] for x in SETTINGS) else None
        out[t] = v
    passers = [t for t in out if t not in ('noise',) and out[t]['H2_passed'] and out[t]['useful_winner']]
    out['ranking_useful_H2'] = sorted(passers, key=lambda t: (-out[t]['rank_score'], out[t]['mean_E_S'] or float('inf')))
    out['noise'] = {f'{s}|{k}': v for (s, k), v in noise.items()}
    return out


R1_SRC = json.loads((HERE / 'results/r1_targets_from_2d_lane.json').read_text())   # extracted from the 2D lane's dv1024 result.json (sha256 recorded)
R1_PINNED = R1_SRC['targets']


def r1_gate(E):
    """R1 (A1.6): frozen bank + deployed rotation reproduces the 2D lane's dv1024 numbers (2 s.f. for rho, 0.01 pp)."""
    if 'frozen_deployed' not in E:
        return dict(passed=False, reason='frozen_deployed missing')
    res = {}
    for s in SETTINGS:
        S = E['frozen_deployed']['settings'][s]
        g = S['rho']['gauss64']['lat64']['max']
        gs = worst(E, 'frozen_deployed', s, 'gref', 'ref_ST_evolved')
        ls = worst(E, 'frozen_deployed', s, 'lat64', 'ref_ST_evolved')
        p = R1_PINNED[s]
        res[s] = dict(g64_lat64=g, g64_ok=f'{g:.1e}' == f"{p['g64_lat64']:.1e}", gref_ST=gs, gref_ok=abs(gs - p['gref_ST']) <= 1e-4,
                      lat64_ST=ls, lat64_ok=abs(ls - p['lat64_ST']) <= 1e-4)
    return dict(per_setting=res, passed=all(v['g64_ok'] and v['gref_ok'] and v['lat64_ok'] for v in res.values()))


# ----------------------------------------------------------------------- plots ----

CAPTION = 'dev6 ∪ val32 (38 cases), one training seed per bank, first-order references: PROVISIONAL'


def plots(E, M, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    banks = [b for b in ORDER if b in M]
    cols = {b: c for b, c in zip(banks, plt.rcParams['axes.prop_cycle'].by_key()['color'] * 3)}
    # frontier: one marker per bank and bar (confirmed Gauss rungs), floor on the y axis
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for j, s in enumerate(SETTINGS):
        for i, (key, ylab) in enumerate((('e_val_max', 'projection floor, worst state (vs S), %'),
                                        ('E_S', 'converged-quadrature rollout error, worst case (vs S), %'))):
            for b in banks:
                y = 100 * M[b][s][key]
                for bb, mk in ((0.06, 's'), (0.01, 'o')):
                    m = M[b][s]['mstar'][f'gref|G|{bb}']
                    if m is None:
                        continue
                    ax[i, j].scatter(m, y, color=cols[b], marker=mk, s=60, alpha=.85,
                                     label=(b if (bb == 0.06 and i == 0 and j == 0) else None))
            groups = {}
            for b in banks:                              # name every bank at its (possibly shared) square marker
                m = M[b][s]['mstar']['gref|G|0.06']
                if m is not None:
                    groups.setdefault((m, round(100 * M[b][s][key], 2)), []).append(b)
            for (m, y), names in groups.items():
                ax[i, j].annotate(', '.join(names), (m, y), fontsize=6, xytext=(5, -2), textcoords='offset points', va='top')
            ax[i, j].set_xscale('log')
            ax[i, j].set_xlabel('Gauss points needed m* (confirmed rung): square ρ≤0.06, circle ρ≤0.01', fontsize=9)
            ax[i, j].set_ylabel(ylab, fontsize=9)
            ax[i, j].set_title(f'{s} (y axis zoomed, not from zero)', fontsize=10)
    ax[0, 0].legend(fontsize=8, title='bank')
    fig.suptitle('Error floor vs quadrature points needed, one marker per bank and bar\n' + CAPTION, fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'frontier.png', dpi=140); plt.close(fig)
    # derivatives
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, s in enumerate(SETTINGS):
        x = np.arange(len(banks))
        for k, (key, mk, lab) in enumerate((('e_val_med', 'o', 'projection floor (median)'),
                                           ('e_grad2c_med', 's', 'gradient error vs D2, common support (median)'),
                                           ('e_grad4_med', '^', 'gradient error vs D4, common support (median)'))):
            ax[j].scatter(x + (k - 1) * .15, [M[b][s][key] for b in banks], marker=mk, s=55, label=lab)
        ax[j].set_yscale('log'); ax[j].set_xticks(x); ax[j].set_xticklabels(banks, rotation=30, fontsize=8)
        ax[j].set_ylabel('relative error (dimensionless)'); ax[j].set_title(f'{s}: S reference states on the 257² nodes', fontsize=10)
    ax[0].legend(fontsize=8)
    fig.suptitle('Representation and mesh-derivative errors of each bank\n' + CAPTION, fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'derivatives.png', dpi=140); plt.close(fig)
    # ladders
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for j, s in enumerate(SETTINGS):
        for b in banks:
            R = E[b]['settings'][s]['rho']
            ax[j].plot([p * p for p in GAUSS], [R[f'gauss{p}']['gref']['max'] for p in GAUSS], '-o', ms=3, color=cols[b], label=b)
            ax[j].plot(FIB, [R[f'fib{n}']['gref']['max'] for n in FIB], '--', color=cols[b], alpha=.5)
        for bb in BARS:
            ax[j].axhline(bb, color='k', lw=.6, ls=':')
            ax[j].text(70, bb * 1.15, f'bar {bb}', fontsize=7)
        ax[j].set_xscale('log'); ax[j].set_yscale('log')
        ax[j].set_title(f'{s}: worst ρ over own reached states', fontsize=10)
        ax[j].set_xlabel('points m (solid: tensor Gauss; dashed: Fibonacci lattice)'); ax[j].set_ylabel('worst ρ (dimensionless)')
    ax[0].legend(fontsize=8, title='bank (solid Gauss)')
    fig.suptitle('Quadrature-error ladders\n' + CAPTION, fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'ladders.png', dpi=140); plt.close(fig)
    # spectra
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, s in enumerate(SETTINGS):
        for b in banks:
            env = M[b][s]['spectra'].get('f|envelope_median')
            if env:
                ax[j].semilogy(env, color=cols[b], label=b)
        ax[j].set_title(f'{s}: median Chebyshev envelope of $f=u\\,(u_x+u_y)$', fontsize=10)
        ax[j].set_xlabel('Chebyshev degree j (512 points per axis)'); ax[j].set_ylabel('normalised envelope $E_j$')
        ax[j].set_ylim(1e-10, 2)
    ax[0].legend(fontsize=8)
    fig.suptitle('Spectra of the advection integrand on 64 own reached states (every 1e-8 bandwidth unresolved)\n' + CAPTION, fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'spectra.png', dpi=140); plt.close(fig)


def e2e_plot(E, M, TM, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    banks = [b for b in ORDER if b in M]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    w = .38
    for j, s in enumerate(SETTINGS):
        x = np.arange(len(banks))
        ax[j].bar(x - w / 2, [100 * M[b][s]['E_ST'] for b in banks], w, label='vs ST (space+time refined reference)')
        ax[j].bar(x + w / 2, [100 * M[b][s]['E_S'] for b in banks], w, label='vs S (space-only refined reference)')
        ax[j].set_xticks(x); ax[j].set_xticklabels(banks, rotation=30, fontsize=8)
        ax[j].set_ylabel('worst evolved error over 38 cases, %'); ax[j].set_title(f'{s}: converged-quadrature (Gauss 640²) rollout', fontsize=10)
    ax[0].legend(fontsize=8)
    if TM and TM.get('median_ms'):
        lab, val = [], []
        for k, v in TM['median_ms'].items():
            lab.append(k.replace('|', ' ')); val.append(v)
        ax[2].barh(np.arange(len(val)), val); ax[2].set_yticks(np.arange(len(val))); ax[2].set_yticklabels(lab, fontsize=8)
        ax[2].set_xlabel('median ms per query (rollout + decode), paired A–B–A')
        ax[2].set_title(f"cost at each bank's own m* rules ({TM['gpu']})\\ntiming: dev6 only, 3 repetitions in both orders; not matched to the accuracy panels", fontsize=9)
    fig.suptitle('End-to-end accuracy and cost\n' + CAPTION, fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'e2e.png', dpi=140); plt.close(fig)


# ---------------------------------------------------------------------- report ----

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default='tr1b', help='comma-separated training jobs')
    ap.add_argument('--eval', default='ev1', help='comma-separated evaluation jobs (the first one supplies the main timing table)')
    ap.add_argument('--out', default='report.md')
    a = ap.parse_args()
    T, E, TM, meta, TM_extra = load(a.train, a.eval)
    M = {b: {s: metrics(E, b, s) for s in SETTINGS} for b in E}
    V = verdicts(E, T, M)
    R1 = r1_gate(E)
    ctrl_ok = bool(meta) and all(all(x['controls'][k]['passed'] for k in ('C2', 'C3', 'C5b')) for x in meta)
    gates_ok = R1['passed'] and ctrl_ok
    V['acceptance_gates'] = dict(R1=R1['passed'], controls=ctrl_ok, passed=gates_ok)
    if not gates_ok:                  # A2.5: no verdict stands if an acceptance gate failed
        for t, v in V.items():
            if isinstance(v, dict) and 'H1' in v:
                v['H1']['passed'] = False; v['H2_passed'] = False; v['useful_winner'] = False
                for x in v['H2'].values():
                    x['passed'] = False
                v['invalidated_by_failed_gate'] = True
        V['ranking_useful_H2'] = []
    if TM is not None:
        inv = TM.get('invocations', [])
        # exact coverage: every (setting, bank, rule) subject x every dev6 case x every repetition x both directions
        want = {(sj['setting'], sj['label'], r_, c, rep_, pos_dir)
                for sj in TM.get('subjects', []) for r_ in sj['rules'] for c in range(6)
                for rep_ in range(TM.get('reps', 0)) for pos_dir in (0, 1)}
        nsub = {sj['setting']: 0 for sj in TM.get('subjects', [])}
        for sj in TM.get('subjects', []):
            nsub[sj['setting']] += len(sj['rules'])
        got = [(x['setting'], x['label'], x['rule'], x['case'], x['rep'], int(x['position'] >= nsub.get(x['setting'], 0)))
               for x in inv]
        ok_tm = (TM.get('complete') and TM.get('valid') and len(got) == len(set(got)) and set(got) == want
                 and all(math.isfinite(x['seconds']) and x['seconds'] > 0 for x in inv))
        if not ok_tm:
            TM = dict(TM, median_ms={}, invalid=True)
    plots(E, M, HERE / 'plots')
    e2e_plot(E, M, TM, HERE / 'plots')
    banks = [b for b in ORDER if b in M]
    L = []
    w = L.append
    w('# Smooth and derivative-trained banks for the off-mesh Burgers 2D ROM (lane C3)\n')
    w('Retrains the Burgers 2D coordinate-network bank with a gradient-matching (Sobolev) loss and with smaller '
      'initial random-Fourier-feature scales, and measures what that does to the quadrature points the off-mesh ROM '
      'needs and to its accuracy. Status: **rounds 1 and 2 (when present), development/validation cohorts (dev6 ∪ val32) only; every '
      'accuracy number is PROVISIONAL** because the references are first-order (ST = space+time refined, S = '
      'space-only refined; both reported). Generated by `make_report.py` from the run JSONs; pre-registration '
      '`DESIGN.md` (amendments A1–A6).\n')
    w('## What ran\n')
    w('| job | role | GPU | job id | content |')
    w('|---|---|---|---|---|')
    tj = next(iter(T.values()), {}) if T else {}
    w(f"| `{a.train}` | training | {tj.get('gpu', '—')} | {tj.get('job_id', '—')} | {', '.join(sorted(T))} |")
    if meta:
        w(f"| `{a.eval}` | evaluation at ${E[banks[0]]['mesh']}^2$ | {meta[0]['gpu']} | {meta[0]['job_id']} | {', '.join(banks)} |")
    if TM:
        w(f"| `{a.eval}` (timing) | paired timing | {TM['gpu']} | {TM['job_id']} | m* rules of every bank |")
    w('')
    w('## Answers in brief (generated; PROVISIONAL, development/validation, one training seed per bank)\n')
    if all(x in M for x in ('base', 'frozen_lane', 'frozen_deployed', 'sob01', 'sig1', 'sig2')):
        mB = lambda b, s, k: M[b][s][k]
        ms = lambda b, s, bb: M[b][s]['mstar'][f'gref|G|{bb}']
        R_ = lambda b, s, r, p='gref': E[b]['settings'][s]['rho'][r][p]['max']
        dS = lambda t, s, k: 100 * (M[t][s][k] / M['base'][s][k] - 1)
        accm = sorted({ms(b, 'acc', 0.06) for b in banks})
        w(f"- **H2 (smoothness lowers the points needed): no resolved improvement** (sig1, sig2: UNRESOLVED; sob01: not met). Own-state $m^*(0.06)$ in `acc`: {', '.join(f'{b} {ms(b, "acc", 0.06)}' for b in banks)}. "
          f"In `fast` it is {ms('base','fast',0.06)} for `base` and {ms('sig1','fast',0.06)} / {ms('sig2','fast',0.06)} / {ms('sob01','fast',0.06)} for sig1 / sig2 / sob01. "
          f"However, `frozen_lane` (the same recipe and seed as `base`) gives {ms('frozen_lane','fast',0.06)}, so the pre-registered verdict is UNRESOLVED, not a pass.")
        w(f"- **Onset.** Gauss 32² gives worst ρ {fmt(min(R_(b,'acc','gauss32') for b in banks))}–{fmt(max(R_(b,'acc','gauss32') for b in banks))} in `acc` for every bank. "
          f"The restricted-vector diagnostic (first 320 tests only, own denominator, solver not rerun) gives {fmt(min(E[b]['settings']['acc']['rho']['gauss32']['gref']['max_320'] for b in banks))}–{fmt(max(E[b]['settings']['acc']['rho']['gauss32']['gref']['max_320'] for b in banks))}. "
          f"This is consistent with a test-frequency contribution to the onset.")
        tv = [M[b][x]['tail']['gref']['varrho'] for b in banks for x in SETTINGS]
        w(f"- **Tail.** Smaller initial σ gives steeper fitted Gauss tails in these runs: in `acc` the fitted parameter is {M['base']['acc']['tail']['gref']['varrho']:.4f} (base), {M['sig2']['acc']['tail']['gref']['varrho']:.4f} (sig2) and {M['sig1']['acc']['tail']['gref']['varrho']:.4f} (sig1); over all banks and settings it spans {min(tv):.4f}–{max(tv):.4f}. "
          f"Worst ρ at Gauss 128² is {fmt(R_('sig1','acc','gauss128'))} (sig1), {fmt(R_('base','acc','gauss128'))} (base) and {fmt(R_('frozen_deployed','acc','gauss128'))} (frozen); at Gauss 256² it is {fmt(R_('sig1','acc','gauss256'))} (sig1) and {fmt(R_('base','acc','gauss256'))} (base). These rungs lie past the 0.01 bar.")
        w(f"- **Representation.** Neither smaller-σ run increases the median `acc` projection floor ({fmt(mB('base','acc','e_val_med'))} base, {fmt(mB('sig2','acc','e_val_med'))} sig2, {fmt(mB('sig1','acc','e_val_med'))} sig1). The trained Fourier frequencies stay near their initial scale: the final median ‖B_j‖ is {T['sig1']['train']['B_final']['0.5']:.2f} for sig1 and {T['base']['train']['B_final']['0.5']:.2f} for base.")
        w(f"- **H1 (Sobolev).** Against the common-support **mesh** derivative targets, sob01 lowers the median gradient error by {-dS('sob01','acc','e_grad2c_med'):.1f}% (D2) / {-dS('sob01','acc','e_grad4_med'):.1f}% (D4) and the median projection floor by {-dS('sob01','acc','e_val_med'):.1f}% in `acc`. "
          f"In `fast` the reductions are only {-dS('sob01','fast','e_grad2c_med'):.1f}% / {-dS('sob01','fast','e_grad4_med'):.1f}%. Its converged-quadrature rollout error vs S changes by {dS('sob01','acc','E_S'):+.1f}% (`acc`) and {dS('sob01','fast','E_S'):+.2f}% (`fast`), so there is no resolved rollout improvement in this run. H1 is not met.")
        w(f"- **Rollout accuracy.** Lower projection and derivative errors did not produce a resolved rollout improvement, and the limiting contribution is not isolated here. Across all evaluated banks the worst `acc` error against ST lies in {pct(min(M[b]['acc']['E_ST'] for b in banks))}–{pct(max(M[b]['acc']['E_ST'] for b in banks))}%, and against S in {pct(min(M[b]['acc']['E_S'] for b in banks))}–{pct(max(M[b]['acc']['E_S'] for b in banks))}%. "
          f"The `base` vs `frozen_lane` comparator differs by {100 * V['noise']['acc|E_S']:.1f}% on `acc` $E_S$, so no treatment-versus-base `acc` $E_S$ change clears the pre-registered noise gate.")
        w(f"- **Evaluation procedure.** The same frozen bank gives a worst `fast` S error of {pct(mB('frozen_deployed','fast','E_S'))}% under the 2D lane's deployed procedure (head-based rotation, trust radius and coefficient population) and {pct(mB('frozen_lane','fast','E_S'))}% under this lane's least-squares procedure (A1.2). "
          f"In `acc` the two are close ({pct(mB('frozen_deployed','acc','E_S'))}% vs {pct(mB('frozen_lane','acc','E_S'))}%). Retrained banks must be compared with `frozen_lane`.")
        if TM and TM.get('median_ms'):
            av = [v for k, v in TM['median_ms'].items() if k.startswith('acc|')]
            w(f"- **Cost.** At the fixed `acc` setting and rule (every bank's $m^*$ there is the same Gauss rung), the observed per-query medians lie in {min(av):.1f}–{max(av):.1f} ms on one {TM['gpu']}.")
        if 'base_s1' in M:
            sd = lambda k, s_: 100 * (M['base_s1'][s_][k] / M['base'][s_][k] - 1)
            w(f"- **Seed variance (round 2).** `base_s1` (seed 1) differs from `base` by {sd('E_S','acc'):+.1f}% (`acc`) and {sd('E_S','fast'):+.1f}% (`fast`) in converged-quadrature rollout error vs S, by {sd('e_grad2c_med','acc'):+.1f}% in the `acc` gradient error, and needs m*(0.06) = {M['base_s1']['acc']['mstar']['gref|G|0.06']} (`acc`) / {M['base_s1']['fast']['mstar']['gref|G|0.06']} (`fast`) points. These differences enter the noise yardstick below.")
        if 'base_ev2r' in M:
            rd = max(abs(M['base_ev2r'][x][k] / M['base'][x][k] - 1) for x in SETTINGS for k in ('E_S', 'E_ST', 'e_val_med', 'e_grad2c_med'))
            w(f"- **Re-evaluation.** `base` evaluated again in the round-2 job on a different A100 model ({E['base_ev2r'].get('eval_job')}) reproduces its round-1 metrics to a largest relative difference of {rd:.1e}.")
        sobs = [x for x in ('sob001', 'sob01', 'sob1') if x in M]
        vals = [x for x in ('frozen_lane', 'base', 'base_s1', 'sig2', 'sig1') if x in M]
        if sobs and vals:
            g = lambda x: M[x]['acc']['e_grad2c_med']
            w(f"- **Derivatives across all banks (descriptive).** In `acc` every Sobolev bank's median gradient error on the common support ({fmt(min(g(x) for x in sobs))}–{fmt(max(g(x) for x in sobs))}) lies below every value-only bank's ({fmt(min(g(x) for x in vals))}–{fmt(max(g(x) for x in vals))}). The value-only banks alone vary by a factor of {max(g(x) for x in vals) / min(g(x) for x in vals):.1f} between seeds and recipes, so the pre-registered treatment-versus-base gate (whose noise yardstick now includes `base_s1`) does not resolve the effect.")
        lad = [x for x in ('sob001', 'sob01', 'sob1') if x in M]
        if len(lad) > 1:
            w('- **λ ladder (gradient error D2 / projection floor / E_S, change vs base, `acc`; m*(0.06) acc/fast):** ' + '; '.join(
                f"λ={T[x]['lam_sob'] if x in T else '?'}: {dS(x,'acc','e_grad2c_med'):+.0f}% / {dS(x,'acc','e_val_med'):+.0f}% / {dS(x,'acc','E_S'):+.1f}%; {M[x]['acc']['mstar']['gref|G|0.06']}/{M[x]['fast']['mstar']['gref|G|0.06']}" for x in lad) + '.')
        fc_ = [r for m_ in meta for r in m_.get('fom_comparators', [])]
        if 'coarse' in E and fc_ and all(r.get('accepted') for r in fc_):
            wc = lambda b, s_: worst(E, b, s_, 'gref', 'ref_S_evolved_129')
            f129 = max(r['ref_S_evolved_129'] for r in fc_ if r['nodes'] == 129)
            f257 = max(r['ref_S_evolved_129'] for r in fc_ if r['nodes'] == 257)
            w(f"- **Coarse-data control.** On the common 129-node restriction against S, the ROM with the bank trained only on 129-node data has worst error {pct(wc('coarse','acc'))}% (`acc`) / {pct(wc('coarse','fast'))}% (`fast`). The FOM has {pct(f129)}% at 129 nodes and {pct(f257)}% at 257 nodes, and `base` has {pct(wc('base','acc'))}% / {pct(wc('base','fast'))}%. See the coarse-control section for ST. The coarse bank needs m*(0.06) = {M['coarse']['acc']['mstar']['gref|G|0.06']} (`acc`) / {M['coarse']['fast']['mstar']['gref|G|0.06']} (`fast`) points against base's {M['base']['acc']['mstar']['gref|G|0.06']} / {M['base']['fast']['mstar']['gref|G|0.06']}, and its median `acc` projection floor is {fmt(M['coarse']['acc']['e_val_med'])} against base's {fmt(M['base']['acc']['e_val_med'])}.")
        w('- **Pre-registered outcome:** no useful winner, so nothing is promoted to 3D and `comb` is not run (A1.5)' + (' (verdicts below re-read with `base_s1` in the noise yardstick).\n' if 'base_s1' in M else '. The round-1 verdicts are provisional until the seed-1 comparator `base_s1` (round 2) is evaluated.\n'))
    w('## Method in one screen\n')
    w('Each backward-Euler step approximately minimises $\\tfrac12\\lVert r(c)\\rVert_2^2$ by Levenberg–Marquardt, with $r(c)=D\\big(Ac-p+\\Delta t\\,(N(c)+\\nu\\Lambda Ac)\\big)$ ($A=\\Phi^{\\mathsf T}G\'$ the exact projection on the $M$ sine tests, $\\Lambda$ their eigenvalues, $D=(I+\\Delta t\\nu\\Lambda)^{-1}$, $p$ the previous step) and the tested advection '
      '$N_a(c)=L\\sum_q w_q\\,\\psi_a(x_q)\\,u(x_q)\\,(u_x+u_y)(x_q)$, where $u=G\'(x)c$ is the rotated bank. For a rule $Q$ and a state $c$, '
      '$\\rho_Q(c)=\\lVert N^{Q}(c)-N^{\\mathrm{G640}}(c)\\rVert_2/\\lVert N^{\\mathrm{G640}}(c)\\rVert_2$. The Sobolev arm adds '
      '$\\lambda\\,\\overline{\\lVert\\nabla\\hat u-D_hu\\rVert^2}/\\overline{\\lVert D_hu\\rVert^2}$ to the value loss, with $D_h$ second-order central differences on the training mesh. '
      'The projection floor is $\\min_c\\lVert G\'c-u_{\\rm ref}\\rVert/\\lVert u_{\\rm ref}\\rVert$ on the 257² nodes, and the gradient error is that of $\\nabla(G\'c^\\star)$ against $D^{(2)}u_{\\rm ref}$ or $D^{(4)}u_{\\rm ref}$.\n')
    w('## Training\n')
    w('| bank | σ (initial) | λ | steps | train h | recon mean (train) | train LS floor median / max | ‖B_j‖ median init→final | ‖B_j‖ max init→final | R2a | FD stencil sensitivity (train, median) |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for b in [x for x in ORDER if x in T]:
        t = T[b]; tr = t['train']
        r2a = ', '.join(f"{x['step']}: {x['got']} vs {x['want']}" for x in tr.get('R2a', [])) or 'n/a'
        w(f"| {b} | {t['sigma']} | {t['lam_sob']} | {tr['steps_done']} | {tr['seconds'] / 3600:.2f} | {fmt(tr['recon_rel_l2_mean'])} | "
          f"{fmt(t['rotation']['train_ls_floor']['median'])} / {fmt(t['rotation']['train_ls_floor']['max'])} | "
          f"{tr['B_init']['0.5']:.2f} → {tr['B_final']['0.5']:.2f} | {tr['B_init']['1']:.2f} → {tr['B_final']['1']:.2f} | {r2a} | "
          f"{fmt(t['data']['fd_target_uncertainty']['median'])} |")
    w('')
    for s in SETTINGS:
        Rp = E[banks[0]]['settings'][s]['R_prime']; Mm = E[banks[0]]['settings'][s]['M']
        w(f'## Setting `{s}` ($R\'={Rp}$, $M={Mm}$)\n')
        w('### Representation and derivatives (discrete, 257² nodes, vs the S reference; PROVISIONAL)\n')
        w('| bank | projection floor median | projection floor worst | gradient error median, D2 all interior | D2 on the D4 support | D4 | gradient error worst (D2) | FD stencil sensitivity (ref., median) |')
        w('|---|---|---|---|---|---|---|---|')
        for b in banks:
            m = M[b][s]
            w(f"| {b} | {fmt(m['e_val_med'])} | {fmt(m['e_val_max'])} | {fmt(m['e_grad_med'])} | {fmt(m['e_grad2c_med'])} | {fmt(m['e_grad4_med'])} | {fmt(m['e_grad_max'])} | {fmt(m['fd_unc_med'])} |")
        w('')
        w('### Quadrature points needed (worst ρ over the population ≤ bar, confirmed at every larger rung)\n')
        w('| bank | Gauss m*(0.116) own / common | Gauss m*(0.06) own / common | Gauss m*(0.01) own / common | interpolated m(0.06), descriptive | Fibonacci m*(0.06) own | tail ϱ̂ (R²) | target checks C4 |')
        w('|---|---|---|---|---|---|---|---|')
        for b in banks:
            m = M[b][s]; ms = m['mstar']; tl = m['tail'].get('gref', {})
            tail = f"{tl['varrho']:.4f} ({tl['r2']:.2f})" if tl.get('varrho') else 'no resolved tail'
            w(f"| {b} | {mfmt(ms['gref|G|0.116'], 'G')} / {mfmt(ms['common|G|0.116'], 'G')} | {mfmt(ms['gref|G|0.06'], 'G')} / {mfmt(ms['common|G|0.06'], 'G')} | "
              f"{mfmt(ms['gref|G|0.01'], 'G')} / {mfmt(ms['common|G|0.01'], 'G')} | {fmt(interp_m(E, b, s, 0.06))} | {mfmt(ms['gref|F|0.06'], 'F')} | {tail} | {'pass' if m['C4_pass'] else 'FAIL'} |")
        w('')
        w('Worst ρ at selected Gauss rungs (own reached states; in brackets: restricted to the first 320 tests, Hari\'s $M$):\n')
        sel = (32, 48, 64, 96, 128)
        w('| bank | ' + ' | '.join(f'Gauss {p}²' for p in sel) + ' |')
        w('|---|' + '---|' * len(sel))
        for b in banks:
            R = E[b]['settings'][s]['rho']
            w(f'| {b} | ' + ' | '.join(f"{fmt(R[f'gauss{p}']['gref']['max'])} ({fmt(R[f'gauss{p}']['gref']['max_320'])})" for p in sel) + ' |')
        w('')
        nsp = len(E[banks[0]]['settings'][s].get('spectra', {}).get('states', []))
        w(f'### Smoothness (Chebyshev, {nsp} own reached states)\n')
        w('| bank | u: n(1e-8) median / max | f: n(1e-8) median / max | f: n(1e-4) median | f classification (geometric/algebraic/inconclusive/unresolved) |')
        w('|---|---|---|---|---|')
        for b in banks:
            sp = M[b][s]['spectra']
            if not sp:
                continue
            cl = sp.get('f|class', {})
            nn = lambda e_: (f"unresolved ({e_['unresolved']}/{nsp})" if e_['median'] is None else f"{fmt(e_['median'])} / {fmt(e_['max'])} ({nsp - e_['unresolved']}/{nsp} resolved)")
            w(f"| {b} | {nn(sp['u|n_1e-08'])} | {nn(sp['f|n_1e-08'])} | {fmt(sp['f|n_1e-04']['median'])} ({nsp - sp['f|n_1e-04']['unresolved']}/{nsp} resolved) | "
              f"{cl.get('geometric', 0)}/{cl.get('algebraic', 0)}/{cl.get('inconclusive', 0)}/{cl.get('unresolved', 0)} |")
        w('\nBandwidths are medians over the states whose n(ε) is resolved (256 vs 512 points agree within 2).\n')
        w('')
        ncs = len(rows(E, banks[0], s, 'gref'))
        w(f'### End-to-end (worst over the {ncs} cases, evolved error %, PROVISIONAL; ST beside S)\n')
        arms = ['gref', 'gauss32', 'gauss48', 'gauss64', 'gauss96', 'fib1597', 'fib4181', 'fib6765', 'lat64']
        w('| bank | ' + ' | '.join(arms) + ' |')
        w('|---|' + '---|' * len(arms))
        for b in banks:
            cells = []
            for arm in arms:
                st_, s_ = worst(E, b, s, arm, 'ref_ST_evolved'), worst(E, b, s, arm, 'ref_S_evolved')
                cells.append(f'{pct(st_)} / {pct(s_)}')
            w(f'| {b} | ' + ' | '.join(cells) + ' |')
        w('\nEach cell: ST / S.\n')
        for b in banks:
            m = M[b][s]
            if m['C6_failed_cases']:
                w(f"- `{b}`: C6-failed cases {', '.join(m['C6_failed_cases'])}; converged-quadrature rollout worst error without them: ST {pct(m['E_ST_without_C6_failed'])}%, S {pct(m['E_S_without_C6_failed'])}%.")
        w('')
    if TM and TM.get('invalid'):
        w('## Cost\n\nThe timing run did not reproduce the evaluation rollouts (or was incomplete); its numbers are suppressed.\n')
    if TM and not TM.get('invalid'):
        w('## Cost at each bank\'s own m* rules (median ms per query, one GPU, paired A–B–A, dev6 × 3 repetitions)\n')
        w('| setting | bank | rule | m | median ms | max coefficient difference vs the evaluation rollout |')
        w('|---|---|---|---|---|---|')
        for k, v in TM['median_ms'].items():
            s, b, rule = k.split('|')
            inv = [x for x in TM['invocations'] if x['setting'] == s and x['label'] == b and x['rule'] == rule]
            w(f"| {s} | {b} | {rule} | {inv[0]['m']} | {v:.1f} | {fmt(max(x['coeff_rel_diff_vs_phase1'] for x in inv))} |")
        w('')
    fc = [r for m_ in meta for r in m_.get('fom_comparators', [])]
    fc_ok = bool(fc) and all(r.get('accepted') for r in fc) and len(fc) == 2 * N_CASES
    if 'coarse' in E and fc and not fc_ok:
        w('## Coarse-data control\n\nThe FOM comparators did not all pass their acceptance check (finite, relative residual ≤ 1e-8, 38 cases at each node count); the comparison is withheld.\n')
    if 'coarse' in E and fc_ok:
        w('## Coarse-data control (A1.7, A2.6; PROVISIONAL)\n')
        w('Worst over the 38 cases of the evolved error % on the common 129-node restriction of the references (each cell ST / S). '
          '`coarse` was trained only on 129-node data; the FOMs are the training generator at 129 and 257 nodes.\n')
        w('| model | acc | fast |')
        w('|---|---|---|')
        for b in [x for x in ('coarse', 'base', 'base_ev2r') if x in E]:
            cells = [f"{pct(worst(E, b, s, 'gref', 'ref_ST_evolved_129'))} / {pct(worst(E, b, s, 'gref', 'ref_S_evolved_129'))}" for s in SETTINGS]
            w(f"| ROM `{b}` (converged quadrature) | " + ' | '.join(cells) + ' |')
        for n in (129, 257):
            rr = [r for r in fc if r['nodes'] == n]
            cell = f"{pct(max(r['ref_ST_evolved_129'] for r in rr))} / {pct(max(r['ref_S_evolved_129'] for r in rr))}"
            w(f"| FOM, {n} nodes ({len(rr)} cases) | {cell} | {cell} |")
        w('\nBeating the 129-node FOM supports only the limited claim that the off-mesh accuracy is not solely inherited from finer training data; not beating it would not show that finer data are necessary (A2.6).\n')
    for ej, tm in TM_extra.items():
        inv = tm.get('invocations', [])
        if tm.get('valid') and tm.get('complete'):
            w(f'## Cost, evaluation job `{ej}` (median ms per query, {tm["gpu"]}; not comparable across jobs)\n')
            w('| setting | bank | rule | median ms |')
            w('|---|---|---|---|')
            for k, v in tm['median_ms'].items():
                s_, b_, r_ = k.split('|')
                w(f'| {s_} | {b_} | {r_} | {v:.1f} |')
            w('')
    w('## Gates and controls\n')
    if meta:
        c = meta[0]['controls']
        w(f"- C2 (bandwidth ordering, manufactured bumps): {'pass' if c['C2']['passed'] else 'FAIL'} ({c['C2']['n8_w02']} < {c['C2']['n8_w005']}); "
          f"C3 (kink classified algebraic): {'pass' if c['C3']['passed'] else 'FAIL'}; C5b (difference stencil response): {'pass' if c['C5b']['passed'] else 'FAIL'}.")
    for b in [x for x in ORDER if x in T]:
        d_ = T[b]['data']
        r2a = T[b]['train'].get('R2a', [])
        w(f"- `{b}` training data: R0 fingerprint {'matches the original job (pass)' if d_.get('R0_fingerprint_matches_r3a') else 'not applicable'}"
          + ('; R2a ' + ', '.join(f"step {x['step']} {x['got']} vs original {x['want']} ({'equal' if x['match'] else 'differs; recorded only, A2.5'})" for x in r2a) if r2a else '') + '.')
    ev_ = (HERE / 'audits/R2b-evidence.txt').read_text()
    import re as _re
    dv = _re.findall(r'lane\(lam=0\) vs original ([0-9.e+-]+)', ev_)
    w(f"- R2b (local GB10): lane trainer at λ=0 vs the original trainer, max parameter difference {dv[0] if dv else '?'} with default XLA autotuning and {dv[-1] if dv else '?'} with deterministic XLA flags (`audits/R2b-evidence.txt`): bit-identical only under deterministic compilation, so retrains are not bitwise reproducible and the noise yardstick is needed.")
    g8 = [E[b]['settings'][s]['rho']['gauss8']['gref']['max'] for b in banks for s in SETTINGS]
    w(f"- C1 stress arm (Gauss 8²): worst ρ {fmt(min(g8))}–{fmt(max(g8))} over banks and settings, far above every bar, as expected.")
    for b in banks:
        el = {s: eligible(E, T, b, s, M) for s in SETTINGS}
        c5 = E[b]['C5a']['rel_error_by_step']
        w(f"- `{b}`: C5a min rel. error {fmt(min(c5.values()))}; " + '; '.join(
            f"{s}: C4 {'pass' if M[b][s]['C4_pass'] else 'FAIL'}, gref-vs-768 rollout {fmt(M[b][s]['g768_max'])}, "
            f"tight-solver distance {fmt(M[b][s]['tight_max'])}, eligibility {'ok' if not el[s] else 'FAILED: ' + ', '.join(el[s])}"
            for s in SETTINGS) + '.')
    w('')
    w(f"- **R1** (frozen bank with the deployed rotation reproduces the 2D lane's job {R1_SRC['job_id']} at $1024^2$; errors PROVISIONAL): {'pass' if R1['passed'] else 'FAIL'} — " + '; '.join(
        f"{s}: Gauss 64² worst ρ on lat64 states {fmt(v['g64_lat64'])} (2D lane {fmt(R1_PINNED[s]['g64_lat64'])}), gref worst ST {pct(v['gref_ST'])}% ({pct(R1_PINNED[s]['gref_ST'])}%), lat64 worst ST {pct(v['lat64_ST'])}% ({pct(R1_PINNED[s]['lat64_ST'])}%)"
        for s, v in R1.get('per_setting', {}).items()) + '.')
    w('')
    w('## Verdicts (DESIGN A1.5, A2.4, A4; PROVISIONAL, development/validation; one seed per arm)\n')
    if not gates_ok:
        w('**Acceptance gates failed (R1 or a manufactured control): the verdicts below are NOT valid and are shown for diagnosis only.**\n')
    if 'noise' in V:
        w('Noise yardstick (max of |base / frozen-lane − 1| and, when present, |base / base_s1 − 1|): ' + ', '.join(f'{k} {v:.3f}' for k, v in V['noise'].items()) + '.\n')
    for t, v in V.items():
        if t in ('noise', 'ranking_useful_H2', 'acceptance_gates'):
            continue
        h2 = '; '.join(f"{k}: ×{x['reduction']:.2f} ({'pass' if x['passed'] else ('unresolved' if not x['resolved'] else 'no')})" for k, x in v['H2'].items())
        would = [x for x in v['H2'].values() if x['reduction'] >= 1.5 and x['other_reduction'] >= 1
                 and x['common_same_direction'] and x['common_same_direction_other']]
        h2s = 'PASS' if v['H2_passed'] else ('UNRESOLVED (every other H2 condition holds at some bar, but base and frozen-lane disagree on m* there)'
                                             if would and not any(x['resolved'] for x in would) else 'not met')
        w(f"- **{t}**: H1 {'PASS' if v['H1']['passed'] else 'not met'} (gradient criterion {v['H1']['grad']}, value {v['H1']['value']}, rollout {v['H1']['rollout']}); "
          f"H2 {h2s} [{h2}]; useful winner: {'YES' if v['useful_winner'] else 'no'}. "
          + ' '.join(f"Δ{s}: E_S {pct(v['delta'][s]['E_S'])}%, gradient error (D2 / D4 on the common support) {pct(v['delta'][s]['e_grad2c_med'])}% / {pct(v['delta'][s]['e_grad4_med'])}%, projection floor {pct(v['delta'][s]['e_val_med'])}%." for s in SETTINGS))
    w(f"\nRanking of useful H2 winners (A2.4/A4): {', '.join(V.get('ranking_useful_H2', [])) or 'none'}.")
    w('')
    w('## Plots\n')
    for p in ('frontier', 'ladders', 'derivatives', 'e2e', 'spectra'):
        w(f'![{p}](plots/{p}.png)\n')
    w('## Glossary\n')
    for k, v in GLOSSARY:
        w(f'- **{k}** — {v}')
    (HERE / a.out).write_text('\n'.join(L) + '\n')
    (HERE / 'report_numbers.json').write_text(json.dumps(dict(metrics=M, verdicts=V), indent=1, default=str))
    print('wrote', a.out)


GLOSSARY = [
    ('bank', 'the frozen coordinate network G(x) whose columns span the reduced trial space; u(x) = G(x)c.'),
    ('σ (initial)', 'the standard deviation of the random Fourier frequencies B at initialisation; B is trained afterwards, so the final ‖B_j‖ are also reported.'),
    ('λ', 'weight of the gradient-matching (Sobolev) term in the training loss.'),
    ('base', 'the original recipe retrained in this lane with the same data and seed: the comparator for every treatment.'),
    ('frozen bank, deployed / lane rotation', 'the bank used by the 2D off-mesh lane, evaluated with that lane\'s rotation (replication) and with this lane\'s rotation procedure (comparator for base).'),
    ('acc / fast', 'linear settings: span of the first R\'=384 (acc) or 128 (fast) importance-ordered bank columns, tested against M=1536 or 512 sine functions.'),
    ('ρ', 'relative error of a quadrature rule\'s tested advection vector against the converged target (Gauss 640²), per state.'),
    ('own / common states', 'own = the states the bank\'s own converged rollout reaches (k=1..50, 38 cases); common = the S reference fields projected onto each bank (same physical states for every bank).'),
    ('m*(b)', 'smallest rule size whose worst ρ is ≤ b and stays ≤ b at every larger rung; ">N" means no rung reached it.'),
    ('tail ϱ̂', 'effective geometric convergence parameter fitted to the median ρ over the resolved Gauss rungs (descriptive, not a certified analyticity width).'),
    ('projection floor', 'relative error of the least-squares projection of the reference field onto the bank span, on the 257² nodes.'),
    ('gradient error (D2 / D4)', 'relative error of the projection\'s exact gradient against second- / fourth-order central differences of the reference field.'),
    ('FD stencil sensitivity', '‖D2 − D4‖/‖D4‖: how much the derivative target itself depends on the stencil (an indicator, not a bound).'),
    ('converged-quadrature rollout error (E_S, E_ST)', 'worst error over cases and output times of the ROM with the converged rule (Gauss 640²) against the S or ST reference.'),
    ('ST / S', 'refined reference solutions at 8192²: ST also refines time (Δt/16), S keeps the ROM\'s Δt (space-only).'),
    ('n(ε)', 'Chebyshev degree beyond which the coefficient envelope of a field stays below ε; a bandwidth.'),
    ('C2–C6, R0–R2', 'pre-registered gates and controls (DESIGN A1.6, A2.5, A4).'),
    ('PROVISIONAL', 'the references are first-order; accuracy numbers can move when second-order references exist.'),
    ('recon mean (train)', 'mean over the 16 384 training states of the relative L2 error of the trained autodecoder (bank times head) on all training points.'),
    ('train LS floor', 'relative error of the least-squares projection of each training state onto the full 512-column bank, on its training mesh (median / max over states).'),
    ('‖B_j‖', 'length of the j-th random Fourier frequency vector (cycles per unit length), initial and after training.'),
    ('R2a / R0 / R2b', 'replication gates: R0 = training data fingerprint equals the original job; R2a = loss at logged steps equals the original log; R2b = local bitwise parity of the trainer code.'),
    ('gref, lat64, Gauss p², Fibonacci n', 'quadrature arms: Gauss 640² (the converged continuum rule), the deployed 63×63 mesh lattice, tensor Gauss–Legendre with p points per axis, rank-1 Fibonacci lattice with n points.'),
    ('evolved error', 'the largest error over the five output times after t = 0 (the initial fit is excluded), relative to the initial field norm on the same nodes.'),
    ('common support', 'interior nodes at least two nodes from the wall, where both the second- and fourth-order differences exist.'),
    ('interpolated m', 'descriptive log-linear interpolation of the points needed between the confirmed rung and the rung below it.'),
    ('R²', 'coefficient of determination of the tail fit (1 = a perfect straight line in log ρ against 2p).'),
    ('geometric / algebraic / inconclusive / unresolved', 'classification of a Chebyshev envelope: geometric decay (analytic-like), power-law decay, neither fit clearly better, or too few resolved blocks.'),
    ('query ms', 'wall time of one ROM query: initial fit, 50 implicit LM steps and decoding six output fields at 1024², median over paired repetitions.'),
    ('coefficient difference vs the evaluation rollout', 'relative difference between the coefficients of a timed rollout and those of the same rollout in the evaluation phase (checks that the timed computation is the evaluated one).'),
    ('noise yardstick', 'relative difference between base and frozen-lane (same recipe and seed; later also base vs base_s1); a treatment effect counts only if it is more than twice this.'),
    ('C6 / non-accepted exit', 'an LM step that stopped on the budget, a tiny step or the damping limit instead of tolerance or stationarity.'),
    ('H1 / H2 / useful winner', 'pre-registered hypotheses (Sobolev helps; smoothness reduces points) and the gate for promoting a bank (DESIGN A1.5).'),
]


if __name__ == '__main__':
    main()
