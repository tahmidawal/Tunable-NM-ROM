"""jcp-dims decision rules (DESIGN 6, A1.2, A1.5, A1.8, A2.2, A2.5, A3.2, A4.1). Pure NumPy; every verdict function is
exercised by test_dims.py on synthetic fixtures (DESIGN 7, decision-rule synthetics).

Conventions: a per-case record is (e_X, e_Y, eps, eligible); "eligible" already includes E0 eligibility, both
configurations numerically qualified and both runs valid (the caller passes unqualified cells as `qualified=False`).
"""
from __future__ import annotations

import numpy as np

BAR = 2.5e-4
RHO_BAR = 0.116
REL = 0.8              # point-efficiency factor (6.3)


# --------------------------------------------------------------------------------------------- paired comparison
def paired(eX, eY, eps, eligible=None):
    """Counts and statistics of the paired change e_Y - e_X on eligible cases; resolved iff |e_Y - e_X| > 2 eps."""
    eX, eY, eps = (np.asarray(a, float) for a in (eX, eY, eps))
    el = np.ones(len(eX), bool) if eligible is None else np.asarray(eligible, bool)
    d = (eY - eX)[el]
    tie = np.abs(d) <= 2 * eps[el]
    res = ~tie
    return dict(n_eligible=int(el.sum()), n_resolved=int(res.sum()), improve=int(np.sum(res & (d < 0))),
                worsen=int(np.sum(res & (d > 0))), tie=int(tie.sum()), ineligible=int((~el).sum()),
                median_change=float(np.median(d)) if d.size else float('nan'),
                worst_change=float(d[np.argmax(np.abs(d))]) if d.size else float('nan'))


def transition(eX, eY, eps, eligible=None, qualified=(True, True)):
    """6.1a (AM-7): X -> Y improves / does not improve / inconclusive (precedence: unqualified, empty, resolution)."""
    if not all(qualified):
        return 'inconclusive (unqualified cell)', None
    p = paired(eX, eY, eps, eligible)
    if p['n_eligible'] == 0:
        return 'inconclusive (no eligible cases)', p
    if p['n_resolved'] < 0.5 * p['n_eligible']:
        return 'inconclusive (resolution)', p
    if p['improve'] > 0.5 * p['n_resolved'] and p['median_change'] < 0:
        return 'improves', p
    return 'does not improve', p


def range_verdict(labels):
    """6.1b: monotone iff every transition improves; 'not monotone' if any does not improve; else inconclusive."""
    if labels and all(x == 'improves' for x in labels):
        return 'monotone'
    if any(x == 'does not improve' for x in labels):
        return 'not monotone'
    return 'inconclusive'


def two_way(eX, eY, eps, eligible=None, qualified=(True, True), names=('X', 'Y')):
    """6.4 GAL vs LSPG (and LMF vs default): '<X> more accurate' / '<Y> more accurate' / 'neither' / inconclusive."""
    if not all(qualified):
        return 'inconclusive (unqualified cell)', None
    p = paired(eY, eX, eps, eligible)                  # change e_X - e_Y; 'improve' = X more accurate than Y
    if p['n_eligible'] == 0:
        return 'inconclusive (no eligible cases)', p
    if p['n_resolved'] < 0.5 * p['n_eligible']:
        return 'inconclusive (resolution)', p
    if p['improve'] > 0.5 * p['n_resolved'] and p['median_change'] < 0:
        return f'{names[0]} more accurate', p
    if p['worsen'] > 0.5 * p['n_resolved'] and p['median_change'] > 0:
        return f'{names[1]} more accurate', p
    return 'neither', p


def control_mr(label_1_to_4, qualified):
    """A1.2 M = r must-degrade control: fired / did not fire (adverse) / unavailable."""
    if not all(qualified):
        return 'unavailable (unqualified cell)'
    if label_1_to_4 == 'improves':
        return 'fired'
    if label_1_to_4 == 'does not improve':
        return 'did not fire'
    return f'unavailable ({label_1_to_4})'


# --------------------------------------------------------------------------------------------- qualification
def q2_label(own_valid, succ, rho_max, check=None, ladder_top=False):
    """A1.5 Q2 state machine for one configuration. succ: None (missing) or (valid_all, worst_distance);
    check: None (missing) or (valid_all, worst_distance) of q+ vs q++ on the check cases."""
    if ladder_top:
        return 'unverified (ladder top)'
    if succ is None or not succ[0]:
        return 'unverified (successor invalid)'
    d = succ[1]
    if not np.isfinite(d) or d > BAR or not np.isfinite(rho_max) or rho_max > RHO_BAR:
        return 'fail'
    if check is None or not check[0]:
        return 'unverified (finer rule unavailable)'
    if not np.isfinite(check[1]) or check[1] > BAR:
        return 'unverified (finer rule not converged)'
    return 'pass'


def mstar(labels, sizes):
    """6.2 + A1.5: labels = per candidate (ascending) 'qualified' / 'not qualified' / 'missing'."""
    for i, (l, m) in enumerate(zip(labels, sizes)):
        if l == 'missing':
            return dict(outcome='not reached (incomplete ladder)', m=None)
        if l == 'qualified':
            rest = labels[i + 1:]
            if any(x == 'missing' for x in rest):
                st = 'stability unverified: incomplete ladder'
            elif all(x == 'qualified' for x in rest):
                st = 'stable'
            else:
                st = 'non-monotone'
            return dict(outcome=f'm* = {m} ({st})', m=m, stability=st,
                        failing_larger=[sizes[i + 1 + k] for k, x in enumerate(rest) if x == 'not qualified'])
    return dict(outcome='none (within the candidate list)', m=None)


def family_mstar(per_draw_labels, sizes):
    """A1.5 random families: smallest candidate qualified in ALL draws; median-draw m* with censoring."""
    D = len(per_draw_labels)
    per = [mstar(l, sizes) for l in per_draw_labels]
    if any(p['outcome'].startswith('not reached') for p in per):
        fam = dict(outcome='not reached (incomplete ladder)', m=None)
    else:
        fam = dict(outcome='none (within the candidate list)', m=None)
        for i, m in enumerate(sizes):
            if all(l[i] == 'qualified' for l in per_draw_labels):
                fam = dict(outcome=f'm* = {m} (all {D} draws)', m=m)
                break
    vals = [p['m'] if p['m'] is not None else np.inf for p in per]
    med = float(np.median(vals))
    fin = [v for v in vals if np.isfinite(v)]
    return dict(family=fam, per_draw=per, n_finite=len(fin), median_draw=('> top candidate' if not np.isfinite(med) else med),
                range=(min(fin), max(fin)) if fin else None, censored=D - len(fin))


def width_mstar(labels_by_spec, sizes):
    """A1.5: smallest candidate qualified for every spec (intersection)."""
    for i, m in enumerate(sizes):
        if all(l[i] == 'qualified' for l in labels_by_spec.values()):
            return dict(outcome=f'm* = {m}', m=m)
    if any(l[i] == 'missing' for l in labels_by_spec.values() for i in range(len(sizes))):
        return dict(outcome='not reached (incomplete ladder)', m=None)
    return dict(outcome='none (within the candidate list)', m=None)


def point_efficiency(mX, mY, names=('X', 'Y')):
    """6.3: family m* comparison."""
    if mX is None and mY is None:
        return 'neither qualifies'
    if mY is None:
        return f'{names[0]} qualifies, {names[1]} does not (within its list)'
    if mX is None:
        return f'{names[1]} qualifies, {names[0]} does not (within its list)'
    if mX <= REL * mY:
        return f'{names[0]} needs fewer points'
    if mY <= REL * mX:
        return f'{names[1]} needs fewer points'
    return 'comparable'


def cross_family_gate(rho_tgt, rollout_available, worst_dist_to_target):
    """A3.2 precedence: (b) state-frozen advection flags first; then the rollout part (a) if available."""
    r = np.asarray(rho_tgt, float)
    if r.size and (np.any(~np.isfinite(r)) or np.any(r > RHO_BAR)):
        return 'flagged'
    if not rollout_available:
        return 'gate unavailable (rollout part)'
    if not np.isfinite(worst_dist_to_target) or worst_dist_to_target > BAR:
        return 'flagged'
    return 'pass'


def fixture_admitted(states):
    """A4.1: every state finite, target norm > 0 and finite ratio >= 1."""
    return bool(states) and all(s['finite'] and s['target_norm'] > 0 and np.isfinite(s['rho_tgt']) and s['rho_tgt'] >= 1
                                for s in states)


# --------------------------------------------------------------------------------------------- Shapley (6.1c)
def shapley3(f):
    """f: dict over corners (i, j, k) in {0, 1}^3 -> value (axis order r, M, m). Returns the Shapley contribution of each
    axis (average over the 6 orderings of the marginal change)."""
    import itertools
    phi = np.zeros(3)
    for perm in itertools.permutations(range(3)):
        cur = [0, 0, 0]
        for ax in perm:
            before = f[tuple(cur)]
            cur[ax] = 1
            phi[ax] += f[tuple(cur)] - before
    return phi / 6.


def shapley_report(f_acc, f_cost, d_acc_eps, cost_interval):
    """A2.5: shares normalised only under the separate accuracy / cost thresholds."""
    pa, pc = shapley3(f_acc), shapley3(f_cost)
    da, dc = f_acc[(1, 1, 1)] - f_acc[(0, 0, 0)], f_cost[(1, 1, 1)] - f_cost[(0, 0, 0)]
    out = dict(acc=pa.tolist(), cost=pc.tolist(), d_acc=da, d_cost=dc)
    out['acc_shares'] = (pa / da).tolist() if abs(da) > 2 * d_acc_eps else None
    lo, hi = cost_interval
    out['cost_shares'] = (pc / dc).tolist() if (abs(dc) / abs(f_cost[(0, 0, 0)]) > 0.05 and (lo > 0 or hi < 0)) else None
    return out


def sign_label(lo, hi, what='error'):
    if lo > 0:
        return f'increases {what}'
    if hi < 0:
        return f'reduces {what}'
    return 'indistinguishable from zero'


# --------------------------------------------------------------------------------------------- elasticity (5, A1.8)
def elasticity(r, M, m, c):
    r, M, m, c = (np.asarray(a, float) for a in (r, M, m, c))
    if len(c) < 12 or not (np.all(np.isfinite(c)) and np.all(c > 0)):
        return dict(status='fit unavailable (cells/costs)')
    X = np.column_stack([np.ones(len(c)), np.log(r), np.log(M), np.log(m)])
    if np.linalg.matrix_rank(X) < 4 or np.linalg.cond(X) > 1e3:
        return dict(status='fit unavailable (rank/condition)', cond=float(np.linalg.cond(X)))
    beta, *_ = np.linalg.lstsq(X, np.log(c), rcond=None)
    return dict(status='ok', beta=beta.tolist(), cond=float(np.linalg.cond(X)))


# --------------------------------------------------------------------------------------------- H2q (6.5, A1.5)
def ties_or_beats(ea, ec, eta):
    """speed3d analyze.py:57-62: per-case endpoints aggregated (median, max) before comparing."""
    ea, ec, eta = np.asarray(ea), np.asarray(ec), np.asarray(eta)
    return bool(np.median(ea - eta) <= np.median(ec + eta)), bool(np.max(ea - eta) <= np.max(ec + eta))


def pick(C, keys):
    """speed3d analyze.py:208-215: cheapest; within 3 % cost the lower median, then the lower worst."""
    if not keys:
        return None
    keys = sorted(keys, key=lambda k: C[k]['cost'])
    best = C[keys[0]]['cost']
    tied = [k for k in keys if C[k]['cost'] <= 1.03 * best]
    return sorted(tied, key=lambda k: (C[k]['median'], C[k]['worst']))[0]


def h2q(C, comp, qualified, panel=None):
    """A1.5 precedence. C: key -> dict(cost, median, worst, e (case dict), eta (case dict), screen_valid, q1);
    qualified: set of numerically qualified keys; panel: confirmation dict(valid, median_ratio) or None."""
    c = C.get(comp)
    if c is None or not c.get('q1') or not c.get('screen_valid'):
        return dict(verdict='inconclusive (comparator unavailable)', winner=None)
    match = []
    for k in qualified:
        v = C.get(k)
        if v is None or not v.get('screen_valid'):
            continue
        ids = [j for j in v['e'] if j in c['e']]
        if not ids:
            continue
        med, wst = ties_or_beats([v['e'][j] for j in ids], [c['e'][j] for j in ids], [c['eta'][j] for j in ids])
        if med and wst:
            match.append(k)
    w = pick(C, match)
    if w is None:
        return dict(verdict='no qualified tie-or-beat configuration among the tested candidates (bounded search)', winner=None)
    if panel is None:
        return dict(verdict='selected (confirmation pending)', winner=w, n_matching=len(match))
    if not panel['valid']:
        return dict(verdict='inconclusive (panel invalid)', winner=w)
    S = 1. / panel['median_ratio']
    return dict(verdict='pass' if S >= 1 / .6 else 'fail (the selected candidate failed confirmation)', winner=w, S=S,
                n_matching=len(match))
