"""jcp-hr-baselines statistics and the executable H-base decision table (DESIGN.md 9; A1.10-A1.12; A2.12, A2.17,
A2.19). CPU only, numpy.

Inputs per pair (X = ours, B = baseline) on the same eligible ST case-kinds:
    eX, eB  (n,) E0 errors;  eps (n,) eps_ref per case;  rho  dict case -> list of valid A-B-A ratios t_B / mean(t_A).
"""
from __future__ import annotations

import numpy as np

NBOOT = 10000
SEED = 0
TIME_EQ = np.log(1.10)


def envelope(e, eps):
    """Conditional reference-error envelopes (A1.10): worst and median of [max(e-eps,0), e+eps]."""
    e, eps = np.asarray(e, float), np.asarray(eps, float)
    lo, hi = np.maximum(e - eps, 0.), e + eps
    return dict(worst=(float(np.max(lo)), float(np.max(hi))), median=(float(np.median(lo)), float(np.median(hi))),
                worst_point=float(np.max(e)), median_point=float(np.median(e)))


def _boot_idx(n, nboot=NBOOT, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(nboot, n))


def time_stats(rho, nboot=NBOOT, seed=SEED):
    """rho: dict case -> list of ratios (cases are the resampling unit, with all their repeats).
    -> median ratio, 95 % and 90 % percentile intervals of the median ratio, and of median log ratio."""
    cases = sorted(rho)
    vals = [np.asarray(rho[c], float) for c in cases]
    allv = np.concatenate(vals)
    idx = _boot_idx(len(cases), nboot, seed)
    meds = np.array([np.median(np.concatenate([vals[i] for i in row])) for row in idx])
    lmeds = np.log(meds)
    return dict(median=float(np.median(allv)), ci95=(float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))),
                ci90_log=(float(np.percentile(lmeds, 5)), float(np.percentile(lmeds, 95))), n_cases=len(cases),
                n_values=int(len(allv)))


def mean_std_equiv(eX, eB, eps, nboot=NBOOT, seed=SEED):
    """A1.11 (i): x_i = (eB - eX) / delta_i, delta_i = max(2 eps_i, 0.1 eX_i) fixed per case; 90 % interval of mean x."""
    eX, eB, eps = map(lambda a: np.asarray(a, float), (eX, eB, eps))
    delta = np.maximum(2 * eps, 0.1 * eX)
    if np.any(delta <= 0):
        return None
    x = (eB - eX) / delta
    idx = _boot_idx(len(x), nboot, seed)
    m = x[idx].mean(1)
    lo, hi = float(np.percentile(m, 5)), float(np.percentile(m, 95))
    return dict(mean=float(x.mean()), ci90=(lo, hi), equivalent=bool(lo > -1 and hi < 1))


def worst_equiv(envX, envB, eps):
    """A2.12: difference envelope [L_B - U_X, U_B - L_X] inside (-dW, dW), dW = max(2 max eps, 0.1 L_X)."""
    LX, UX = envX['worst']
    LB, UB = envB['worst']
    dW = max(2 * float(np.max(eps)), 0.1 * LX)
    if dW <= 0:
        return dict(dW=dW, equivalent=False)
    return dict(dW=dW, diff=(LB - UX, UB - LX), equivalent=bool(LB - UX > -dW and UB - LX < dW))


def pair_label(eX, eB, eps, rho, min_cases=3):
    """The frozen decision table (A2.17). Returns dict(label, ...). Fail-closed on invalid inputs."""
    eX, eB, eps = map(lambda a: np.asarray(a, float), (eX, eB, eps))
    ok = (len(eX) > 0 and len(eX) == len(eB) == len(eps) and np.all(np.isfinite(eX)) and np.all(np.isfinite(eB))
          and np.all(np.isfinite(eps)) and np.all(eps >= 0))
    rho = {c: [v for v in vs if np.isfinite(v) and v > 0] for c, vs in (rho or {}).items()}
    rho = {c: vs for c, vs in rho.items() if vs}
    if not ok or len(rho) < min_cases:
        return dict(label='not assessed', reason='invalid inputs or fewer than %d timing cases' % min_cases)
    if np.any(np.maximum(2 * eps, 0.1 * eX) <= 0):
        return dict(label='not assessed', reason='zero equivalence margin')
    envX, envB = envelope(eX, eps), envelope(eB, eps)
    ts = time_stats(rho)
    if not all(np.isfinite(v) for v in (ts['median'],) + ts['ci95'] + ts['ci90_log']):
        return dict(label='not assessed', reason='non-finite timing statistics')
    b_faster, x_faster = ts['ci95'][1] < 1., ts['ci95'][0] > 1.
    b_better, x_better = envB['worst'][1] < envX['worst'][0], envX['worst'][1] < envB['worst'][0]
    ms = mean_std_equiv(eX, eB, eps)
    we = worst_equiv(envX, envB, eps)
    te = bool(ts['ci90_log'][0] > -TIME_EQ and ts['ci90_log'][1] < TIME_EQ)
    eq = bool(ms is not None and ms['equivalent'] and we['equivalent'] and te)
    if b_faster and b_better:
        lab = 'baseline dominates'
    elif x_faster and x_better:
        lab = 'ours dominates'
    elif (b_faster and x_better) or (x_faster and b_better):
        lab = 'neither dominates'
    elif eq:
        lab = 'matches'
    else:
        lab = 'unresolved'
    if lab in ('baseline dominates', 'ours dominates') and eq:
        lab += ' (within equivalence margins)'
    return dict(label=lab, time=ts, envX=envX, envB=envB, mean_std=ms, worst_eq=we, time_eq=te,
                b_faster=bool(b_faster), x_faster=bool(x_faster), b_better=bool(b_better), x_better=bool(x_better),
                qualifier='developmental, in-sample, unadjusted')


# ------------------------------------------------------------------------------------- selection helpers ---

def front(items):
    """items: list of dict(id, t, W). Non-dominated in (t, W) (both lower better); ties kept; sorted by id."""
    out = []
    for a in items:
        dom = any((b['t'] <= a['t'] and b['W'] <= a['W']) and (b['t'] < a['t'] or b['W'] < a['W']) for b in items)
        if not dom:
            out.append(a)
    return sorted(out, key=lambda z: z['id'])


def near_front(items, F, factor=1.5, cap=24):
    """A2.6: q near-front iff exists f in F with W_q <= 1.5 W_f and t_q <= 1.5 t_f; up to `cap`, alternately the fastest
    and the most accurate remaining (ties by id)."""
    fid = {f['id'] for f in F}
    cand = [q for q in items if q['id'] not in fid and any(q['W'] <= factor * f['W'] and q['t'] <= factor * f['t']
                                                           for f in F)]
    by_t = sorted(cand, key=lambda z: (z['t'], z['id']))
    by_w = sorted(cand, key=lambda z: (z['W'], z['id']))
    out, seen = [], set()
    i = j = 0
    while len(out) < min(cap, len(cand)):
        for lst, k in ((by_t, 'i'), (by_w, 'j')):
            if len(out) >= min(cap, len(cand)):
                break
            ptr = i if k == 'i' else j
            while ptr < len(lst) and lst[ptr]['id'] in seen:
                ptr += 1
            if ptr < len(lst):
                out.append(lst[ptr])
                seen.add(lst[ptr]['id'])
                ptr += 1
            if k == 'i':
                i = ptr
            else:
                j = ptr
    return out


def total_order_key(c):
    """A1.12 / A2.17 total order: worst, median, time, r, kind, HR size (fewer points first), scheme, dt, solver, id."""
    korder = {'none': 0, 'ecsw': 1, 'qdeim': 2}
    sorder = {'native': 0, 'common': 1}
    return (c['W'], c['med'], c['t'], c['r'], korder[c['kind']], c.get('hr_points', 0), c['scheme'], c['dt'],
            sorder[c['solver']], c['id'])


def picks(front_items, X):
    """Section 6 picks on the full-cohort Stage-B data. front_items: dict(id, t, W, med, r, kind, ...).
    X: dict(t, W). B_acc: fastest with W <= W_X (ties: smaller median, then total order). B_cost: smallest W with
    t <= t_X (ties: faster, then total order). Either may be None."""
    acc = [c for c in front_items if c['W'] <= X['W']]
    cost = [c for c in front_items if c['t'] <= X['t']]
    b_acc = min(acc, key=lambda c: (c['t'], c['med']) + total_order_key(c)) if acc else None
    b_cost = min(cost, key=lambda c: (c['W'], c['t']) + total_order_key(c)) if cost else None
    return b_acc, b_cost
