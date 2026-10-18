"""jcp-hr-baselines statistics and the executable H-base decision table (DESIGN.md 9; A1.10-A1.12; A2.12, A2.17,
A2.19). CPU only, numpy.

Inputs per pair (X = ours, B = baseline) on the same eligible ST case-kinds:
    eX, eB  (n,) E0 errors;  eps (n,) eps_ref per case;  rho  dict case -> list of valid A-B-A ratios t_B / mean(t_A).
"""
from __future__ import annotations

import numpy as np

NBOOT = 10000
SEED = 0
TIMING_CASES = (0, 1, 2, 3, 4, 5)        # registered timing cases (A2.19): 2D dev6 0-5, 3D 923801 0-5
TIME_EQ = np.log(1.10)


def envelope(e, eps):
    """Conditional reference-error envelopes (A1.10): worst and median of [max(e-eps,0), e+eps]."""
    e, eps = np.asarray(e, float), np.asarray(eps, float)
    lo, hi = np.maximum(e - eps, 0.), e + eps
    return dict(worst=(float(np.max(lo)), float(np.max(hi))), median=(float(np.median(lo)), float(np.median(hi))),
                worst_point=float(np.max(e)), median_point=float(np.median(e)))


def _boot_idx(n, nboot=NBOOT, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(nboot, n))


def case_values(rho):
    """A3.9: per timing case, the median of that case's valid ratios."""
    return {c: float(np.median(np.asarray(v, float))) for c, v in rho.items()}


def time_stats(rho, nboot=NBOOT, seed=SEED):
    """rho: dict case -> list of valid ratios. Estimand T = median over cases of the per-case medians (A3.9); the
    bootstrap resamples cases (10,000, seed 0). Returns T, the 95 % percentile interval of T, and the 90 % interval of
    log T (time equivalence, A1.11 iii)."""
    cases = sorted(rho)
    v = np.array([case_values(rho)[c] for c in cases])
    idx = _boot_idx(len(cases), nboot, seed)
    T = np.median(v[idx], axis=1)
    lT = np.log(T)
    return dict(median=float(np.median(v)), per_case=dict(zip(map(str, cases), v.tolist())),
                ci95=(float(np.percentile(T, 2.5)), float(np.percentile(T, 97.5))),
                ci90_log=(float(np.percentile(lT, 5)), float(np.percentile(lT, 95))), n_cases=len(cases),
                n_values=int(sum(len(rho[c]) for c in cases)))


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


def pair_label(eX, eB, eps, rho, timing_cases=TIMING_CASES):
    """The frozen decision table (A2.17). rho must hold valid ratios for EXACTLY the six registered timing cases
    (A3.9, A6.6) -- a caller-supplied shorter list is rejected; anything else -> 'not assessed'. Fail-closed."""
    if tuple(sorted(timing_cases)) != TIMING_CASES:
        return dict(label='not assessed', reason='timing cases are not the six registered ones')
    eX, eB, eps = map(lambda a: np.asarray(a, float), (eX, eB, eps))
    ok = (len(eX) > 0 and len(eX) == len(eB) == len(eps) and np.all(np.isfinite(eX)) and np.all(np.isfinite(eB))
          and np.all(np.isfinite(eps)) and np.all(eps >= 0))
    rho = {c: [v for v in vs if np.isfinite(v) and v > 0] for c, vs in (rho or {}).items()}
    rho = {c: vs for c, vs in rho.items() if vs}
    if not ok:
        return dict(label='not assessed', reason='invalid error inputs')
    if sorted(rho) != sorted(timing_cases):
        return dict(label='not assessed', reason='timing coverage incomplete (every registered timing case needs >= 1 '
                                                 'valid triple)', have=sorted(rho), want=sorted(timing_cases))
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


def picks(items, W_X):
    """A4.8 / A5.3 roles from the X-specific front. items: dict(id, ratio (median paired ratio to X over the 6 timing
    cases), W (worst error), med, r, kind, ...) for Q1-qualified configurations on the full cohort.
    Front = non-dominated in (ratio, W). B_acc: smallest ratio with W <= W_X (ties: smaller median, total order).
    B_cost: smallest W with ratio <= 1 (ties: smaller ratio, total order). Either may be None."""
    F = front([dict(c, t=c['ratio']) for c in items])
    acc = [c for c in F if c['W'] <= W_X]
    cost = [c for c in F if c['ratio'] <= 1.]
    key = lambda c: total_order_key(dict(c, t=c['ratio']))
    b_acc = min(acc, key=lambda c: (c['ratio'], c['med']) + key(c)) if acc else None
    b_cost = min(cost, key=lambda c: (c['W'], c['ratio']) + key(c)) if cost else None
    return b_acc, b_cost, F


def stage_b(items_abs, x_fronts, factor=1.5, cap=24):
    """A2.6 + A5.3 Stage-B membership: absolute-time front F, every X-specific front member (never capped), and up to
    `cap` near-front additions of F. items_abs: dict(id, t, W) on the screening subset. Returns sorted ids."""
    F = front(items_abs)
    near = near_front(items_abs, F, factor, cap)
    ids = {c['id'] for c in F} | {c['id'] for c in near}
    for fx in x_fronts:
        ids |= {c['id'] for c in fx}
    return sorted(ids)
