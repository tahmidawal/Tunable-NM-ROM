"""jcp-qualify statistics beyond P1's p1stats (copied unchanged): the frozen dominance rule and fronts (DESIGN 6, A1.7),
and the B4 observed-order classifier (DESIGN 5, A1.4, A3.5). Pure NumPy.
"""
from __future__ import annotations

import numpy as np

COST_BAND, REL_ERR = 1.05, .02


def tau(eX, eY, epsX, epsY):
    return max(2 * max(epsX, epsY), REL_ERR * max(eX, eY))


def dominates(X, Y):
    """X, Y: dict(c=cost, e=error statistic, eps=median eps_ref). DESIGN 6 (symmetric margins)."""
    t = tau(X['e'], Y['e'], X['eps'], Y['eps'])
    return bool(X['c'] <= COST_BAND * Y['c'] and X['e'] <= Y['e'] + t and
                (X['c'] < Y['c'] / COST_BAND or X['e'] < Y['e'] - t))


def front(points):
    """points: {name: dict(c, e, eps)} of CANDIDATES only. Non-dominated set, each tested against every original
    candidate (A1.7: no incremental pruning). Returns (members, {name: [dominators]})."""
    dom = {y: [x for x in points if x != y and dominates(points[x], points[y])] for y in points}
    return sorted([y for y in points if not dom[y]], key=lambda k: (points[k]['c'], points[k]['e'])), dom


# ------------------------------------------------------------------------------------------------- B4 order
def order_case(e, d_anchor, d_solver, d_quad, d_finer, valid, hs=(1., .5, .25)):
    """One case (DESIGN 5 + A1.4 + A3.5). e: {h: error vs the same-form anchor}; d_anchor = d(h_a, h_a/2);
    d_solver = d(tight, tighter) at the finest step; d_quad = A1.7 vector contamination at the finest step;
    d_finer = d(q+, q++) at the finest step. Returns dict(p, pairwise, interpretable, checks)."""
    es = np.array([e.get(h, np.nan) for h in hs], float)
    if not (np.isfinite(es).all() and np.all(es > 0)):
        return dict(p=None, pairwise=None, interpretable=False, checks=dict(defined=False))
    lh = np.log(np.array(hs))
    p = float(np.polyfit(lh, np.log(es), 1)[0])
    pw = [float(np.log(es[i] / es[i + 1]) / np.log(hs[i] / hs[i + 1])) for i in range(len(hs) - 1)]
    diff = es[-2] - es[-1]
    checks = dict(defined=bool(diff > 0), asymptotic=abs(pw[0] - pw[-1]) <= .3,
                  anchor=bool(np.isfinite(d_anchor) and d_anchor <= .1 * es[-1]),
                  solver=bool(diff > 0 and np.isfinite(d_solver) and d_solver <= .1 * diff),
                  quadrature=bool(diff > 0 and np.isfinite(d_quad) and d_quad <= .1 * diff),
                  finer_rule=bool(np.isfinite(d_finer) and d_finer <= 2.5e-4), valid=bool(valid))
    return dict(p=p, pairwise=pw, interpretable=all(checks.values()), checks=checks)


def order_label(case_results, n_required=5):
    """Label per (scheme, r) from the per-case results (DESIGN 5, A1.4)."""
    ok = [r for r in case_results if r['interpretable']]
    if len(ok) < n_required:
        return dict(label='order not established (not interpretable)', interpretable=len(ok), p_median=None, p_range=None)
    ps = np.array([r['p'] for r in ok])
    lab = 'observed order, no integer label'
    if np.all((ps >= .8) & (ps <= 1.2)):
        lab = 'order 1'
    elif np.all((ps >= 1.8) & (ps <= 2.2)):
        lab = 'order 2 (finite-range observed order over dt0..dt0/4)'
    return dict(label=lab, interpretable=len(ok), p_median=float(np.median(ps)), p_range=[float(ps.min()), float(ps.max())])
