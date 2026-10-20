"""jcp-heads statistics and verdict procedures (DESIGN 5 as amended by A1.6, A2.1-A2.2, A3.1-A3.3, A4.1-A4.3, A5).
CPU only. Every function returns the registered label plus the numbers behind it. Wilcoxon / TOST / H-acc-r transitions
are P1's (p1stats, copied unchanged).

Error labels (paired, fixed population)      : error_label(eX, eY, eps)
Cost labels (case bootstrap of median ratio) : cost_label(ratio_per_case)
Dominance                                    : dominance(err, cost, qualified)
Q-LH-cost (equal-cost, at-most-cost, bracket): lh_cost(head, rungs)
Family-wise Holm over (K, r) components      : holm(pvals)
F5 lane reading                              : f5(...)
"""
from __future__ import annotations

import numpy as np

import p1stats as PS

ALPHA = .05
BAND = (1 / 1.1, 1.1)
NBOOT, SEED = 10000, 20261018


def margin(eps, eX, eY):
    return np.maximum(np.maximum(2. * np.asarray(eps, float), .1 * np.minimum(np.asarray(eX, float), np.asarray(eY, float))), 1e-12)


def error_label(eX, eY, eps):
    """A1.6: superiority first (one-sided signed rank, resolved coverage >= 50 %, median d < 0), else TOST equivalence on
    d / delta, else unresolved. 'X more accurate' / 'Y more accurate' / 'matches' / 'unresolved' / 'no population'."""
    eX, eY, eps = (np.asarray(x, float) for x in (eX, eY, eps))
    n = len(eX)
    if n == 0:
        return dict(label='no population', n=0, p_x=1., p_y=1.)
    d = eX - eY
    res_mask = np.abs(d) > 2. * eps
    cov = float(np.mean(res_mask))
    med = float(np.median(d))
    p_x = PS.wilcoxon(d, 'less')[0]
    p_y = PS.wilcoxon(d, 'greater')[0]
    t = PS.tost(d / margin(eps, eX, eY), ALPHA)
    out = dict(n=n, coverage=cov, median_d=med, p_x=p_x, p_y=p_y, tost=t, equivalent_flag=bool(t['equivalent']),
               worst_x=float(np.max(eX)), worst_y=float(np.max(eY)), median_x=float(np.median(eX)), median_y=float(np.median(eY)))
    if cov >= .5 and p_x < ALPHA and med < 0:
        out['label'] = 'X more accurate'
    elif cov >= .5 and p_y < ALPHA and med > 0:
        out['label'] = 'Y more accurate'
    elif t['equivalent']:
        out['label'] = 'matches'
    else:
        out['label'] = 'unresolved'
    out['superiority_predicates_x'] = bool(cov >= .5 and med < 0)
    return out


def cost_label(ratio, complete=True, nboot=NBOOT, seed=SEED):
    """ratio: per-case c_X / c_Y on the population. Case bootstrap (percentile 95 %) of the median.
    'X cheaper' (upper < 1/1.1), 'X costlier' (lower > 1.1), 'equal cost' (interval within [1/1.1, 1.1], ends included),
    'unresolved' otherwise, 'unknown' if timing is incomplete."""
    if not complete:
        return dict(label='unknown')
    r = np.asarray(ratio, float)
    if r.size == 0 or not np.isfinite(r).all() or np.any(r <= 0):
        return dict(label='unknown')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(nboot, len(r)))
    meds = np.median(r[idx], axis=1)
    lo, hi = float(np.quantile(meds, .025)), float(np.quantile(meds, .975))
    med = float(np.median(r))
    if hi < BAND[0]:
        lab = 'X cheaper'
    elif lo > BAND[1]:
        lab = 'X costlier'
    elif lo >= BAND[0] and hi <= BAND[1]:
        lab = 'equal cost'
    else:
        lab = 'unresolved'
    return dict(label=lab, median=med, lo=lo, hi=hi, n=len(r))


def dominance(err, cost, qualified=True):
    """A1.6 / A2.6 precedence: unqualified > no population > the rest."""
    if not qualified:
        return 'unqualified'
    e, c = err['label'], cost['label']
    if e == 'no population':
        return 'no population'
    x_err, y_err = e in ('X more accurate', 'matches'), e in ('Y more accurate', 'matches')
    x_cost, y_cost = c in ('X cheaper', 'equal cost'), c in ('X costlier', 'equal cost')
    if e == 'matches' and c == 'equal cost':
        return 'equivalent'
    if x_err and x_cost:
        return 'X dominates'
    if y_err and y_cost:
        return 'Y dominates'
    if (e == 'X more accurate' and c == 'X costlier') or (e == 'Y more accurate' and c == 'X cheaper'):
        return 'neither (trade-off)'
    return 'unresolved'


def holm(pvals, alpha=ALPHA):
    """pvals: {key: p}. Holm step-down; returns {key: rejected}."""
    keys = sorted(pvals, key=lambda k: (pvals[k], str(k)))
    m = len(keys)
    rej, stop = {}, False
    for i, k in enumerate(keys):
        if not stop and pvals[k] < alpha / (m - i):
            rej[k] = True
        else:
            stop = True
            rej[k] = False
    return rej


def lh_cost(head, rungs, rejected):
    """Q-LH-cost for one head size (A2.1 as amended by A3.1-A3.3, A4.3, A5).
    head: dict(qualified (bool, |S'_K| >= 2), complete (timing)).
    rungs: {r: dict(qualified, cost=cost_label(c_L / c_H), err=error_label(e_H, e_L, eps))} for every ladder rung
    (err with X = head, Y = linear). rejected: {r: Holm decision of H0_{K,r}} (family-wise, A3.1); REQUIRED (a missing
    rung counts as not rejected), so the positive label can never bypass the registered adjustment.
    Returns the equal-cost, at-most-cost and bracket results and cost-coverage status."""
    if not head.get('qualified') or not head.get('complete', True):
        return dict(status='unavailable (head unqualified or timing incomplete)', at_most_cost='unavailable', equal_cost=[],
                    certified_affordable=[], possibly_affordable=[])
    q = {r: v for r, v in rungs.items() if v['qualified']}
    lab = {r: v['cost']['label'] for r, v in q.items()}
    # cost labels are c_L / c_H: 'X cheaper' means the LINEAR rung is cheaper than the head
    cert_aff = sorted(r for r, l in lab.items() if l in ('X cheaper', 'equal cost'))
    eq = sorted(r for r, l in lab.items() if l == 'equal cost')
    possibly = sorted(r for r, l in lab.items() if l != 'X costlier')
    costlier = sorted(r for r, l in lab.items() if l == 'X costlier')
    equal_cost = [dict(r=r, dominance=_dom_head(q[r])) for r in eq]
    if not q:
        amc, reason = 'no possibly-affordable rung', 'no numerically qualified linear rung'
    elif not possibly:
        amc, reason = 'no possibly-affordable rung', 'every qualified rung certified costlier'
    else:
        reason = None
        beats_all = all(q[r]['err']['label'] == 'X more accurate' and q[r]['err']['superiority_predicates_x']
                        and bool(rejected.get(r, False)) for r in possibly)
        lin_ok = [r for r in cert_aff if q[r]['err']['label'] in ('Y more accurate', 'matches')]
        if beats_all:
            amc = 'head more accurate than every possibly-affordable rung'
        elif lin_ok:
            amc = f'linear at no more cost is at least as accurate (rung {lin_ok[0]}: {q[lin_ok[0]]["err"]["label"]})'
        else:
            amc = 'unresolved'
    ratio = lambda r: (q[r]['cost'].get('median', np.inf), r)
    lo = max(cert_aff, key=lambda r: (q[r]['cost'].get('median', -np.inf), -r)) if cert_aff else None
    hi = min(costlier, key=ratio) if costlier else None
    lowest_median = min(cert_aff, key=lambda r: (q[r]['err']['median_y'], r)) if cert_aff else None
    return dict(status='computed', at_most_cost=amc, at_most_cost_reason=reason, equal_cost=equal_cost or 'no equal-cost rung on the ladder',
                certified_affordable=cert_aff, possibly_affordable=possibly,
                coverage_status=None if cert_aff else 'no rung certified affordable',
                bracket=dict(r_lo=lo if lo is not None else 'no rung below', r_hi=hi if hi is not None else 'no rung above',
                             lo_dominance=_dom_head(q[lo]) if lo is not None else None,
                             hi_dominance=_dom_head(q[hi]) if hi is not None else None),
                lowest_median_affordable=lowest_median,
                unqualified_rungs=sorted(r for r, v in rungs.items() if not v['qualified']))


def _dom_head(v):
    """Dominance of head (X) against a linear rung (Y); the stored cost label is c_L / c_H, so swap its direction."""
    c = dict(v['cost'])
    c['label'] = {'X cheaper': 'X costlier', 'X costlier': 'X cheaper'}.get(c['label'], c['label'])
    return dominance(v['err'], c, True)


def f5(family, per_k):
    """A2.2 / A3.1 / A4.1: per_k[K] = dict(positive_components (bool), lh_dim (label, head = X), amc (at-most-cost label),
    qualified (|S'_K| >= 2), resolved_negative (bool)). Returns the lane outcome (exploratory validation evidence)."""
    pos = [K for K in family if per_k[K]['positive_components'] and per_k[K]['lh_dim'] in ('X dominates', 'equivalent', 'neither (trade-off)')]
    if pos:
        return dict(outcome='exploratory validation evidence of an accuracy-cost benefit of the head', K=pos)
    if family and all(per_k[K]['qualified'] and per_k[K]['resolved_negative'] for K in family):
        return dict(outcome='no benefit shown on validation evidence', K=list(family))
    return dict(outcome='inconclusive', K=[])
