"""imp-faststep frozen selection and labelling rules (pure Python; used in-job and by the offline checker, so the two
cannot diverge). DESIGN A1.8, A2.2, A2.6, A3.1, A3.4, A3.6, A4.1.

Records are plain dicts: per subject a list of per-case rows {case, valid, stable, e} (e = E0 ST error, inf if missing).
"""
from __future__ import annotations

import math

BAND = (1 / 1.1, 1.1)
TIE = 1.01


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float('nan')
    return xs[n // 2] if n % 2 else .5 * (xs[n // 2 - 1] + xs[n // 2])


def worst(rows, cases):
    """Worst E0 error over `cases`; a missing or non-finite error -> +inf (A1.7: nothing passes vacuously)."""
    by = {r['case']: r for r in rows}
    w = 0.
    for c in cases:
        r = by.get(c)
        e = float('inf') if r is None or r.get('e') is None else float(r['e'])
        if not math.isfinite(e):
            return float('inf')
        w = max(w, e)
    return w


def ok_on(rows, cases):
    by = {r['case']: r for r in rows}
    return all(c in by and bool(by[c].get('valid')) and bool(by[c].get('stable')) for c in cases)


def case_ratio_stats(trips, cases, reps=3):
    """A3.6: per case the median over its valid repetitions of the triplet ratio, then the median over cases. Every case
    needs `reps` valid repetitions, else 'incomplete'. Any identity failure -> 'failed (identity)'."""
    if any(not t.get('identity_ok', True) for t in trips):
        return dict(status='failed (identity)', value=None)
    per = {}
    for c in cases:
        v = [t['ratio'] for t in trips if t['case'] == c and t.get('valid')]
        if len(v) < reps:
            return dict(status='incomplete', value=None, missing_case=c, valid_reps=len(v))
        per[c] = median(v[:reps])
    return dict(status='ok', value=median(list(per.values())), per_case=per)


def pick_min(cands):
    """A3.4 tie rule: cands = [(id, t, dev_worst)]; tie set {t <= 1.01 t*}; then lower dev worst, then id."""
    if not cands:
        return None
    ts = min(c[1] for c in cands)
    tie = [c for c in cands if c[1] <= TIE * ts]
    return sorted(tie, key=lambda c: (c[2], c[0]))[0][0]


def matched_competitor(X_rows, pool, screen, dev):
    """B*(X), A1.8/A3.4: pool {id: rows}; screen {id: dev6 median fused ratio to the common anchor (None = unusable)}.
    Members valid and stable on every dev case with dev worst <= X's dev worst; the lowest screening time wins."""
    wx = worst(X_rows, dev)
    cands = []
    for mid, rows in pool.items():
        t = screen.get(mid)
        if t is None or not ok_on(rows, dev):
            continue
        wm = worst(rows, dev)
        if wm <= wx:
            cands.append((mid, t, wm))
    return pick_min(cands)


def match_status(X_rows, B_rows, populations):
    """A3.4: per population, matched iff B valid + stable on every case there and B's worst <= X's worst there."""
    out = {}
    for name, cases in populations.items():
        out[name] = bool(ok_on(B_rows, cases) and worst(B_rows, cases) <= worst(X_rows, cases))
    return out


def pod_successors(members):
    """A2.2/A3.1: members = ordered [(id, rule_id)] of one (r, dt) ending with ('...none...', 'none'). Successor of a
    member = the next member with a distinct rule id; the last distinct rule -> none."""
    succ = {}
    for i, (mid, h) in enumerate(members):
        if h == 'none':
            continue
        nxt = members[-1][0]
        for mid2, h2 in members[i + 1:]:
            if h2 == 'none' or h2 != h:
                nxt = mid2
                break
        succ[mid] = nxt
    return succ


def classify_growth(g8, g16, refined_ok=True):
    """A4.1 exhaustive classifier (diagnostic only)."""
    if not refined_ok or not (math.isfinite(g8) and math.isfinite(g16)) or abs(g8 - g16) > .005:
        return 'unresolved'
    if g8 > 1.055 and g16 > 1.055:
        return 'growth present in the tested refinements'
    if g8 < 1.045 and g16 < 1.045:
        return 'growth absent in the tested refinements'
    return 'unresolved'
