"""jcp-p1-scaling statistics and verdict procedures (DESIGN 4 as amended by A1.9, A2.1; AM-4, AM-7). CPU only.

Every function returns its outcome plus the numbers behind it. Outcomes are exactly the registered labels.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import optimize, stats

SEED = 20261017
NBOOT = 10000


# ------------------------------------------------------------------ Wilcoxon signed-rank (A1.9) ----

def _exact_null_safe(n):
    m = n * (n + 1) // 2
    c = np.zeros(m + 1)
    c[0] = 1.
    for k in range(1, n + 1):
        new = c.copy()
        new[k:] += c[:m + 1 - k]
        c = new
    return c / c.sum()


def wilcoxon(x, alternative):
    """Signed-rank test of x against 0. Zeros dropped ('wilcox'); exact null when n <= 50 and no ties among |x|,
    else normal approximation with tie-corrected variance and no continuity correction. n = 0 -> p = 1 (no rejection).
    alternative in {'greater', 'less', 'two-sided'}. Returns (p, n, W+, method)."""
    x = np.asarray(x, float)
    x = x[x != 0]
    n = len(x)
    if n == 0:
        return 1., 0, 0., 'empty'
    ax = np.abs(x)
    ranks = stats.rankdata(ax)
    wplus = float(np.sum(ranks[x > 0]))
    ties = len(np.unique(ax)) < n
    if n <= 50 and not ties:
        dist = _exact_null_safe(n)
        w = int(round(wplus))
        p_ge = float(dist[w:].sum())
        p_le = float(dist[:w + 1].sum())
        if alternative == 'greater':
            p = p_ge
        elif alternative == 'less':
            p = p_le
        else:
            p = min(1., 2 * min(p_ge, p_le))
        return p, n, wplus, 'exact'
    mu = n * (n + 1) / 4
    _, cnt = np.unique(ax, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - float(np.sum(cnt ** 3 - cnt)) / 48
    z = (wplus - mu) / math.sqrt(var) if var > 0 else 0.
    if alternative == 'greater':
        p = float(stats.norm.sf(z))
    elif alternative == 'less':
        p = float(stats.norm.cdf(z))
    else:
        p = float(min(1., 2 * stats.norm.sf(abs(z))))
    return p, n, wplus, 'approx'


def tost(d, alpha=.05):
    """Normalised signed-rank equivalence (A1.9): d_i + 1 > 0 ('greater') and d_i - 1 < 0 ('less')."""
    d = np.asarray(d, float)
    p_lo = wilcoxon(d + 1., 'greater')[0]
    p_hi = wilcoxon(d - 1., 'less')[0]
    return dict(p_lower=p_lo, p_upper=p_hi, equivalent=bool(p_lo < alpha and p_hi < alpha))


# ------------------------------------------------------------------ H-acc-N (4.3, A1.9) ----

def margin(eps, e1024):
    return np.maximum(np.maximum(2. * np.asarray(eps, float), .1 * np.asarray(e1024, float)), 1e-12)


def h_acc_n(pairs, slopes, qualified=True, alpha=.05):
    """pairs: {'P256': (e_N, e_1024, eps), 'P4096': (...)} on each pair's eligible population (arrays; may be absent);
    slopes: per-case OLS slopes of e vs log2 N over the three meshes (16-case three-mesh eligible intersection) or None.
    Returns the registered outcome and its numbers."""
    res = dict(pairs={}, trend=None)
    for k, v in pairs.items():
        eN, e1, eps = (np.asarray(x, float) for x in v)
        n = len(eN)
        if n == 0:
            res['pairs'][k] = dict(n=0, equivalent=False, within=0., empty=True)
            continue
        d = (eN - e1) / margin(eps, e1)
        t = tost(d, alpha)
        res['pairs'][k] = dict(n=n, d=d.tolist(), within=float(np.mean(np.abs(d) <= 1.)), empty=False, **t)
    if slopes is not None:
        s = np.asarray(slopes, float)
        nz = int(np.sum(s != 0))
        if nz < 6:
            res['trend'] = dict(testable=False, n_nonzero=nz)
        else:
            p = wilcoxon(s, 'two-sided')[0]
            res['trend'] = dict(testable=True, n_nonzero=nz, p=p, rejects=bool(p < alpha),
                                median_slope=float(np.median(s)))
    P_ = res['pairs']
    if set(P_) != {'P256', 'P4096'}:               # registered verdict needs both pairs (three meshes)
        out = 'two-mesh evidence, registered three-mesh verdict not reached'
    elif any(v['empty'] for v in P_.values()):
        out = 'inconclusive'
    else:
        all_eq = all(v['equivalent'] for v in P_.values())
        within_ok = all(v['within'] >= .9 for v in P_.values())
        any_fail_eq = any(not v['equivalent'] for v in P_.values())
        trend_rej = bool(res['trend'] and res['trend'].get('rejects'))
        any_low = any(v['within'] < .5 for v in P_.values())
        if all_eq and within_ok:
            out = 'pass'
        elif any_fail_eq and (trend_rej or any_low):
            out = 'fail'
        else:
            out = 'inconclusive'
    res['procedure_outcome'] = out
    res['outcome'] = out if qualified else 'inconclusive (unqualified configuration)'
    return res


# ------------------------------------------------------------------ H-acc-r (4.4, AM-7) ----

def h_acc_r_transition(de, eps, qualified_lo, qualified_hi):
    """de = e(r_hi) - e(r_lo) per eligible case; qualification per case on each rung."""
    de = np.asarray(de, float)
    eps = np.asarray(eps, float)
    n = len(de)
    if n == 0:
        return dict(outcome='inconclusive', reason='empty population', n=0)
    qual = np.asarray(qualified_lo, bool) & np.asarray(qualified_hi, bool)
    unq = int(np.sum(~qual))
    base = dict(n=n, unqualified=unq, improve=int(np.sum(qual & (de < -2 * eps))), worsen=int(np.sum(qual & (de > 2 * eps))),
                tie=int(np.sum(qual & (np.abs(de) <= 2 * eps))), worst_change=float(np.max(de)) if n else None,
                median_change_all=float(np.median(de)))
    if unq > .25 * n:
        return dict(base, outcome='inconclusive', reason='> 25 % lack qualification')
    res_mask = qual & (np.abs(de) > 2 * eps)
    nr = int(np.sum(res_mask))
    base['resolved'] = nr
    if nr < .5 * n:
        return dict(base, outcome='inconclusive', reason='resolved < 50 %')
    frac = float(np.mean(de[res_mask] < 0))
    med = float(np.median(de[res_mask]))
    base.update(improve_fraction=frac, median_change_resolved=med)
    if frac > .5 and med < 0:
        return dict(base, outcome='improves')
    return dict(base, outcome='does not improve')


def h_acc_r_range(outcomes):
    if outcomes and all(o == 'improves' for o in outcomes):
        return 'monotone'
    if any(o == 'does not improve' for o in outcomes):
        return 'not monotone'
    return 'inconclusive'


# ------------------------------------------------------------------ H-cost-N (4.1, AM-4, A1.9) ----

def h_cost_n(ratio, valid, meshes, nboot=NBOOT, seed=SEED, bar=1.25, registered=3):
    """ratio, valid: arrays (n_cases, n_meshes, n_reps) of B/A ratios and triplet validity for ONE subject family
    (r, scheme) across meshes. Returns the registered outcome."""
    ratio = np.asarray(ratio, float)
    valid = np.asarray(valid, bool)
    nc, nm, nr = ratio.shape
    if nm != len(meshes) or nm < 2:
        return dict(outcome='verdict not reached', reason='missing mesh')
    two_mesh = nm < registered                    # the registered verdict needs all three meshes (V4-3)
    n_inv = (~valid).sum(axis=(0, 2))
    inv_frac = n_inv / (nc * nr)
    if np.any(3 * n_inv > nc * nr):                     # > 1/3 invalid, in integer arithmetic
        return dict(outcome='verdict not reached', reason='panel invalid (> 1/3 invalid triplets on a mesh)',
                    invalid_fraction=inv_frac.tolist())

    def stat(idx):
        med = []
        for j in range(nm):
            v = ratio[idx, j][valid[idx, j]]
            if v.size == 0:
                return None, None
            med.append(float(np.median(v)))
        return max(med) / min(med), med

    R, med = stat(np.arange(nc))
    rng = np.random.default_rng(seed)
    bs, undef = [], 0
    for _ in range(nboot):
        s_, _m = stat(rng.integers(0, nc, nc))
        if s_ is None:
            undef += 1
        else:
            bs.append(s_)
    base = dict(R=R, medians=dict(zip([str(m) for m in meshes], med)), undefined_fraction=undef / nboot,
                invalid_fraction=inv_frac.tolist())
    if undef / nboot > .025:
        return dict(base, outcome='inconclusive (insufficient valid timings)')
    lo, hi = np.percentile(bs, [2.5, 97.5])
    base.update(ci=[float(lo), float(hi)])
    if two_mesh:
        proc = 'pass' if hi <= bar else ('fail' if lo > bar else 'inconclusive')
        return dict(base, outcome='two-mesh evidence, registered three-mesh verdict not reached', procedure_outcome=proc)
    if hi <= bar:
        return dict(base, outcome='pass')
    if lo > bar:
        return dict(base, outcome='fail')
    return dict(base, outcome='inconclusive')


# ------------------------------------------------------------------ H-cost-r (4.2, A1.9, A2.1) ----

def fit_power(r, t, measured=True):
    """t = a + exp(beta) r^p (A2.1). Returns dict(ok, boundary, flat, a, b, p, rel_rms, status)."""
    r = np.asarray(r, float)
    t = np.asarray(t, float)
    out = dict(ok=False, boundary=False, flat=False)
    if measured and t.max() / t.min() < 1.10:
        return dict(out, flat=True)
    a0 = .5 * t.min() if t.min() > 0 else 0.
    y = np.log(np.maximum(t - a0, 1e-300))
    p0, lb0 = np.polyfit(np.log(r), y, 1)
    p0 = float(np.clip(p0, .05, 3.95))

    def f(x):
        a, beta, p = x
        return (a + np.exp(beta) * r ** p - t) / t
    try:
        sol = optimize.least_squares(f, x0=[a0, lb0, p0], bounds=([0., -np.inf, 0.], [np.inf, np.inf, 4.]), x_scale='jac')
    except Exception as e_:  # noqa: BLE001
        return dict(out, error=str(e_))
    a, beta, p = sol.x
    b = float(np.exp(beta))
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    ok = bool(sol.status > 0 and np.all(np.isfinite(sol.x)) and rms <= .2)
    bnd = bool(p <= 1e-6 or p >= 4 - 1e-6 or b * (r.max() ** p - r.min() ** p) < .01 * a) if measured else False
    return dict(ok=ok, boundary=bnd, flat=False, a=float(a), b=b, p=float(p), rel_rms=rms, status=int(sol.status))


def h_cost_r(ranks, times_by_case, p_decl, flops=None, bytes_=None, nboot=NBOOT, seed=SEED):
    """times_by_case: array (n_cases, n_fix_per_case, n_ranks) of fixture medians. Bootstrap resamples cases."""
    ranks = np.asarray(ranks, float)
    T = np.asarray(times_by_case, float)
    nc = T.shape[0]
    point = fit_power(ranks, T.reshape(-1, T.shape[-1]).mean(0))
    res = dict(point=point, p_decl=p_decl)
    if point.get('flat'):
        res['outcome'] = 'inconclusive'
        res['reason'] = 'cost flat in r (exponent unidentified)'
    elif not point['ok'] or point['boundary']:
        res['outcome'] = 'inconclusive'
        res['reason'] = 'fit failure or boundary solution'
    else:
        rng = np.random.default_rng(seed)
        ps, bad = [], 0
        for _ in range(nboot):
            idx = rng.integers(0, nc, nc)
            ft = fit_power(ranks, T[idx].reshape(-1, T.shape[-1]).mean(0))
            if ft.get('flat') or not ft['ok'] or ft['boundary']:
                bad += 1
            else:
                ps.append(ft['p'])
        res['bad_fraction'] = bad / nboot
        if bad / nboot > .05:
            res['outcome'] = 'inconclusive'
            res['reason'] = '> 5 % bootstrap fits failing or at the boundary'
        else:
            lo, hi = np.percentile(ps, [2.5, 97.5])
            res['ci'] = [float(lo), float(hi)]
            res['half_width'] = float((hi - lo) / 2)
            if lo <= p_decl <= hi and (hi - lo) / 2 <= .5:
                res['outcome'] = 'pass'
            elif not (lo <= p_decl <= hi):
                res['outcome'] = 'fail'
            else:
                res['outcome'] = 'inconclusive'
                res['reason'] = 'interval half-width > 0.5'
    if flops is not None:
        pf = fit_power(ranks, flops, measured=False)['p']
        pb = fit_power(ranks, bytes_, measured=False)['p']
        res['p_flops'], res['p_bytes'] = pf, pb
        if res.get('outcome') == 'fail':
            res['secondary'] = 'operation-model' if (abs(pf - p_decl) > .25 or abs(pb - p_decl) > .25) else 'hardware scaling'
    return res


# ------------------------------------------------------------------ Q3 ----

def conditioning(svals, tol=1e-13):
    s = np.sort(np.asarray(svals, float))[::-1]
    rank = int(np.sum(s > s[0] * tol))
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float('inf')
    return dict(rank=rank, n=len(s), deficient=bool(rank < len(s)), cond=cond)
