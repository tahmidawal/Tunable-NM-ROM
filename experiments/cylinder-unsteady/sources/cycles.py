"""Shedding-cycle statistics shared by the reference stepper and all reports.

Cycle definition follows the DFG 2D-2 page: from one lift minimum to the next.
Minima are refined by a parabola through the three neighbouring samples.
"""
import numpy as np


def lift_minima(t, cl):
    t, cl = np.asarray(t, float), np.asarray(cl, float)
    out = []
    for i in range(1, len(cl) - 1):
        if cl[i] < cl[i - 1] and cl[i] <= cl[i + 1]:
            y0, y1, y2 = cl[i - 1], cl[i], cl[i + 1]
            h = t[i + 1] - t[i]
            den = y0 - 2 * y1 + y2
            shift = 0.5 * h * (y0 - y2) / den if den != 0 else 0.0
            out.append((i, t[i] + float(np.clip(shift, -h, h))))
    return out


def extremum(t, y, w, sign):
    """Parabola-refined extreme value of y on index window w (sign=+1 max, -1 min)."""
    idx = np.arange(len(y))[w]
    i = idx[np.argmax(sign * y[w])]
    if i <= 0 or i >= len(y) - 1:
        return float(y[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if den == 0:
        return float(y1)
    s = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
    return float(y1 - 0.25 * (y0 - y2) * s)


def cycles(t, cd, cl, dp=None):
    """Return one record per complete cycle (lift minimum to lift minimum)."""
    t, cd, cl = (np.asarray(a, float) for a in (t, cd, cl))
    keep = np.isfinite(t) & np.isfinite(cd) & np.isfinite(cl)
    if dp is not None:
        dp = np.asarray(dp, float)
        keep &= np.isfinite(dp)
        dp = dp[keep]
    t, cd, cl = t[keep], cd[keep], cl[keep]
    mins = lift_minima(t, cl)
    recs = []
    for (i0, t0), (i1, t1) in zip(mins[:-1], mins[1:]):
        w = slice(i0, i1 + 1)
        period = t1 - t0
        rec = {'t0': float(t0), 't1': float(t1), 'period': float(period), 'St': float(0.1 / period),
               'CD_max': extremum(t, cd, w, 1), 'CD_min': extremum(t, cd, w, -1),
               'CL_max': extremum(t, cl, w, 1), 'CL_min': extremum(t, cl, w, -1),
               'CD_max_sample': float(cd[w].max()), 'CL_max_sample': float(cl[w].max())}
        if dp is not None:
            dp = np.asarray(dp, float)
            # Featflow convention: t0 at the C_L minimum.
            rec['dp_at_t0_plus_half_period'] = float(np.interp(t0 + 0.5 * period, t, dp))
            # Schaefer-Turek 1996 convention: t0 at the C_L maximum (parabola-refined time).
            j = i0 + int(np.argmax(cl[w]))
            tmax = t[j]
            if 0 < j < len(cl) - 1:
                y0, y1, y2 = cl[j - 1], cl[j], cl[j + 1]; den = y0 - 2 * y1 + y2; h = t[j + 1] - t[j]
                if den != 0:
                    tmax = t[j] + float(np.clip(0.5 * h * (y0 - y2) / den, -h, h))
            te = tmax + 0.5 * period   # no extrapolation: outside the history -> NaN (excluded)
            rec['dp_1996_t0_at_CLmax_plus_half_period'] = float(np.interp(te, t, dp)) if te <= t[-1] else float('nan')
        recs.append(rec)
    return recs


def summarize(recs):
    """R1d aggregate (G5-A5): mean over every complete cycle, with per-cycle spread."""
    keys = [k for k in recs[0] if k not in ('t0', 't1')]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in recs], float); v = v[np.isfinite(v)]
        out[k] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()), 'count': int(v.size)}
    return out | {'cycles': len(recs)}


def periodic(recs, n=3, cl_tol=1e-4, cd_rel_tol=1e-4, period_rel_tol=1e-4):
    """Predeclared developed-flow rule: the last n consecutive cycle pairs agree."""
    if len(recs) < n + 1:
        return False
    if not all(np.isfinite(list(r.values())).all() for r in recs[-n - 1:]):
        return False
    for a, b in zip(recs[-n - 1:-1], recs[-n:]):
        if abs(b['CL_max'] - a['CL_max']) > cl_tol or abs(b['CL_min'] - a['CL_min']) > cl_tol:
            return False
        if abs(b['CD_max'] - a['CD_max']) > cd_rel_tol * abs(a['CD_max']):
            return False
        if abs(b['period'] - a['period']) > period_rel_tol * a['period']:
            return False
    return True
