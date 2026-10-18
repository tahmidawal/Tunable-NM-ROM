"""jcp-references: order verification, Richardson uncertainty and case outcome (DESIGN §4, A1.5, A1.8, A2.1, A2.2,
A3.1, A3.2). Pure NumPy, no JAX: used by the job, the collector and the loader; audit_ref.py re-implements it
independently.

Fields are lattice arrays of shape (6, ...) (2D (6, 257, 257); 3D (6, 63^3)); norms are Euclidean over the lattice.
Outputs t_k, k >= 1 only; n0 = ||canonical t0 field||.
"""
from __future__ import annotations

import math

import numpy as np

P_LO, P_HI = 1.8, 2.2          # order gate (§4 b)
C_MAX = 0.5                    # direction consistency (A3.1)
PECLET_MAX = 2.0               # cell Peclet (§4 c), running max |u| at the coarsest level
FLOOR_STAB = 10.0              # tau_L = FLOOR_STAB * sum_n max(r_n, R_FLOOR)  (A3.2)
R_FLOOR = 1e-14
FLOOR_MARGIN = 10.0            # d_ij / n0 >= FLOOR_MARGIN * (tau_i + tau_j)
ACCEPT_RESIDUAL = 2e-11        # per-step acceptance (§3.1)
P_BRACKET = (0.25, 8.0)
OUTCOMES = ('accepted', 'ic-mismatch', 'solver-fail', 'unresolved-floor', 'peclet-fail', 'order-fail',
            'direction-inconsistent')     # precedence order of the failures (A5.2)
METRIC_ID = {2: '2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1',
             3: '3|lattice=64|times=0:0.05:0.25|norm=fixed_t0|max_k>=1'}


def tnorm(a):
    """Per-time Euclidean norm over the lattice: (6, ...) -> (6,)."""
    a = np.asarray(a, np.float64)
    return np.sqrt(np.sum(a.reshape(a.shape[0], -1) ** 2, axis=1))


def ratio_fn(p, h1, h2, h3):
    return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p)


def is_dyadic(h1, h2, h3):
    return abs(h1 / h2 - 2.0) < 1e-12 and abs(h2 / h3 - 2.0) < 1e-12


def solve_order(d12, d23, h1, h2, h3):
    """Observed order from two adjacent differences; None if undefined (zero, non-finite, unbracketed). Never clipped."""
    if not (np.isfinite(d12) and np.isfinite(d23)) or d12 <= 0 or d23 <= 0:
        return None
    r = d12 / d23
    lo, hi = P_BRACKET
    if is_dyadic(h1, h2, h3):
        p = math.log2(r)
        return p if lo <= p <= hi else None          # same bracket policy as the non-uniform solve (A1.5)
    flo, fhi = ratio_fn(lo, h1, h2, h3), ratio_fn(hi, h1, h2, h3)
    if not (flo <= r <= fhi):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ratio_fn(mid, h1, h2, h3) < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rich_factor(p, h2, h3):
    """Richardson multiplier: error(u3) ~ (u3 - u2) * h3^p / (h2^p - h3^p); p is bracketed in [0.25, 8] by
    solve_order, so the denominator is positive."""
    den = h2 ** p - h3 ** p
    if not (np.isfinite(den) and den > 0):
        raise ValueError(f'invalid Richardson denominator for p={p}')
    return h3 ** p / den


def floor_proxy(res, n_stages):
    r = np.asarray(res, np.float64)[:n_stages]
    if not np.all(np.isfinite(r)):
        return float('inf')
    return float(FLOOR_STAB * np.sum(np.maximum(r, R_FLOOR)))


def steps_ok(res, n_stages, accept=ACCEPT_RESIDUAL):
    r = np.asarray(res, np.float64)[:n_stages]
    return bool(len(r) == n_stages and np.all(np.isfinite(r)) and np.all(r <= accept))


def judge(levels, t0, nu):
    """levels: three dicts (coarse -> fine) with keys L, lat (6, ...), tau (floor proxy), steps_ok, umax.
    t0: canonical lattice t0 field. Returns the full per-time record and the outcome (precedence order of OUTCOMES)."""
    assert len(levels) == 3
    h = [1.0 / lv['L'] for lv in levels]
    assert h[0] > h[1] > h[2]
    u1, u2, u3 = (np.asarray(lv['lat'], np.float64) for lv in levels)
    n0 = float(np.linalg.norm(np.asarray(t0, np.float64).ravel()))
    t0_dev = max(float(np.linalg.norm((u[0] - t0).ravel())) / n0 for u in (u1, u2, u3))
    d12, d23 = tnorm(u1 - u2), tnorm(u2 - u3)
    NT = u1.shape[0]                      # jcp-budget: number of output times (6 at T = 0.25)
    p_t, eps_t, c_t = [], [], []
    for k in range(1, NT):
        p = solve_order(d12[k], d23[k], *h)
        p_t.append(p)
        if p is None:
            eps_t.append(None)
            c_t.append(None)
            continue
        eps_t.append(float(d23[k] / n0 * rich_factor(p, h[1], h[2])))
        rho = ratio_fn(p, *h)
        c_t.append(float(np.linalg.norm(((u1[k] - u2[k]) - rho * (u2[k] - u3[k])).ravel()) / d12[k]))
    tau = [float(lv['tau']) for lv in levels]
    peclet = float(levels[0]['umax'] * h[0] / nu)
    checks = dict(
        solver=all(lv['steps_ok'] for lv in levels),
        peclet=bool(np.isfinite(peclet) and peclet < PECLET_MAX),
        floor=bool(all(d12[k] / n0 >= FLOOR_MARGIN * (tau[0] + tau[1]) and
                       d23[k] / n0 >= FLOOR_MARGIN * (tau[1] + tau[2]) for k in range(1, NT))),
        order=bool(all(p is not None and P_LO <= p <= P_HI for p in p_t)),
        direction=bool(all(c is not None and c <= C_MAX for c in c_t)),
        t0_shared=bool(t0_dev <= 1e-13))
    if not checks['t0_shared']:
        outcome = 'ic-mismatch'
    elif not checks['solver']:
        outcome = 'solver-fail'
    elif not checks['floor']:
        outcome = 'unresolved-floor'
    elif not checks['peclet']:
        outcome = 'peclet-fail'
    elif not checks['order']:
        outcome = 'order-fail'
    elif not checks['direction']:
        outcome = 'direction-inconsistent'
    else:
        outcome = 'accepted'
    eps_ok = [e for e in eps_t if e is not None]
    return dict(levels=[lv['L'] for lv in levels], h=h, n0=n0, t0_dev=t0_dev,
                d12=(d12[1:] / n0).tolist(), d23=(d23[1:] / n0).tolist(), p=p_t, eps_t=eps_t,
                eps_ref=(max(eps_ok) if len(eps_ok) == NT - 1 else None), c=c_t, tau=tau, peclet=peclet,
                umax=[float(lv['umax']) for lv in levels], checks=checks, outcome=outcome)


def richardson_field(u2, u3, p_t, h2, h3):
    """Diagnostic extrapolated field (no uncertainty attached): u3 + (u3 - u2) * h3^p/(h2^p - h3^p), per time."""
    out = np.array(u3, np.float64, copy=True)
    for k in range(1, out.shape[0]):
        p = p_t[k - 1]
        if p is None:
            out[k] = np.nan
        else:
            out[k] = u3[k] + (u3[k] - u2[k]) * rich_factor(p, h2, h3)
    return out


def rel_err(field, ref, n0):
    """Campaign metric: per-time ||f - R|| / n0 (k >= 1) and its max."""
    e = tnorm(np.asarray(field, np.float64) - np.asarray(ref, np.float64))[1:] / n0
    return float(np.max(e)), e.tolist()


# ----------------------------------------------------------------------------------------- cross-check (A2.4) --

PS_Q_MIN = 0.5          # A10.1 (was 2.5 under A1.7)
PS_ROUNDOFF = 1e-12     # A10.1 round-off level of the PS self-difference (relative to n0)


def ps_spatial_indicator(da, db, Ns, n0):
    """A10.1: per-time spatial error indicator of the finest PS level; returns (s_x or None, q list, ok, old_rule)."""
    hs = [1.0 / N for N in Ns]
    q, ind, ok = [], [], True
    NT = len(da)
    for k in range(1, NT):
        qk = solve_order(da[k], db[k], *hs)
        q.append(qk)
        if not np.isfinite(db[k]):
            ok = False
            continue
        if db[k] / n0 <= PS_ROUNDOFF:                   # A10.3(ii): no abrupt drop at the floor
            m = rich_factor(qk, hs[1], hs[2]) if (qk is not None and qk >= PS_Q_MIN) else 10.0
            ind.append(max(PS_ROUNDOFF, db[k] / n0 * max(1.0, m)))
            continue
        if qk is None or qk < PS_Q_MIN:
            ok = False
            continue
        ind.append(db[k] / n0 * max(1.0, rich_factor(qk, hs[1], hs[2])))
    old_rule_ok = all(x is not None and x >= 2.5 for x in q)
    return (max(ind) if ok and len(ind) == NT - 1 else None), q, ok, old_rule_ok


def crosscheck(fd_record, u3, ps, n0):
    """Ordered cross-check procedure (A2.4/A2.12/A3.6). fd_record: judge() output of R_ST (final).
    ps: dict(spatial=[(N, lat), (N, lat), (N, lat)] coarse->fine at dt, half=lat at N3 and dt/2, full=lat at N3 and dt,
    finite=bool, gates_ok=bool, stability_ok=bool). Returns dict with verdict and numbers."""
    rec = dict(x_ps=None, s_x=None, s_t=None, sigma=None, q=None, eps_ref=fd_record.get('eps_ref'))
    rec['old_rule_verdict'] = old_rule_verdict(fd_record, u3, ps, n0)
    if (fd_record.get('outcome') != 'accepted' or rec['eps_ref'] is None or not np.isfinite(rec['eps_ref'])
            or not (np.isfinite(n0) and n0 > 0)):
        return dict(rec, verdict='FD-INVALID')
    eps = rec['eps_ref']
    arrays = [x for _, x in ps['spatial']] + [ps['half'], ps['full'], u3]
    if not ps['finite'] or not all(np.isfinite(np.asarray(x)).all() for x in arrays):
        return dict(rec, verdict='UNRESOLVED', reason='non-finite')
    if not ps['gates_ok']:
        return dict(rec, verdict='UNRESOLVED', reason='gates')
    (N1, a), (N2, b), (N3, c) = ps['spatial']
    da, db = tnorm(a - b), tnorm(b - c)
    s_x, q, ok, old_ok = ps_spatial_indicator(da, db, [N1, N2, N3], n0)
    rec['q'] = q
    rec['s_x_raw'] = float(np.max(db[1:]) / n0)
    rec['old_rule_order_ok'] = old_ok
    if not ok:
        return dict(rec, verdict='UNRESOLVED', reason='spatial-order')
    if not ps['stability_ok']:
        return dict(rec, verdict='UNRESOLVED', reason='stability')
    s_t = float(np.max(tnorm(ps['full'] - ps['half'])[1:]) / n0)
    x_ps = float(np.max(tnorm(ps['half'] - u3)[1:]) / n0)
    rec.update(s_x=s_x, s_t=s_t, sigma=s_x + s_t, x_ps=x_ps, eta=x_ps / eps if eps > 0 else None)
    if not all(np.isfinite(v) for v in (s_x, s_t, x_ps)):
        return dict(rec, verdict='UNRESOLVED', reason='non-finite')
    if s_x + s_t > eps / 2:
        return dict(rec, verdict='UNRESOLVED', reason='resolution')
    return dict(rec, verdict='PASS' if x_ps <= 2 * eps else 'FAIL', reason=None)


def old_rule_verdict(fd_record, u3, ps, n0):
    """A10.3(iv): the pre-A10 rule (q >= 2.5 at every time, raw s_x) on the SAME selected ladder, on every path."""
    eps = fd_record.get('eps_ref')
    if (fd_record.get('outcome') != 'accepted' or eps is None or not np.isfinite(eps)
            or not (np.isfinite(n0) and n0 > 0)):
        return 'FD-INVALID'
    arrays = [x for _, x in ps['spatial']] + [ps['half'], ps['full'], u3]
    if not ps['finite'] or not ps['gates_ok'] or not all(np.isfinite(np.asarray(x)).all() for x in arrays):
        return 'UNRESOLVED'
    (N1, a), (N2, b), (N3, c) = ps['spatial']
    da, db = tnorm(a - b), tnorm(b - c)
    q = [solve_order(da[k], db[k], 1.0 / N1, 1.0 / N2, 1.0 / N3) for k in range(1, len(da))]
    if any(x is None or x < 2.5 for x in q) or not ps['stability_ok']:
        return 'UNRESOLVED'
    raw = float(np.max(db[1:]) / n0) + float(np.max(tnorm(ps['full'] - ps['half'])[1:]) / n0)
    x_ps = float(np.max(tnorm(ps['half'] - u3)[1:]) / n0)
    if not (np.isfinite(raw) and np.isfinite(x_ps)) or raw > eps / 2:
        return 'UNRESOLVED'
    return 'PASS' if x_ps <= 2 * eps else 'FAIL'


def dimension_status(verdicts):
    """A2.5: suspended if any FAIL (takes precedence over missing cases); validated if >= 6 of the 8 PASS; else
    unvalidated. `verdicts` must list all 8 pre-registered cases ('MISSING' for an absent one)."""
    if any(v == 'FAIL' for v in verdicts):
        return 'suspended'
    return 'validated' if sum(v == 'PASS' for v in verdicts) >= 6 else 'unvalidated'
