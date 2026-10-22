"""imp-geometry scoring: error metric, Richardson reference allowance, three-way matching (DESIGN 3-4, A1)."""
from __future__ import annotations

import math
import numpy as np

TAUS = (0.01, 0.03, 0.10)
P_E0 = (1.8, 2.2)          # E0 observed-order band
P_CORNER = (1.3, 2.5)      # corner-limited band (A1): eligible with the observed order, labelled


def err(F, R):
    """F, R (..., 6, P): e = max_{k>=1} ||F_k - R_k|| / ||R_0||; non-finite -> inf."""
    F = np.asarray(F)
    R = np.asarray(R)
    num = np.sqrt(np.sum((F - R) ** 2, axis=-1))
    den = np.sqrt(np.sum(R[..., 0, :] ** 2, axis=-1))
    e = np.max(num[..., 1:], axis=-1) / den
    return np.where(np.all(np.isfinite(F), axis=(-1, -2)), e, np.inf)


def richardson(U1, U2, U3):
    """Coarse, middle, fine reference levels on the lattice, (6, P) each, one case. Positive times only.
    Returns dict(p (5,), eps (scalar, normalised), label in {E0, corner, ineligible}, d12, d23)."""
    n0 = float(np.linalg.norm(U3[0]))
    d12 = np.linalg.norm(U1[1:] - U2[1:], axis=-1) / n0
    d23 = np.linalg.norm(U2[1:] - U3[1:], axis=-1) / n0
    ok = np.all(np.isfinite(d12)) and np.all(np.isfinite(d23)) and np.all(d23 > 0) and np.all(d12 > 0)
    if not ok:
        return dict(p=[float('nan')] * 5, eps=float(np.max(d23)) if np.all(np.isfinite(d23)) else float('inf'),
                    label='ineligible', reason='zero or non-finite difference', d12=d12.tolist(), d23=d23.tolist())
    p = np.log2(d12 / d23)
    if np.all((p >= P_E0[0]) & (p <= P_E0[1])):
        lab = 'E0'
    elif np.all((p >= P_CORNER[0]) & (p <= P_CORNER[1])):
        lab = 'corner'
    else:
        lab = 'ineligible'
    if lab == 'ineligible':
        eps = float(np.max(d23))            # conservative stand-in: the last difference itself (labelled)
    else:
        eps = float(np.max(d23 / (2.0 ** p - 1)))
    return dict(p=p.tolist(), eps=eps, label=lab, reason='', d12=d12.tolist(), d23=d23.tolist())


def allowance(rich, delta_t):
    """Cell allowance U = 2 max_case (eps_case) + delta_t (time check), and the eligibility count."""
    eps = np.array([r['eps'] for r in rich])
    labs = [r['label'] for r in rich]
    n_ok = sum(l in ('E0', 'corner') for l in labs)
    return dict(U=float(2 * np.max(eps) + delta_t), n_eligible=n_ok, n_E0=labs.count('E0'),
                n_corner=labs.count('corner'), n_ineligible=labs.count('ineligible'), eps_max=float(np.max(eps)),
                delta_t=float(delta_t))


def three_way(stat, U, tau):
    if not np.isfinite(stat):
        return 'misses'
    if stat + U <= tau:
        return 'meets'
    if stat - U > tau:
        return 'misses'
    return 'unresolved'


def stats(e):
    e = np.asarray(e)
    return dict(worst=float(np.max(e)), median=float(np.median(e)))


def match(configs, U, tau, which='worst'):
    """configs: list of dict(stat={'worst','median'}, time, qualified, key). Fastest qualified config that meets tau."""
    ok = [c for c in configs if c.get('qualified', False) and three_way(c['stat'][which], U, tau) == 'meets'
          and c.get('time') is not None and np.isfinite(c['time'])]
    if not ok:
        return None
    return min(ok, key=lambda c: c['time'])


# ------------------------------------------------------------------------------------------- fixtures ---------

def selftest():
    """SW1a (metric), SW1b (three-way), MF-REF (reference gate) deterministic fixtures. Returns dict of booleans."""
    rng = np.random.default_rng(0)
    R = rng.standard_normal((6, 500))
    out = {}
    good = True
    for eps in (1e-3, 1e-2):
        Fp = R.copy()
        for k in range(1, 6):
            d = rng.standard_normal(500)
            Fp[k] = R[k] + d / np.linalg.norm(d) * eps * np.linalg.norm(R[0])
        good &= abs(err(Fp, R) / eps - 1) < 1e-10
    out['SW1a'] = bool(good)
    fx = [((0.5, 0.1, 1), 'meets'), ((1.5, 0.1, 1), 'misses'), ((0.95, 0.1, 1), 'unresolved'),
          ((0.9, 0.1, 1), 'meets'), ((1.1, 0.1, 1), 'unresolved'), ((float('inf'), 0.1, 1), 'misses')]
    out['SW1b'] = all(three_way(*a) == b for a, b in fx)
    # MF-REF: a clean second-order ladder is E0; finest := middle (zero difference) and a first-order ladder are not
    base = rng.standard_normal((6, 500))
    pert = rng.standard_normal((6, 500))
    L = [base + pert * (4.0 ** -j) * 1e-3 for j in range(3)]
    ok_clean = richardson(*L)['label'] == 'E0'
    ok_zero = richardson(L[0], L[1], L[1].copy())['label'] == 'ineligible'
    L1 = [base + pert * (2.0 ** -j) * 1e-3 for j in range(3)]
    ok_first = richardson(*L1)['label'] == 'ineligible'
    out['MF-REF'] = bool(ok_clean and ok_zero and ok_first)
    return out
