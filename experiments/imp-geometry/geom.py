"""imp-geometry: domains, cohorts, scoring lattice, envelopes, macro-block meshes and composite quadrature (DESIGN 1-2, 5.2-5.3).

Domains inside the unit square (DESIGN A1.2):
  L  = (0,1)^2 minus [0.625,1]^2            (re-entrant corner at (0.625, 0.625))
  H  = (0,1)^2 minus closed disk((0.7,0.7), 0.1)
  B  = (0,1)^2                              (box control)
Initial condition (A1.2): u0 = a exp(-|x-c|^2/(2w^2)) * chi_Omega(x), chi = smooth cutoff vanishing to 4th order on the
boundary: chi = psi(x) psi(1-x) psi(y) psi(1-y) [* psi(phi_Q / 2) (L) | * psi(phi_hole) (H)], psi(s) = 1 - exp(-(s/delta)^4),
delta = 0.05, phi_Q = f1 + f2 + sqrt(f1^2 + f2^2) (f = LC - x, LC - y), phi_hole = (|x-c_h|^2 - r^2) / (2 r).
Everything here is NumPy (offline); envelopes also have a jnp version for the bank.
"""
from __future__ import annotations

import math
import numpy as np

SIDE = 1.0
DELTA = 0.05
# geometry / family variants (DESIGN A2): v0 = features downstream of the family square; v2 = features inside it with a
# shared rejection zone for bump centres (descriptors stay identical for L, H and B)
VARIANTS = {
    'v0': dict(LC=0.625, HC=(0.7, 0.7), HR=0.1, HSQ=(0.55, 0.85), wide_c0=0.15, narrow_c0=0.47, reject=None),
    'v2': dict(LC=0.45, HC=(0.45, 0.45), HR=0.1, HSQ=(0.3, 0.6), wide_c0=0.15, narrow_c0=0.30, reject=0.03),
}
VARIANT = 'v2'


def configure(variant):
    global LC, HC, HR, HSQ, REJECT, VARIANT
    v = VARIANTS[variant]
    VARIANT = variant
    LC, HC, HR, HSQ, REJECT = v['LC'], v['HC'], v['HR'], v['HSQ'], v['reject']
    FAMS['wide']['c0'] = v['wide_c0']
    FAMS['narrow']['c0'] = v['narrow_c0']


def rejected(cx, cy):
    """Bump centre inside the shared exclusion zone (union of the L quadrant and the hole, grown by REJECT)."""
    if REJECT is None:
        return False
    m = REJECT
    inq = (cx > LC - m) and (cy > LC - m)
    inh = (cx - HC[0]) ** 2 + (cy - HC[1]) ** 2 < (HR + m) ** 2
    return bool(inq or inh)
T_END = 0.25
N_OUT = 5
LAT = 64                          # scoring lattice spacing 1/64 per unit length
DOMS = ('L', 'H', 'B')

# ------------------------------------------------------------------------------------------------ cohorts ------
FAMS = {'narrow': dict(W=0.08, c0=0.47), 'wide': dict(W=0.40, c0=0.15)}
FAM_ID = {'narrow': 0, 'wide': 1}
LEVELS = {'S1': (0.02, 0.04), 'S2': (0.02 / 3, 0.04 / 3)}
LEV_ID = {'S1': 1, 'S2': 2}
A_RANGE = (1.0, 2.0)
W_RANGE = (0.035, 0.05)
SPLITS = {'train': (1, 192), 'dev': (2, 8), 'val': (3, 32), 'test': (4, 32)}
# every campaign / pilot seed we know of (asserted disjoint)
CAMPAIGN_SEEDS = {0, 7090702, 911702, 20260927, 20260916, 923801, 923901, 3_000_001, 4_000_001, 3_000_000, 4_000_000}


LC = HC = HR = HSQ = REJECT = None


def seed_of(fam, lev, split):
    return 7_100_000 + 1000 * FAM_ID[fam] + 100 * LEV_ID[lev] + SPLITS[split][0]


def draw(fam, lev, split, n=None):
    """(n, 5) descriptors (cx, cy, w, a, nu); row j from default_rng([seed, j])."""
    seed = seed_of(fam, lev, split)
    assert seed not in CAMPAIGN_SEEDS
    n = SPLITS[split][1] if n is None else n
    F = FAMS[fam]
    lo, hi = LEVELS[lev]
    rows = []
    for j in range(n):
        r = np.random.default_rng([seed, j])
        cx = r.uniform(F['c0'], F['c0'] + F['W'])
        cy = r.uniform(F['c0'], F['c0'] + F['W'])
        while rejected(cx, cy):
            cx = r.uniform(F['c0'], F['c0'] + F['W'])
            cy = r.uniform(F['c0'], F['c0'] + F['W'])
        w = r.uniform(*W_RANGE)
        a = r.uniform(*A_RANGE)
        nu = math.exp(r.uniform(math.log(lo), math.log(hi)))
        rows.append((cx, cy, w, a, nu))
    return np.asarray(rows)


def check_cohorts():
    """G-COH: seeds disjoint from the campaign; descriptor rows disjoint across splits of each (family, level)."""
    seen = set()
    for fam in FAMS:
        for lev in LEVELS:
            D = {s: draw(fam, lev, s) for s in SPLITS}
            for s in SPLITS:
                sd = seed_of(fam, lev, s)
                assert sd not in CAMPAIGN_SEEDS and sd not in seen
                seen.add(sd)
            keys = list(SPLITS)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    A, B = D[keys[i]], D[keys[j]]
                    assert not np.any(np.all(np.isclose(A[:, None, :], B[None, :, :], rtol=0, atol=1e-12), axis=-1))
    return True


def steep(D):
    return D[:, 3] / D[:, 4]


# ------------------------------------------------------------------------------------------------ geometry -----

def inside(dom, x, y, tol=1e-12):
    """Strict interior of the domain (boundary points are False)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > tol) & (x < SIDE - tol) & (y > tol) & (y < SIDE - tol)
    if dom == 'L':
        m &= ~((x >= LC - tol) & (y >= LC - tol))
    elif dom == 'H':
        m &= (x - HC[0]) ** 2 + (y - HC[1]) ** 2 > (HR + tol) ** 2
    else:
        assert dom == 'B'
    return m


def area(dom):
    return {'B': SIDE ** 2, 'L': SIDE ** 2 - (SIDE - LC) ** 2, 'H': SIDE ** 2 - math.pi * HR * HR}[dom]


def envelope_np(dom, x, y, box_only=False):
    X, Y = x / SIDE, y / SIDE
    pb = 16.0 * X * (1 - X) * Y * (1 - Y)
    if dom == 'B' or box_only:
        return pb
    if dom == 'L':
        f1, f2 = LC - x, LC - y
        return pb * (f1 + f2 + np.sqrt(f1 * f1 + f2 * f2))
    return pb * ((x - HC[0]) ** 2 + (y - HC[1]) ** 2 - HR * HR) / (2 * HR)


def envelope_jnp(dom, xy, box_only=False):
    import jax.numpy as jnp
    x, y = xy[..., 0], xy[..., 1]
    X, Y = x / SIDE, y / SIDE
    pb = 16.0 * X * (1 - X) * Y * (1 - Y)
    if dom == 'B' or box_only:
        return pb
    if dom == 'L':
        f1, f2 = LC - x, LC - y
        # sqrt(f1^2+f2^2) with a safe gradient at the corner point itself
        rr = f1 * f1 + f2 * f2
        s = jnp.sqrt(jnp.where(rr > 0, rr, 1.0)) * (rr > 0)
        return pb * (f1 + f2 + s)
    return pb * ((x - HC[0]) ** 2 + (y - HC[1]) ** 2 - HR * HR) / (2 * HR)


def boundary_samples(dom, n=500):
    """Points on the boundary of the domain (MF-ENV)."""
    s = (np.arange(n) + 0.5) / n
    pts = []
    if dom == 'L':
        S_, C_ = SIDE, LC
        segs = [((0, 0), (S_, 0)), ((S_, 0), (S_, C_)), ((S_, C_), (C_, C_)), ((C_, C_), (C_, S_)), ((C_, S_), (0, S_)),
                ((0, S_), (0, 0))]
    else:
        S_ = SIDE
        segs = [((0, 0), (S_, 0)), ((S_, 0), (S_, S_)), ((S_, S_), (0, S_)), ((0, S_), (0, 0))]
    for (a, b) in segs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        pts.append(a[None] + s[:, None] * (b - a)[None])
    if dom == 'H':
        th = 2 * np.pi * s
        pts.append(np.stack([HC[0] + HR * np.cos(th), HC[1] + HR * np.sin(th)], -1))
    return np.concatenate(pts)


def lattice(dom, lat=LAT):
    """Lattice points x = i/lat (i = 1 .. SIDE lat - 1) strictly inside the domain: (P, 2) coords, (P,) i, (P,) j."""
    i = np.arange(1, int(round(SIDE * lat)))
    I, J = np.meshgrid(i, i, indexing='ij')
    x, y = I / lat, J / lat
    m = inside(dom, x, y)
    return np.stack([x[m], y[m]], -1), I[m], J[m]


def lat_from_grid(dom, n, lat=LAT):
    """Indices into the (SIDE n - 1)^2 FD interior array of the lattice nodes (n a multiple of lat)."""
    assert n % lat == 0
    _, I, J = lattice(dom, lat)
    s = n // lat
    return I * s - 1, J * s - 1


def _psi(xp, s):
    return 1.0 - xp.exp(-(xp.maximum(s, 0.0) / DELTA) ** 4)


def chi(dom, x, y, xp=np):
    c = _psi(xp, x) * _psi(xp, SIDE - x) * _psi(xp, y) * _psi(xp, SIDE - y)
    if dom == 'L':
        f1, f2 = LC - x, LC - y
        rr = f1 * f1 + f2 * f2
        s = xp.sqrt(xp.where(rr > 0, rr, 1.0)) * (rr > 0)
        c = c * _psi(xp, (f1 + f2 + s) / 2)
    elif dom == 'H':
        c = c * _psi(xp, ((x - HC[0]) ** 2 + (y - HC[1]) ** 2 - HR * HR) / (2 * HR))
    return c


def ic(dom, d, x, y, xp=np):
    """u0 at points (x, y) for descriptor d (numpy or jax.numpy via xp; d may be traced)."""
    cx, cy, w, a = d[0], d[1], d[2], d[3]
    return a * xp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w)) * chi(dom, x, y, xp)


# ------------------------------------------------------------------------------------- macro-block meshes ------
# A macro is a map M: [0,1]^2 -> R^2 with analytic derivatives; elements are parametric sub-rectangles of a macro.

class RectMacro:
    curved = False

    def __init__(self, x0, x1, y0, y1):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def map(self, s, t):
        x = self.x0 + (self.x1 - self.x0) * s
        y = self.y0 + (self.y1 - self.y0) * t
        z = np.zeros_like(s)
        return (np.stack([x, y], -1), np.stack([z + self.x1 - self.x0, z], -1), np.stack([z, z + self.y1 - self.y0], -1))


class CurvedMacro:
    """t = 0: straight side P00 -> P10; t = 1: circle arc theta0 -> theta1 (centre HC, radius HR); radial sides straight.
    chordal=True replaces the arc by its chord (MF-QUAD / MF-SEM control)."""
    curved = True

    def __init__(self, P00, P10, th0, th1, chordal=False):
        self.P00, self.P10 = np.asarray(P00, float), np.asarray(P10, float)
        self.th0, self.th1 = th0, th1
        self.chordal = chordal

    def arc(self, s):
        th = self.th0 + (self.th1 - self.th0) * s
        A = np.stack([HC[0] + HR * np.cos(th), HC[1] + HR * np.sin(th)], -1)
        dA = (self.th1 - self.th0) * HR * np.stack([-np.sin(th), np.cos(th)], -1)
        if self.chordal:
            a0 = np.array([HC[0] + HR * math.cos(self.th0), HC[1] + HR * math.sin(self.th0)])
            a1 = np.array([HC[0] + HR * math.cos(self.th1), HC[1] + HR * math.sin(self.th1)])
            A = a0 + s[..., None] * (a1 - a0)
            dA = np.broadcast_to(a1 - a0, A.shape).copy()
        return A, dA

    def map(self, s, t):
        B = self.P00 + s[..., None] * (self.P10 - self.P00)
        dB = np.broadcast_to(self.P10 - self.P00, B.shape)
        A, dA = self.arc(s)
        tt = t[..., None]
        X = (1 - tt) * B + tt * A
        return X, (1 - tt) * dB + tt * dA, A - B


def graded_breaks(k, grade, toward_end):
    """k sub-intervals of [0,1]; grade None = uniform, else geometric ratio grade (each element grade x the previous
    one when moving toward the graded end)."""
    if grade is None or k == 1:
        b = np.linspace(0, 1, k + 1)
    else:
        sz = grade ** np.arange(k)
        b = np.concatenate([[0], np.cumsum(sz)]) / sz.sum()
    if not toward_end:
        b = 1 - b[::-1]
    return b


class Mesh:
    """Conforming macro-block quad mesh of a domain: list of (macro, s0, s1, t0, t1) elements."""

    def __init__(self, dom, k, grade=None, chordal=False):
        self.dom, self.k, self.grade, self.chordal = dom, k, grade, chordal
        E = []
        if dom in ('B', 'L'):
            bt = graded_breaks(k, grade if dom == 'L' else None, True)    # [0,LC] graded toward LC
            bf = graded_breaks(k, grade if dom == 'L' else None, False)   # [LC,SIDE] graded toward LC (its start)
            C_, S_ = LC, SIDE
            blocks = [((0, C_, 0, C_), bt, bt), ((C_, S_, 0, C_), bf, bt), ((0, C_, C_, S_), bt, bf)]
            if dom == 'B':
                blocks.append(((C_, S_, C_, S_), bf, bf))
            for (x0, x1, y0, y1), bs, bts in blocks:
                M = RectMacro(x0, x1, y0, y1)
                for i in range(k):
                    for j in range(k):
                        E.append((M, bs[i], bs[i + 1], bts[j], bts[j + 1]))
        else:
            assert dom == 'H' and grade is None
            xb = [0.0, HSQ[0], HSQ[1], SIDE]
            u = np.linspace(0, 1, k + 1)
            for a in range(3):
                for b in range(3):
                    if a == 1 and b == 1:
                        continue
                    M = RectMacro(xb[a], xb[a + 1], xb[b], xb[b + 1])
                    for i in range(k):
                        for j in range(k):
                            E.append((M, u[i], u[i + 1], u[j], u[j + 1]))
            kr = max(1, math.ceil(k / 2))
            ur = np.linspace(0, 1, kr + 1)
            d = math.pi / 4
            a_, b_ = HSQ
            cm = [CurvedMacro((a_, a_), (b_, a_), 5 * d, 7 * d, chordal),
                  CurvedMacro((b_, a_), (b_, b_), 7 * d, 9 * d, chordal),
                  CurvedMacro((b_, b_), (a_, b_), 1 * d, 3 * d, chordal),
                  CurvedMacro((a_, b_), (a_, a_), 3 * d, 5 * d, chordal)]
            for M in cm:
                for i in range(k):
                    for j in range(kr):
                        E.append((M, u[i], u[i + 1], ur[j], ur[j + 1]))
        self.elems = E

    def map_ref(self, xi, eta):
        """Physical coordinates and Jacobian of every element at reference points xi, eta in [-1,1] (broadcast arrays):
        X (E, ..., 2), dX/dxi (E, ..., 2), dX/deta (E, ..., 2)."""
        Xs, Dxi, Deta = [], [], []
        for (M, s0, s1, t0, t1) in self.elems:
            s = s0 + (s1 - s0) * (xi + 1) / 2
            t = t0 + (t1 - t0) * (eta + 1) / 2
            X, Ms, Mt = M.map(np.asarray(s, float), np.asarray(t, float))
            Xs.append(X)
            Dxi.append(Ms * (s1 - s0) / 2)
            Deta.append(Mt * (t1 - t0) / 2)
        return np.stack(Xs), np.stack(Dxi), np.stack(Deta)


def gauss_legendre(q):
    x, w = np.polynomial.legendre.leggauss(q)
    return x, w


def composite_rule(dom, k, q, grade=None, chordal=False):
    """Composite q x q Gauss rule on the macro mesh: points (m, 2), weights (m,) (|det J| included)."""
    g, w = gauss_legendre(q)
    XI, ET = np.meshgrid(g, g, indexing='ij')
    W2 = np.outer(w, w)
    X, Dx, De = Mesh(dom, k, grade, chordal).map_ref(XI, ET)
    det = Dx[..., 0] * De[..., 1] - Dx[..., 1] * De[..., 0]
    wts = np.abs(det) * W2[None]
    return X.reshape(-1, 2), wts.reshape(-1)


def rule_selfcheck(dom, k, q):
    """MF-QUAD numbers: area error and the error of int phi^2 against a 2x finer rule (k doubled, q+4)."""
    X, w = composite_rule(dom, k, q)
    Xf, wf = composite_rule(dom, 2 * k, q + 4)
    a = float(w.sum())
    f = lambda P: envelope_np(dom, P[:, 0], P[:, 1]) ** 2
    i1, i2 = float(w @ f(X)), float(wf @ f(Xf))
    return dict(area=a, area_exact=area(dom), area_relerr=abs(a - area(dom)) / area(dom),
                phi2=i1, phi2_fine=i2, phi2_relerr=abs(i1 - i2) / abs(i2),
                all_inside=bool(np.all(inside(dom, X[:, 0], X[:, 1]))))


configure(VARIANT)
