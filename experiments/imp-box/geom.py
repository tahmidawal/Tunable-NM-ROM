"""imp-box geometry adapter: the unit box, cohorts (family width W x steepness level), scoring lattice, envelope, IC and
composite Gauss rules (DESIGN §1-§2, A1.1, A2.1, A2.3). Provides the attribute set the reused imp-geometry modules read
(rom.py: ic, N_OUT, T_END; bank.py: SIDE, envelope_jnp, composite_rule, lattice).

IC (the pilot's Part A): u0 = a exp(-|x-c|^2 / (2 w^2)) in the interior, zero boundary trace. Domain id is always 'B'.
"""
from __future__ import annotations

import hashlib
import numpy as np

SIDE = 1.0
T_END = 0.25
N_OUT = 5
LAT = 256
DOM = 'B'

WIDTHS = {'W08': 0.08, 'W16': 0.16, 'W28': 0.28, 'W40': 0.40}
W_ID = {'W08': 1, 'W16': 2, 'W28': 3, 'W40': 4}
LEVELS = {'A1': (0.06, 128), 'A2': (0.02, 192), 'A3': (0.02 / 3, 384), 'A4': (0.02 / 9, 768)}   # (nu_lo, N_ref)
LEV_ID = {'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4}
SEEDS = {'train': 4_100_001, 'dev': 4_200_001, 'val': 4_300_001, 'test': 4_400_001}
NSPLIT = {'train': 96, 'dev': 8, 'val': 32, 'test': 32}
CAMPAIGN_SEEDS = {7090702, 911702, 20260927, 20260916, 923651, 923801, 923901, 0, 5550001, 5550002,
                  1_000_001, 2_000_001}
C_BAR = 0.5


# ------------------------------------------------------------------------------------------------- cohorts ----

def base_rows(split, n=None):
    """(n, 5) base uniforms (u_x, u_y, u_w, u_a, u_nu); row j from default_rng([seed, j]) (prefix-stable)."""
    seed = SEEDS[split]
    assert seed not in CAMPAIGN_SEEDS
    n = NSPLIT[split] if n is None else n
    return np.array([np.random.default_rng([seed, j]).uniform(size=5) for j in range(n)])


def to_desc(B, W, lev, centred=False):
    """Base rows -> descriptors (cx, cy, w, a, nu) of cell (W, lev); centred=True puts c at (0.5, 0.5) (train-c)."""
    wv = WIDTHS[W] if isinstance(W, str) else float(W)
    nu_lo = LEVELS[lev][0]
    D = np.empty_like(B)
    D[:, 0] = C_BAR + wv * (B[:, 0] - 0.5)
    D[:, 1] = C_BAR + wv * (B[:, 1] - 0.5)
    if centred:
        D[:, :2] = C_BAR
    D[:, 2] = 0.035 + 0.015 * B[:, 2]
    D[:, 3] = 1.0 + B[:, 3]
    D[:, 4] = nu_lo * 2.0 ** B[:, 4]
    return D


def cohorts(W, lev, n_train=None):
    out = {s: to_desc(base_rows(s, n_train if s == 'train' else None), W, lev) for s in SEEDS}
    out['train_c'] = to_desc(base_rows('train', n_train), W, lev, centred=True)
    return out


def check_disjoint(n_train=None):
    """A2.1: no base row of one split isclose to a row of another (all five uniforms); the actual training population."""
    rows = {s: base_rows(s, n_train if s == 'train' else None) for s in SEEDS}
    names = list(rows)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            close = np.all(np.isclose(rows[a][:, None, :], rows[b][None, :, :]), axis=-1)
            assert not close.any(), (a, b)
    return {s: hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest() for s, r in rows.items()}


def steep(D):
    return D[:, 3] / D[:, 4]


# ------------------------------------------------------------------------------------------------ geometry -----

def inside(dom, x, y, tol=1e-12):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return (x > tol) & (x < SIDE - tol) & (y > tol) & (y < SIDE - tol)


def envelope_np(dom, x, y, box_only=False):
    return 16.0 * x * (1 - x) * y * (1 - y)


def envelope_jnp(dom, xy, box_only=False):
    x, y = xy[..., 0], xy[..., 1]
    return 16.0 * x * (1 - x) * y * (1 - y)


def boundary_samples(dom, n=500):
    s = (np.arange(n) + 0.5) / n
    z, o = np.zeros(n), np.ones(n)
    return np.concatenate([np.stack(p, -1) for p in ((s, z), (o, s), (s, o), (z, s))])


def lattice(dom=DOM, lat=LAT):
    """Full scoring lattice x = i/lat (i = 0..lat, walls included; 257^2 at lat = 256), x-index major: (P, 2), I, J.
    Every arm except R-TRANS is exactly zero on the walls; R-TRANS's wall values are scored (code audit 1)."""
    i = np.arange(0, lat + 1)
    I, J = np.meshgrid(i, i, indexing='ij')
    return np.stack([(I / lat).ravel(), (J / lat).ravel()], -1), I.ravel(), J.ravel()


def ic(dom, d, x, y, xp=np):
    cx, cy, w, a = d[0], d[1], d[2], d[3]
    return a * xp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w))


# ------------------------------------------------------------------------------------- composite Gauss rules ---

def gauss_legendre(q):
    return np.polynomial.legendre.leggauss(q)


def rule_1d(k, q):
    """Composite q-point Gauss-Legendre on k equal elements of [0, 1]: points (k q,), weights (k q,)."""
    g, w = gauss_legendre(q)
    h = 1.0 / k
    left = np.arange(k) * h
    x = (left[:, None] + 0.5 * h * (g[None, :] + 1.0)).ravel()
    wt = np.tile(0.5 * h * w, k)
    return x, wt


def composite_rule(dom, k, q, grade=None, chordal=False):
    """Tensor composite rule on the unit box, ordered x-index major: points (m, 2), weights (m,)."""
    x, w = rule_1d(k, q)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.stack([X.ravel(), Y.ravel()], -1), np.outer(w, w).ravel()
