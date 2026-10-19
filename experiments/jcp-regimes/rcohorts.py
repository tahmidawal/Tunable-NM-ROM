"""jcp-regimes: regime cohorts (DESIGN §3, §3.1, §3.2, §3.3). NumPy only (the 3D peak normalisation uses the vendored
b3d_common through cohorts.py's import path when called).

Each regime g is a set of fresh development cases. Row j of regime g is drawn from
    np.random.default_rng([TAG, S_g, j])        TAG = 0x52454749 ('REGI'), S_g >= 2**31 (DESIGN §3.2)
a domain-separated stream (negligible collision probability; DESIGN A1.3, not a proof of disjointness) relative to E0's
historical cohorts (one-word seeds), E0's new sealed / OOD cohorts (2D two-word [seed, j], 3D one-word, seeds < 2**31) and
the training draws (one-word seeds 0, 1000). Descriptor-level duplicates are refused by assert_all_disjoint (A2.1).

2D descriptor of a single-bump case: (cx, cy, w, a, nu)                   -> IC = engines.initial (identical, asserted)
2D descriptor of a two-bump case:    (sign, a, nu, cx1, cy1, w1, b1, cx2, cy2, w2, b2, s_star)
                                      u0 = a (b1 G1 + sign b2 G2) / s_star, s_star = max over the 1025^2 nodes of |g|
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

TAG = 0x52454749
SEEDS = {2: {'N0': 3100000001, 'Ap': 3100000002, 'Am': 3100000003, 'C1': 3100000004, 'C2': 3100000005,
             'B1': 3100000006, 'B2': 3100000007, 'N0s': 3100000008},
         3: {'Ap3': 3100000101, 'Am3': 3100000102, 'B23': 3100000103}}
COUNT = 16
COUNTS = {'N0s': 8}                 # DESIGN A1.5: the stiff-corner in-range control has 8 cases
E0_HISTORICAL = {7090702, 911702, 20260927, 20260916, 923651, 923801, 923901}
E0_PUBLIC = {5550001, 5550002}
TRAIN_SEEDS = {0, 1000}
NU_TRAIN = (.01, .1)
A_TRAIN = (.5, 2.)
S_TRAIN = (5., 200.)
REGIMES_2D = ('N0', 'N0s', 'Ap', 'Am', 'C1', 'C2', 'B1', 'B2')
TWO_BUMP = ('B1', 'B2')
S_REF_L = 1024              # s_star grid of the two-bump families (DESIGN §3.1)
MAX_TRIES = 1000


class CohortError(ValueError):
    pass


def check_seed(seed):
    """G-seed (DESIGN §3.2): >= 2**31 and outside every E0 / training seed set."""
    s = int(seed)
    if s < 2 ** 31 or s in E0_HISTORICAL or s in E0_PUBLIC or s in TRAIN_SEEDS:
        raise CohortError(f'seed {s} not admissible (must be >= 2**31 and not an E0 / training seed)')
    return s


def _rng(seed, j):
    return np.random.default_rng([TAG, check_seed(seed), int(j)])


# ------------------------------------------------------------------------------------------------- 2D draws ---

def _bump_xyw(r):
    return r.uniform(.15, .85), r.uniform(.15, .85), r.uniform(.05, .20)


def draw_row_2d(regime, seed, j):
    r = _rng(seed, j)
    if regime in TWO_BUMP:
        nu = float(np.exp(r.uniform(np.log(.01), np.log(.1))))
        a = r.uniform(.5, 2.)
        for _ in range(MAX_TRIES):
            b = []
            for _k in range(2):
                cx, cy, w = _bump_xyw(r)
                b.append((cx, cy, w, r.uniform(.5, 1.)))
            if np.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1]) >= b[0][2] + b[1][2]:
                break
        else:
            raise CohortError(f'{regime} row {j}: no separated bump pair in {MAX_TRIES} tries')
        sign = 1. if regime == 'B1' else -1.
        row = np.array([sign, a, nu, *b[0], *b[1], 0.])
        row[-1] = s_star_2d(row)
        return row
    if regime == 'C1':
        # DESIGN A1.1: C1 row j is paired with A- row j: identical (cx, cy, w, nu); only the amplitude is redrawn (from
        # C1's own stream) in C1's interval
        cx, cy, w, _a, nu = draw_row_2d('Am', SEEDS[2]['Am'], j)
        a = r.uniform(250. * nu, min(2., 500. * nu))
        return np.array([cx, cy, w, a, nu])
    cx, cy, w = _bump_xyw(r)
    if regime == 'N0':
        nu = float(np.exp(r.uniform(np.log(.01), np.log(.1))))
        a = r.uniform(.5, 2.)
    elif regime == 'Ap':
        nu = 10. ** r.uniform(-1., -.4)
        a = r.uniform(max(.5, 5. * nu), 2.)
    elif regime == 'Am':
        nu = 10. ** r.uniform(-2.5, -2.15)
        a = r.uniform(.5, min(2., 200. * nu))
    elif regime == 'N0s':
        nu = float(np.exp(r.uniform(np.log(.01), np.log(.0126))))
        a = r.uniform(1.5, 2.)
    elif regime == 'C2':
        nu = 10. ** r.uniform(-2.8, -2.45)
        a = r.uniform(500. * nu, min(2., 1000. * nu))
    else:
        raise CohortError(regime)
    return np.array([cx, cy, w, a, nu])


def bump_sum(row, X, Y):
    sign, a, nu, cx1, cy1, w1, b1, cx2, cy2, w2, b2 = row[:11]
    return (b1 * np.exp(-((X - cx1) ** 2 + (Y - cy1) ** 2) / (2 * w1 * w1))
            + sign * b2 * np.exp(-((X - cx2) ** 2 + (Y - cy2) ** 2) / (2 * w2 * w2)))


def s_star_2d(row):
    x = np.arange(S_REF_L + 1) / S_REF_L
    X, Y = np.meshgrid(x, x, indexing='ij')
    return float(np.max(np.abs(bump_sum(row, X, Y))))


def check_bounds_2d(regime, row):
    """G-bounds (DESIGN §3 table): every row inside its regime box and, for A+/A-/C, outside the training box in the
    declared factor only. Raises CohortError."""
    lo_nu, hi_nu = NU_TRAIN
    if regime in TWO_BUMP:
        sign, a, nu = row[:3]
        b = (row[3:7], row[7:11])
        ok = (sign == (1. if regime == 'B1' else -1.) and A_TRAIN[0] <= a <= A_TRAIN[1] and lo_nu <= nu <= hi_nu
              and all(.15 <= q[0] <= .85 and .15 <= q[1] <= .85 and .05 <= q[2] <= .20 and .5 <= q[3] <= 1. for q in b)
              and np.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1]) >= b[0][2] + b[1][2] and row[11] > 0)
        if not ok:
            raise CohortError(f'{regime} row out of bounds: {row}')
        return
    cx, cy, w, a, nu = row
    s = a / nu
    common = .15 <= cx <= .85 and .15 <= cy <= .85 and .05 <= w <= .20 and A_TRAIN[0] <= a <= A_TRAIN[1]
    lg = np.log10(nu)
    tol = 1e-12
    box = {'N0': lo_nu <= nu <= hi_nu and S_TRAIN[0] - tol <= s <= S_TRAIN[1] + tol,
           'N0s': lo_nu <= nu <= .0126 and 1.5 <= a <= 2. and s <= S_TRAIN[1] + tol,
           'Ap': -1. <= lg <= -.4 and nu > hi_nu and S_TRAIN[0] - tol <= s <= S_TRAIN[1] + tol,
           'Am': -2.5 <= lg <= -2.15 and nu < lo_nu and S_TRAIN[0] - tol <= s <= S_TRAIN[1] + tol,
           'C1': -2.5 <= lg <= -2.15 and 250. - 1e-9 <= s <= 500. + 1e-9,
           'C2': -2.8 <= lg <= -2.45 and 500. - 1e-9 <= s <= 1000. + 1e-9}[regime]
    if not (common and box):
        raise CohortError(f'{regime} row out of bounds: {row}')


def descriptors_2d(regime, seed=None, n=None):
    seed = SEEDS[2][regime] if seed is None else seed
    n = COUNTS.get(regime, COUNT) if n is None else n
    rows = np.array([draw_row_2d(regime, seed, j) for j in range(n)])
    for r_ in rows:
        check_bounds_2d(regime, r_)
    return rows


def initial_2d(L, regime, row):
    """Nodal IC on the (L+1)^2 grid, wall nodes zero (the training convention)."""
    if regime in TWO_BUMP:
        x = np.arange(L + 1) / L
        X, Y = np.meshgrid(x, x, indexing='ij')
        u = row[1] * bump_sum(row, X, Y) / row[11]
    else:
        cx, cy, w, a, _ = row
        x = np.arange(L + 1) / L
        X, Y = np.meshgrid(x, x, indexing='ij')
        u = a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w * w))
    u = np.array(u, float)
    u[[0, -1], :] = 0.
    u[:, [0, -1]] = 0.
    return u


def nu_2d(regime, row):
    return float(row[2] if regime in TWO_BUMP else row[4])


def amp_2d(regime, row):
    return float(row[1] if regime in TWO_BUMP else row[3])


def desc_2d(regime, row):
    if regime in TWO_BUMP:
        keys = ('sign', 'a', 'nu', 'cx1', 'cy1', 'w1', 'b1', 'cx2', 'cy2', 'w2', 'b2', 's_star')
    else:
        keys = ('cx', 'cy', 'w', 'a', 'nu')
    d = dict(zip(keys, map(float, row)))
    d['s'] = amp_2d(regime, row) / nu_2d(regime, row)
    lg = np.log10(nu_2d(regime, row))
    d['d_nu'] = float(abs(lg - np.clip(lg, -2., -1.)))
    ls = np.log10(d['s'])
    d['d_s'] = float(abs(ls - np.clip(ls, np.log10(S_TRAIN[0]), np.log10(S_TRAIN[1]))))
    return d


def sha_rows(rows):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, float)).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------- 3D draws ---

def draw_table_3d(regime, seed=None, m=COUNT):
    """Raw 3D table (DESIGN §3.3), b3d_common.draw_param_table layout, rows from the three-word stream. Phase 2: refused
    until the 3D amendment (DESIGN A1.12, A2.7); the B23 draft lacks the sign rejection and |.| peak normalisation."""
    raise NotImplementedError('3D regimes need their own DESIGN amendment (A1.12)')
    seed = SEEDS[3][regime] if seed is None else seed
    B = np.zeros(m, dtype=np.int64)
    c = np.zeros((m, 3, 3))
    w = np.zeros((m, 3))
    rho = np.zeros((m, 3))
    A = np.zeros(m)
    nu = np.zeros(m)
    for j in range(m):
        r = _rng(seed, j)
        B[j] = r.integers(2, 4) if regime == 'B23' else r.integers(1, 4)
        for b in range(3):
            c[j, b] = r.uniform(.2, .8, 3)
            w[j, b] = r.uniform(.10, .20)
            rho[j, b] = r.uniform(.5, 1.)
        if regime == 'B23':
            nu[j] = np.exp(r.uniform(np.log(.01), np.log(.1)))
            A[j] = r.uniform(.5, 2.)
            rho[j, int(r.integers(0, B[j]))] *= -1.
        elif regime == 'Ap3':
            nu[j] = 10. ** r.uniform(-1., -.4)
            A[j] = r.uniform(max(.5, 5. * nu[j]), 2.)
        elif regime == 'Am3':
            nu[j] = 10. ** r.uniform(-2.5, -2.15)
            A[j] = r.uniform(.5, min(2., 200. * nu[j]))
        else:
            raise CohortError(regime)
    return dict(B=B, c=c, w=w, rho=rho, A=A, nu=nu, seed=int(seed), m=int(m), regime=regime)


# --------------------------------------------------------------------------------------------- disjointness ---

def training_rows_2d():
    """The bank's training draws (burgers2d_film.sample_params semantics: one default_rng(seed), column-wise)."""
    out = []
    for seed, m in ((0, 576), (1000, 4032)):
        r = np.random.default_rng(seed)
        cx, cy, w, a = r.uniform(.15, .85, m), r.uniform(.15, .85, m), r.uniform(.05, .20, m), r.uniform(.5, 2., m)
        nu = np.exp(r.uniform(np.log(.01), np.log(.1), m))
        out.append(np.stack([cx, cy, w, a, nu], 1))
    return np.concatenate(out)


def assert_disjoint_2d(rows, others):
    """G-train: no single-bump regime row within 1e-12 (max abs) of any training / dev / val / test row."""
    o = np.asarray(others, float)
    for r_ in rows:
        if r_.shape == (5,) and np.any(np.max(np.abs(o - r_[None, :]), axis=1) <= 1e-12):
            raise CohortError(f'row {r_} coincides with an existing cohort / training row')


SINGLE = tuple(g for g in REGIMES_2D if g not in TWO_BUMP)


def assert_all_disjoint(others):
    """DESIGN A2.1: single-bump rows of every regime vs `others` (training draws, dev6, val32, test64 descriptors) and vs
    each other (within and across regimes); two-bump rows of B1/B2 vs each other. Raises CohortError."""
    single = [(g, j, r_) for g in SINGLE for j, r_ in enumerate(descriptors_2d(g))]
    A = np.array([r_ for _, _, r_ in single])
    assert_disjoint_2d(A, others)
    for k, (g, j, r_) in enumerate(single):
        d = np.max(np.abs(A - r_[None, :]), axis=1)
        d[k] = np.inf
        if np.any(d <= 1e-12):
            m = int(np.argmin(d))
            raise CohortError(f'duplicate rows {g}|{j} and {single[m][0]}|{single[m][1]}')
    two = np.array([r_ for g in TWO_BUMP for r_ in descriptors_2d(g)])
    for k in range(len(two)):
        d = np.max(np.abs(two - two[k][None, :]), axis=1)
        d[k] = np.inf
        if np.any(d <= 1e-12):
            raise CohortError(f'duplicate two-bump rows {k} and {int(np.argmin(d))}')
    return dict(single=len(single), two_bump=len(two), others=len(others))


# ----------------------------------------------------------------------------------------- Cohort interface ---

class Cohort:
    """Same interface as E0 cohorts.Cohort (.dim, .name, .n, .nu(j), .ic(L, j) interior, .desc(j), .sha256)."""

    def __init__(self, dim, name, seed=None):
        if dim != 2 or name not in REGIMES_2D:
            raise CohortError(f'{dim}D regime {name} not available (3D is phase 2, DESIGN §3.3)')
        self.dim, self.name = dim, name
        self.D = descriptors_2d(name, seed)
        self.n = len(self.D)
        self.sha256 = sha_rows(self.D)

    def nu(self, j):
        return nu_2d(self.name, self.D[j])

    def full_ic(self, L, j):
        return initial_2d(L, self.name, self.D[j])

    def ic(self, L, j):
        return self.full_ic(L, j)[1:-1, 1:-1]

    def desc(self, j):
        return desc_2d(self.name, self.D[j])

    def ps_coefs(self, j, N, panels=256):
        """Exact sine coefficients of the continuous IC for the pseudo-spectral cross-check (E0 pscore.ic_coefs_2d per
        separable Gaussian; the two-bump families are linear combinations of two of them)."""
        import pscore as PS
        r_ = self.D[j]
        if self.name not in TWO_BUMP:
            return PS.ic_coefs_2d(r_, N, panels)
        sign, a, nu = r_[:3]
        b1 = PS.ic_coefs_2d((r_[3], r_[4], r_[5], 1., nu), N, panels)
        b2 = PS.ic_coefs_2d((r_[7], r_[8], r_[9], 1., nu), N, panels)
        return a * (r_[6] * b1 + sign * r_[10] * b2) / r_[11]


def registry():
    """runs/COHORTS.json content: seeds, counts, descriptor hashes and per-case descriptors (2D)."""
    reg = dict(tag=TAG, count=COUNT, seeds={f'{d}d': v for d, v in SEEDS.items()}, cohorts={})
    for g in REGIMES_2D:
        D = descriptors_2d(g)
        reg['cohorts'][g] = dict(sha256=sha_rows(D), n=len(D), desc=[desc_2d(g, r_) for r_ in D])
    return reg


if __name__ == '__main__':
    import json
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    reg = registry()
    s = json.dumps(reg, indent=1)
    if p:
        p.write_text(s + '\n')
    for g, v in reg['cohorts'].items():
        print(g, v['sha256'][:16], 'nu', min(x['nu'] for x in v['desc']), max(x['nu'] for x in v['desc']),
              's', round(min(x['s'] for x in v['desc']), 1), round(max(x['s'] for x in v['desc']), 1))
