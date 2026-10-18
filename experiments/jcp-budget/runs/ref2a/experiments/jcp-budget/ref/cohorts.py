"""jcp-references: cohort descriptors and canonical initial conditions (DESIGN §2, §9, A1.1, A1.10, A2.11).

Historical cohorts reuse the campaign's own generators unchanged:
  2D  engines.params_draw(seed, count) (column-wise; NOT prefix-stable, so always called with the exact count)
      dev6 = draw(7090702, 4) ++ draw(911702, 2); val32 = draw(20260927, 32); test64 = draw(20260916, 64)
  3D  common.table(seed, count) (b3d_common.build_param_table: row-wise draws + s_star peak on the 257^3 grid)
      dev4 = 923651 x 4; val = 923801 x 64; test = 923901 x 32
New cohorts (sealed, OOD; secret seeds) are row-wise and prefix-stable:
  2D  row j from np.random.default_rng([seed, j]): cx, cy, w, a, log nu (in that order)
  3D  b3d_common.draw_param_table semantics (one sequential generator, every row consumes the same draws) with the
      OOD ranges substituted (_draw3d_table), then s_star by b3d_common.peak_on_reference_grid.
OOD: log nu ~ U[log .01, log .01 + 0.1 log 10]; 2D w ~ U[.030, .050); 3D every blob w ~ U[.070, .100).

Canonical ICs: 2D engines.initial(L, desc) (nodal Gaussian, wall nodes zeroed, (L+1, L+1)); 3D
common.initial_interior(n, tab, j) (interior, flat, x slowest), with the cohort's own s_star at every resolution.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]          # jcp-budget: this copy lives in experiments/jcp-budget/ref
for _p in (ROOT / 'experiments' / 'mr-burgers2d', HERE.parent / 'vendor' / 'quad3d' / 'vendor' / 'burgers3d-span'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import engines as E2          # noqa: E402  (2D family, IC, vendor FOM)
import common as C3           # noqa: E402  (3D family, IC, vendor FOM)
b3 = C3.b3

HISTORICAL = {
    2: {'dev6': [(7090702, 4), (911702, 2)], 'val32': [(20260927, 32)], 'test64': [(20260916, 64)]},
    3: {'dev4': [(923651, 4)], 'val64': [(923801, 64)], 'test32': [(923901, 32)]},
}
# new cohorts: name -> (count, kind); their seeds are secret (DESIGN A1.11) and passed in at run time
NEW = {2: {'sealed128': (128, 'family'), 'ood32': (32, 'ood')},
       3: {'sealed64': (64, 'family'), 'ood16': (16, 'ood')}}
SEALED = {(d, n) for d in NEW for n in NEW[d]}
PUBLIC_CHECK_SEEDS = {2: 5550001, 3: 5550002}          # public seeds for the RNG unit test only (A2.11)
HISTORICAL_SEEDS = {s for d in HISTORICAL for v in HISTORICAL[d].values() for s, _ in v}

CALIB_COHORT = {2: 'dev6', 3: 'dev4'}
LOGNU_FAMILY = (np.log(.01), np.log(.1))
LOGNU_OOD = (np.log(.01), np.log(.01) + 0.1 * np.log(10.))


def cohort_names(dim):
    return list(HISTORICAL[dim]) + list(NEW[dim])


def is_sealed(dim, name):
    return (dim, name) in SEALED


def count(dim, name):
    if name in HISTORICAL[dim]:
        return sum(c for _, c in HISTORICAL[dim][name])
    return NEW[dim][name][0]


# ---------------------------------------------------------------------------------------------------------- 2D --

def rowwise_2d(seed, n, kind):
    """New 2D cohorts: row j from default_rng([seed, j]); prefix-stable."""
    w_lo, w_hi = (.05, .20) if kind == 'family' else (.030, .050)
    lo, hi = LOGNU_FAMILY if kind == 'family' else LOGNU_OOD
    rows = []
    for j in range(n):
        r = np.random.default_rng([int(seed), int(j)])
        cx, cy = r.uniform(.15, .85), r.uniform(.15, .85)
        w = r.uniform(w_lo, w_hi)
        a = r.uniform(.5, 2.)
        nu = np.exp(r.uniform(lo, hi))
        rows.append((cx, cy, w, a, nu))
    return np.asarray(rows, float)


def descriptors_2d(name, seed=None):
    """(n, 5) array (cx, cy, w, a, nu)."""
    if name in HISTORICAL[2]:
        return np.concatenate([E2.params_draw(s, c) for s, c in HISTORICAL[2][name]])
    n, kind = NEW[2][name]
    assert seed is not None and int(seed) not in HISTORICAL_SEEDS, name
    return rowwise_2d(seed, n, kind)


def ic_2d(L, desc):
    """Canonical nodal IC on the (L+1)^2 grid (engines.initial, wall nodes zero)."""
    return np.asarray(E2.initial(L, desc), float)


# ---------------------------------------------------------------------------------------------------------- 3D --

def _draw3d_table(seed, m, kind):
    """b3d_common.draw_param_table with the kind's ranges (identical draw order; prefix-stable)."""
    if kind == 'family':
        return b3.draw_param_table(seed, m)
    rng = np.random.default_rng(seed)
    B = np.zeros(m, dtype=np.int64)
    c = np.zeros((m, 3, 3))
    w = np.zeros((m, 3))
    rho = np.zeros((m, 3))
    A = np.zeros(m)
    nu = np.zeros(m)
    for j in range(m):
        B[j] = rng.integers(1, 4)
        for b in range(3):
            c[j, b] = rng.uniform(0.2, 0.8, 3)
            w[j, b] = rng.uniform(0.07, 0.10)
            rho[j, b] = rng.uniform(0.5, 1.0)
        A[j] = rng.uniform(0.5, 2.0)
        nu[j] = np.exp(rng.uniform(*LOGNU_OOD))
    overlap = np.full(m, np.inf)
    for j in range(m):
        if B[j] > 1:
            dmin = min(np.linalg.norm(c[j, a] - c[j, b_]) for a in range(B[j]) for b_ in range(a + 1, B[j]))
            overlap[j] = dmin / np.mean(w[j, :B[j]])
    return dict(B=B, c=c, w=w, rho=rho, A=A, nu=nu, overlap=overlap, seed=int(seed), m=int(m))


def _finish_table(tab):
    tab['s_star'] = b3.peak_on_reference_grid(tab)
    h = hashlib.sha256()
    for k in ('B', 'c', 'w', 'rho', 'A', 'nu', 's_star'):
        h.update(np.ascontiguousarray(tab[k]).tobytes())
    tab['sha256'] = h.hexdigest()
    return tab


def table_3d(name, seed=None):
    """paper-b3d parameter table dict (B, c, w, rho, A, nu, s_star, sha256, m)."""
    if name in HISTORICAL[3]:
        (s, c), = HISTORICAL[3][name]
        return C3.table(s, c)
    n, kind = NEW[3][name]
    assert seed is not None and int(seed) not in HISTORICAL_SEEDS, name
    return _finish_table(_draw3d_table(int(seed), n, kind))


def ic_3d(L, tab, j, chunk=1 << 24):
    """Canonical interior IC (flat, x slowest) on the (L+1)-node grid: common.initial_interior for L <= 256; above that
    the same b3d_common._blob_sum evaluated on coordinate chunks (memory; elementwise identical formula, checked
    against the unchunked vendor call by the unit gate Uic3d)."""
    if L <= 256:
        return np.asarray(C3.initial_interior(L + 1, tab, j), float)
    return ic_3d_chunked(L, tab, j, chunk)


def ic_3d_chunked(L, tab, j, chunk=1 << 24):
    n = L + 1
    ni = n - 2
    ax = np.linspace(0.0, 1.0, n)[1:-1]
    row = b3.table_row(tab, j)
    out = np.empty(ni ** 3)
    plane = ni * ni
    step = max(1, chunk // plane)
    Y, Z = np.meshgrid(ax, ax, indexing='ij')
    yz = np.stack([Y.ravel(), Z.ravel()], 1)
    for i0 in range(0, ni, step):
        xs = ax[i0:i0 + step]
        coords = np.concatenate([np.concatenate([np.full((plane, 1), x), yz], 1) for x in xs])
        s = np.asarray(b3._blob_sum(coords, row))
        out[i0 * plane:(i0 + len(xs)) * plane] = row['A'] * s / float(tab['s_star'][j])
    return out


# ------------------------------------------------------------------------------------------------------ generic --

class Cohort:
    """Uniform view: .n, .nu(j), .ic(L, j) -> interior array of shape (L-1,)*dim, .desc(j) (JSON-able), .sha256."""

    def __init__(self, dim, name, seed=None):
        self.dim, self.name = dim, name
        if dim == 2:
            self.D = descriptors_2d(name, seed)
            self.n = len(self.D)
            self.sha256 = hashlib.sha256(np.ascontiguousarray(self.D).tobytes()).hexdigest()
        else:
            self.T = table_3d(name, seed)
            self.n = int(self.T['m'])
            self.sha256 = self.T['sha256']

    def nu(self, j):
        return float(self.D[j, 4]) if self.dim == 2 else float(self.T['nu'][j])

    def ic(self, L, j):
        if self.dim == 2:
            return ic_2d(L, self.D[j])[1:-1, 1:-1]
        return ic_3d(L, self.T, j).reshape((L - 1,) * 3)

    def desc(self, j):
        if self.dim == 2:
            return dict(zip(('cx', 'cy', 'w', 'a', 'nu'), map(float, self.D[j])))
        T = self.T
        B = int(T['B'][j])
        return dict(B=B, c=T['c'][j][:B].tolist(), w=T['w'][j][:B].tolist(), rho=T['rho'][j][:B].tolist(),
                    A=float(T['A'][j]), nu=float(T['nu'][j]), s_star=float(T['s_star'][j]))


def crosscheck_cases(dim):
    """DESIGN §6/A1.7: the 4 validation cases with the smallest nu (ties by lower index) and the 4 lowest-index
    remaining cases. 2D val32, 3D val64."""
    if dim == 2:
        nus = descriptors_2d('val32')[:, 4]
    else:
        (s, n), = HISTORICAL[3]['val64']
        nus = b3.draw_param_table(s, n)['nu']           # raw draws only (s_star does not affect nu)
    nus = np.asarray(nus, float)
    c = type('C', (), {'n': len(nus)})
    low = list(np.lexsort((np.arange(c.n), nus))[:4])
    rest = [j for j in range(c.n) if j not in low][:4]
    return sorted(int(j) for j in low + rest)
