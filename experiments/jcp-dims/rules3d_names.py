"""jcp-dims 3D rule names (DESIGN 2.3). Pinned files (speed3d rules.npz + rules_extra.npz, lane rules_dims.npz with
lat262144) give every seed-0 Gauss / CBC-lattice rule byte for byte; draws and new families are generated here from the
vendored, unchanged generators (vendor/quad3d/rules.py) and identified by the sha256 of points and weights:

  gl<p>, lat<n>, smol<l>(=smol8 file), sob16384     pinned files (speed3d names)
  lat<n>s<k>          the same CBC generating vector z (recovered from the pinned seed-0 points and verified by
                      regeneration bit for bit), shift numpy default_rng(k).uniform(size=3)
  sob<m>s<k>          sobol3 (scrambled, scramble seed k); m a power of two
  smolyak<l>          smolyak3(level l) (Clenshaw-Curtis, boundary nodes removed)
  badlat<n>           rank-1 lattice z = (1, 1, 1), shift seed 0 (must-fail fixture)
  mc<m>s<k>, hal<m>s<k>   stage (ii): uniform Monte Carlo / scrambled Halton (scipy qmc.Halton(d=3)), seed k
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import rules as RV  # noqa: E402  vendored generators, unchanged

DIMS_FILE = (HERE / 'rules3d' / 'rules_dims.npz', None)     # sha pinned after generation (see rules3d/rules_dims.json)
_PAT = re.compile(r'^(gl|lat|sob|smolyak|smol|badlat|mc|hal)(\d+)(?:s(\d+))?$')


def sha_rule(X, w):
    return hashlib.sha256(np.ascontiguousarray(X, np.float64).tobytes() + np.ascontiguousarray(w, np.float64).tobytes()).hexdigest()


def parse(name):
    m = _PAT.match(name)
    if not m:
        raise ValueError(name)
    fam, n, s = m.group(1), int(m.group(2)), m.group(3)
    return fam, n, (None if s is None else int(s))


def is_pinned(name):
    fam, _, s = parse(name)
    return s is None and fam in ('gl', 'lat', 'smol', 'sob')


def z_of(X0, n):
    """Generating vector of a shifted rank-1 lattice from its points: X_1 - X_0 = z / n (mod 1)."""
    d = (X0[1] - X0[0]) % 1.0
    return np.rint(d * n).astype(np.int64) % n


def make(name, pinned):
    """(X, w) of a lane rule. pinned: dict name -> (X, w) of the seed-0 pinned rules (s3common.load_rules + lat262144)."""
    fam, n, s = parse(name)
    if is_pinned(name):
        return pinned[name]
    if fam == 'lat':
        X0, w0 = pinned[f'lat{n}']
        z = z_of(X0, n)
        Xr, wr = RV.rank1_lattice3(n, z, seed=0)
        assert np.array_equal(Xr, X0) and np.array_equal(wr, w0), f'lat{n}: generating vector not recovered'
        return RV.rank1_lattice3(n, z, seed=s)
    if fam == 'sob':
        assert n & (n - 1) == 0, name
        return RV.sobol3(n, seed=s)
    if fam == 'smolyak':
        assert s is None
        return RV.smolyak3(n)
    if fam == 'badlat':
        assert s is None
        return RV.rank1_lattice3(n, np.array([1, 1, 1]), seed=0)
    if fam == 'mc':
        assert s is not None
        return np.random.default_rng(s).uniform(size=(n, 3)), np.full(n, 1.0 / n)
    if fam == 'hal':
        from scipy.stats import qmc
        assert s is not None
        return qmc.Halton(d=3, scramble=True, seed=s).random(n), np.full(n, 1.0 / n)
    raise ValueError(name)


def build(names, pinned):
    out = {}
    for nm in names:
        X, w = make(nm, pinned)
        X, w = np.ascontiguousarray(X, np.float64), np.ascontiguousarray(w, np.float64)
        assert X.ndim == 2 and X.shape[1] == 3 and len(w) == len(X) and np.all((X >= 0) & (X <= 1)), nm
        out[nm] = (X, w)
    return out
