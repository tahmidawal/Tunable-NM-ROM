"""jcp-dims 2D rule names (DESIGN 2.2). Wraps the vendored, unchanged generators (qcore.offmesh_rule and Hari's
hari_quadrature) and adds draw suffixes, Monte Carlo and the bad-generating-vector fixture:

  gauss<p>                 tensor Gauss-Legendre, m = p^2 (qcore)
  fib<n>[s<k>]             Fibonacci rank-1 lattice, m = n (a Fibonacci number), random shift seed k (default 0 = qcore)
  sobol<m>[s<k>]           scrambled Sobol, scramble seed k
  halton<m>[s<k>]          scrambled Halton, scramble seed k
  smolyak<l>               Clenshaw-Curtis sparse grid, boundary nodes removed (qcore)
  mc<m>s<k>                uniform Monte Carlo, seed k (stage ii)
  badlat<n>                rank-1 lattice with generating vector z = (1, 1), shift seed 0 (must-fail fixture)

Seed 0 without suffix is the deployed realisation; 'fib4181' and 'fib4181s0' produce identical points (asserted in tests).
"""
from __future__ import annotations

import re

import numpy as np

import p1core as T2

Q = T2.load_qcore()
HQ = Q.HQ if hasattr(Q, 'HQ') else __import__('hari_quadrature')

_PAT = re.compile(r'^(gauss|fib|sobol|halton|smolyak|mc|badlat)(\d+)(?:s(\d+))?$')


def parse(name):
    m = _PAT.match(name)
    if not m:
        raise ValueError(name)
    fam, n, s = m.group(1), int(m.group(2)), m.group(3)
    seed = int(s) if s is not None else 0
    if fam in ('gauss', 'smolyak', 'badlat') and s is not None:
        raise ValueError(f'{name}: deterministic family takes no draw')
    if fam == 'mc' and s is None:
        raise ValueError(f'{name}: Monte Carlo needs an explicit draw')
    return fam, n, seed


def family(name):
    return parse(name)[0]


def size_of(name):
    fam, n, _ = parse(name)
    return n * n if fam == 'gauss' else n if fam != 'smolyak' else None


def points(name):
    fam, n, seed = parse(name)
    if fam in ('gauss', 'smolyak') or (fam in ('fib', 'sobol', 'halton') and seed == 0 and 's' not in name[len(fam):]):
        X, w = Q.offmesh_rule(f'{fam}{n}')
    elif fam == 'fib':
        X, w = HQ.fibonacci_lattice(Q._fib_index(n), seed=seed)
    elif fam == 'sobol':
        X, w = HQ.sobol(n, scramble=True, seed=seed)
    elif fam == 'halton':
        X, w = HQ.halton(n, scramble=True, seed=seed)
    elif fam == 'mc':
        X, w = HQ.uniform_mc(n, seed=seed)
    elif fam == 'badlat':
        i = np.arange(n)
        X = (np.stack([i / n, i / n], 1) + np.random.default_rng(0).uniform(size=2)) % 1.0
        w = np.full(n, 1.0 / n)
    else:  # pragma: no cover
        raise ValueError(name)
    X, w = np.asarray(X, float), np.asarray(w, float)
    assert X.ndim == 2 and X.shape[1] == 2 and len(w) == len(X), name
    assert np.all((X > 0) & (X < 1)), name
    return X, w
