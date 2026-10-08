"""Quadrature rules for the tested nonlinear term.

Off-mesh rules return points X in (0,1)^2 and weights w with sum(w) = 1 (they
approximate int_[0,1]^2 f dx); the decoder is read directly at X (partial
decoding).  Mesh rules return interior node indices and weights such that
    (P F)_ab ~ sum_q w_q psi_ab(i_q, j_q) F_{i_q j_q}
with psi the mesh-orthonormal sine vectors, i.e. w_q ~ (number of mesh nodes
represented by node q).
"""
import numpy as np
from scipy.stats import qmc
from scipy.optimize import nnls


# ------------------------------------------------------------- 1D rules on [0,1]
def gauss_legendre_1d(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def clenshaw_curtis_1d(n):
    """n-point Clenshaw--Curtis on [0,1] (nested for n = 2^l + 1); n=1 is the midpoint."""
    if n == 1:
        return np.array([0.5]), np.array([1.0])
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))          # Chebyshev extrema on [-1,1]
    w = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(1, (n - 1) // 2 + 1):
            b = 1.0 if 2 * j != n - 1 else 0.5
            s += b / (4 * j * j - 1) * np.cos(2 * j * np.pi * i / (n - 1))
        w[i] = 2.0 / (n - 1) * (1.0 - 2.0 * s)
    w[0] /= 2; w[-1] /= 2
    x = 0.5 * (x[::-1] + 1.0); w = 0.5 * w[::-1]
    return x, w


def _level_size(rule, i):
    if rule == "cc":
        return 1 if i == 1 else 2 ** (i - 1) + 1
    if rule == "gl_lin":
        return 2 * i - 1
    if rule == "gl_exp":
        return 2 ** i - 1
    raise ValueError(rule)


def _rule1d(rule, n):
    return clenshaw_curtis_1d(n) if rule == "cc" else gauss_legendre_1d(n)


def smolyak(level, rule="cc"):
    """Smolyak sparse grid on [0,1]^2 at the given level (level 0 = one point).
    Combination formula:  A = sum_{|i| = q} Q_i1 x Q_i2 - sum_{|i| = q-1} Q_i1 x Q_i2,
    q = level + 2.  Nested 1D rules (cc) merge points; negative weights can occur."""
    q = level + 2
    pts = {}
    def add(i1, i2, sign):
        x1, w1 = _rule1d(rule, _level_size(rule, i1))
        x2, w2 = _rule1d(rule, _level_size(rule, i2))
        for a in range(len(x1)):
            for b in range(len(x2)):
                key = (round(x1[a], 12), round(x2[b], 12))
                pts[key] = pts.get(key, 0.0) + sign * w1[a] * w2[b]
    for i1 in range(1, q):
        i2 = q - i1
        if i2 >= 1:
            add(i1, i2, +1.0)
    for i1 in range(1, q - 1):
        i2 = q - 1 - i1
        if i2 >= 1:
            add(i1, i2, -1.0)
    keys = [k for k, v in pts.items() if abs(v) > 1e-15]
    X = np.array(keys); w = np.array([pts[k] for k in keys])
    # drop boundary points: the integrand vanishes there (Dirichlet factor)
    keep = (X[:, 0] > 0) & (X[:, 0] < 1) & (X[:, 1] > 0) & (X[:, 1] < 1)
    return X[keep], w[keep]


def gauss_tensor(p):
    x, w = gauss_legendre_1d(p)
    X = np.array([(a, b) for a in x for b in x]); W = np.array([wa * wb for wa in w for wb in w])
    return X, W


def sobol(m, scramble=True, seed=0):
    eng = qmc.Sobol(d=2, scramble=scramble, seed=seed)
    X = eng.random_base2(int(np.log2(m))) if (m & (m - 1)) == 0 else eng.random(m)
    if not scramble:   # unscrambled Sobol includes the origin: shift by half a cell
        X = (X + 0.5 / m) % 1.0
    return X, np.full(len(X), 1.0 / len(X))


def halton(m, scramble=True, seed=0):
    X = qmc.Halton(d=2, scramble=scramble, seed=seed).random(m)
    return X, np.full(m, 1.0 / m)


def _fib(k):
    a, b = 1, 1
    for _ in range(k - 1):
        a, b = b, a + b
    return a


def fibonacci_lattice(k, shift=None, tent=False, seed=0):
    """Rank-1 Fibonacci lattice with m = F_k points; optional random shift and tent
    (baker) transform, which periodises smooth non-periodic integrands."""
    m, g = _fib(k), _fib(k - 1)
    i = np.arange(m)
    X = np.stack([i / m, (i * g / m) % 1.0], axis=1)
    if shift is not None:
        X = (X + np.asarray(shift)) % 1.0
    elif seed is not None:
        X = (X + np.random.default_rng(seed).uniform(size=2)) % 1.0
    if tent:
        X = 1.0 - np.abs(2.0 * X - 1.0)
    return X, np.full(m, 1.0 / m)


def uniform_mc(m, seed=0):
    return np.random.default_rng(seed).uniform(size=(m, 2)), np.full(m, 1.0 / m)


# ------------------------------------------------------------------ mesh rules
def mesh_lattice(N, stride):
    """Uniform sub-lattice of interior nodes (indices i = stride, 2 stride, ...)."""
    idx1 = np.arange(stride, N, stride)
    I, J = np.meshgrid(idx1, idx1, indexing="ij")
    lin = (I.ravel() - 1) * (N - 1) + (J.ravel() - 1)
    return lin, np.full(lin.size, float(stride * stride))


def mesh_all(N):
    n = (N - 1) ** 2
    return np.arange(n), np.ones(n)


def fit_eq_nnls(A, beta, m, batch=64, verbose=False):
    """Non-negative empirical quadrature (Lawson--Hanson style greedy support growth
    with exact NNLS refits, hard support cap m).  A: (rows, n_cand), beta: (rows,).
    Returns weights (n_cand,), support size."""
    n_cand = A.shape[1]
    active = np.zeros(n_cand, dtype=bool)
    w = np.zeros(n_cand)
    res = beta.copy()
    while active.sum() < m:
        grad = A.T @ res
        grad[active] = -np.inf
        nadd = min(batch, m - active.sum())
        new = np.argpartition(-grad, nadd - 1)[:nadd]
        new = new[grad[new] > 0]
        if new.size == 0:
            break
        active[new] = True
        cols = np.flatnonzero(active)
        wa, _ = nnls(A[:, cols], beta, maxiter=20 * len(cols))
        w[:] = 0.0; w[cols] = wa
        active[:] = False; active[cols[wa > 0]] = True
        res = beta - A @ w
        if verbose:
            print(f"    eq support {active.sum()} rel res {np.linalg.norm(res)/np.linalg.norm(beta):.3e}", flush=True)
    return w, int(active.sum())
