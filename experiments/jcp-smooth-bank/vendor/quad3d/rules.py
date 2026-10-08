"""quadrature-burgers3d: the off-mesh quadrature rules on the unit cube (NumPy only).

The generators are copied from Hari's study (external/quadrature-study-2026-09-30/quadrature/nmrom/dim3.py and
nmrom/quadrature.py, imported into this repository at commit 77718e548), unchanged in their mathematics:
tensor Gauss-Legendre, Smolyak sparse grid with nested Clenshaw-Curtis (boundary points dropped), scrambled Sobol,
rank-1 lattices with a uniform random shift, the CBC generating vector (P_2 criterion, product weights 1) and the
Korobov search. Every rule has weights summing to one (the off-mesh term is N^{3/2} sum_q w_q psi(x_q) F(x_q)).

`python rules.py --out rules/rules.npz` writes every rule of DESIGN.md section 3 plus the continuum-target rules, and
rules/rules.json with names, sizes, generating vectors, P_2 merits, weight sums and per-rule SHA256. The rules are
generated once, locally, and committed: the cluster never regenerates them (no scipy dependency, no drift).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ 1D rules (Hari, nmrom/quadrature.py)
def gauss_legendre_1d(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def clenshaw_curtis_1d(n):
    """n-point Clenshaw--Curtis on [0,1] (nested for n = 2^l + 1); n=1 is the midpoint."""
    if n == 1:
        return np.array([0.5]), np.array([1.0])
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    w = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(1, (n - 1) // 2 + 1):
            b = 1.0 if 2 * j != n - 1 else 0.5
            s += b / (4 * j * j - 1) * np.cos(2 * j * np.pi * i / (n - 1))
        w[i] = 2.0 / (n - 1) * (1.0 - 2.0 * s)
    w[0] /= 2
    w[-1] /= 2
    x = 0.5 * (x[::-1] + 1.0)
    w = 0.5 * w[::-1]
    return x, w


# ------------------------------------------------------------------ 3D rules (Hari, nmrom/dim3.py)
def gauss_tensor3(p):
    x, w = gauss_legendre_1d(p)
    X = np.stack(np.meshgrid(x, x, x, indexing='ij'), -1).reshape(-1, 3)
    W = np.einsum('i,j,k->ijk', w, w, w).ravel()
    return X, W


def _cc_size(i):
    return 1 if i == 1 else 2 ** (i - 1) + 1


def smolyak3(level):
    """Smolyak, nested Clenshaw--Curtis, A(q,3) = sum_{q-2<=|i|<=q} (-1)^{q-|i|} C(2,q-|i|) Q_i1 x Q_i2 x Q_i3."""
    q = level + 3
    pts, rules = {}, {}

    def r1(i):
        if i not in rules:
            rules[i] = clenshaw_curtis_1d(_cc_size(i))
        return rules[i]
    for i1 in range(1, q + 1):
        for i2 in range(1, q + 1):
            for i3 in range(1, q + 1):
                s = i1 + i2 + i3
                if s < q - 2 or s > q:
                    continue
                coef = (-1) ** (q - s) * comb(2, q - s)
                x1, w1 = r1(i1)
                x2, w2 = r1(i2)
                x3, w3 = r1(i3)
                for a in range(len(x1)):
                    for b in range(len(x2)):
                        for c in range(len(x3)):
                            key = (round(x1[a], 12), round(x2[b], 12), round(x3[c], 12))
                            pts[key] = pts.get(key, 0.0) + coef * w1[a] * w2[b] * w3[c]
    keys = sorted(k for k, v in pts.items() if abs(v) > 1e-15)
    X = np.array(keys)
    w = np.array([pts[k] for k in keys])
    keep = np.all((X > 0) & (X < 1), axis=1)
    return X[keep], w[keep]


def sobol3(m, seed=0):
    from scipy.stats import qmc
    X = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(int(np.log2(m)))
    return X, np.full(len(X), 1.0 / len(X))


def rank1_lattice3(n, z, seed=0):
    i = np.arange(n)
    X = (np.outer(i, z) / n) % 1.0
    shift = np.random.default_rng(seed).uniform(size=3)
    X = (X + shift) % 1.0
    return X, np.full(n, 1.0 / n)


def p2_merit3(z, n):
    k = np.arange(n)
    b2 = lambda x: x * x - x + 1.0 / 6.0
    prod = np.ones(n)
    for j in range(3):
        prod = prod * (1.0 + 2 * np.pi ** 2 * b2(((k * int(z[j])) % n) / n))
    return float(prod.mean() - 1.0)


def cbc_lattice_vector3(n, chunk=256):
    """Component-by-component generating vector minimising P_2 (Sloan--Kuo--Joe), candidates coprime to n."""
    k = np.arange(n)
    b2 = lambda x: x * x - x + 1.0 / 6.0
    cand = np.array([a for a in range(1, n) if np.gcd(a, n) == 1])
    z = [1]
    prod = 1.0 + 2 * np.pi ** 2 * b2(k / n)
    best = None
    for j in range(1, 3):
        best, best_a = np.inf, None
        for a0 in range(0, len(cand), chunk):
            A = cand[a0:a0 + chunk]
            x = ((k[None, :] * A[:, None]) % n) / n
            P = (prod[None, :] * (1.0 + 2 * np.pi ** 2 * b2(x))).mean(1) - 1.0
            i = int(np.argmin(P))
            if P[i] < best:
                best, best_a = float(P[i]), int(A[i])
        z.append(best_a)
        prod = prod * (1.0 + 2 * np.pi ** 2 * b2(((k * best_a) % n) / n))
    return np.array(z), best


def korobov_search3(n, chunk=256):
    k = np.arange(n)
    best, best_a = np.inf, None
    b2 = lambda x: x * x - x + 1.0 / 6.0
    f1 = 1.0 + 2 * np.pi ** 2 * b2(k / n)
    for a0 in range(1, n, chunk):
        A = np.arange(a0, min(n, a0 + chunk))
        x2 = ((k[None, :] * (A % n)[:, None]) % n) / n
        x3 = ((k[None, :] * ((A * A) % n)[:, None]) % n) / n
        P = (f1[None, :] * (1.0 + 2 * np.pi ** 2 * b2(x2)) * (1.0 + 2 * np.pi ** 2 * b2(x3))).mean(1) - 1.0
        j = int(np.argmin(P))
        if P[j] < best:
            best, best_a = float(P[j]), int(A[j])
    return np.array([1, best_a, (best_a * best_a) % n]), best


# ------------------------------------------------------------------ the lane's rule set (DESIGN.md section 3)
HARI_CBC = {4096: (1, 1557, 1741), 32768: (1, 12031, 7247)}     # SUMMARY.md section 9 (reproduction check)


def build_all(log=print):
    rules, meta = {}, {}

    def add(name, family, X, w, role, **extra):
        X = np.ascontiguousarray(X, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == 3 and len(w) == len(X)
        assert np.all((X >= 0) & (X <= 1)), name
        rules[name] = (X, w)
        h = hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest()
        meta[name] = dict(family=family, m=int(len(w)), role=role, weight_sum=float(w.sum()), w_min=float(w.min()),
                          sha256=h, **extra)
        log(f'{name}: m={len(w)} wsum={w.sum():.15f} wmin={w.min():.3e} {extra}')

    for n in (256, 4096, 8192, 16384, 32768):
        z, P = cbc_lattice_vector3(n)
        if n in HARI_CBC:
            assert tuple(int(v) for v in z) == HARI_CBC[n], (n, z)
        role = 'must-fail control (far too few points)' if n == 256 else ('ladder' if n == 8192 else 'arm')
        add(f'lat{n}', 'cbc_lattice', *rank1_lattice3(n, z, seed=0), role, z=[int(v) for v in z], p2=P,
            p2_check=p2_merit3(z, n), shift_seed=0)
    z, P = korobov_search3(16381)
    add('kor16381', 'korobov', *rank1_lattice3(16381, z, seed=0), 'arm', z=[int(v) for v in z], p2=P, shift_seed=0)
    for p in (16, 24, 32):
        add(f'gl{p}', 'gauss_tensor', *gauss_tensor3(p), 'arm', p=p)
    add('sob16384', 'sobol', *sobol3(16384, seed=0), 'arm (1/m reference)', scramble_seed=0)
    add('smol8', 'smolyak_cc', *smolyak3(8), 'must-fail control', level=8)
    for p in (64, 80):
        add(f'gl{p}', 'gauss_tensor', *gauss_tensor3(p), 'continuum target' if p == 80 else 'continuum check', p=p)
    return rules, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'rules' / 'rules.npz'))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rules, meta = build_all()
    np.savez(out, **{f'{k}_X': v[0] for k, v in rules.items()}, **{f'{k}_w': v[1] for k, v in rules.items()})
    (out.parent / 'rules.json').write_text(json.dumps(dict(rules=meta, npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),
                                                      indent=1) + '\n')


if __name__ == '__main__':
    main()
