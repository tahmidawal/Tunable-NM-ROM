"""Competitor convergence on the L-shaped domain (CPU, seconds).

Five-point finite differences for Laplace's equation on the L-shaped domain
Omega = (-1,1)^2 minus [0,1)x(-1,0], with exact Dirichlet data from the corner
singular solution u = r^{2/3} sin(2 theta/3) (theta in (0, 3pi/2)), solved by sparse direct
LU. A smooth control u = exp(x) sin(y) is solved on the same grids. Records the max-norm and
discrete-L2 errors, the fitted convergence order, and the solve time.

Purpose: quantify how a uniform low-order full-order solver converges when the corner
singularity is present (expected order 2/3 in max norm and about 4/3 in L2), against order 2
for the smooth control. This sets the uniform-grid resolution a low-order FOM needs to reach a
given accuracy on this domain. Graded hp-FEM restores exponential convergence; that is
literature, not measured here.

Writes lshape_fd.json next to this file.
"""
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def exact_sing(x, y):
    r = np.hypot(x, y)
    th = np.mod(np.arctan2(y, x), 2 * np.pi)
    return r ** (2 / 3) * np.sin(2 * th / 3)


def exact_smooth(x, y):
    return np.exp(x) * np.sin(y)


def solve(N, exact):
    h = 2.0 / N
    xs = -1 + h * np.arange(N + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    inside = (np.abs(X) < 1 - 1e-12) & (np.abs(Y) < 1 - 1e-12) & ~((X >= -1e-12) & (Y <= 1e-12))
    idx = -np.ones(X.shape, dtype=np.int64)
    idx[inside] = np.arange(inside.sum())
    n = int(inside.sum())
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    I, Jj = np.nonzero(inside)
    k = idx[I, Jj]
    rows.append(k); cols.append(k); vals.append(np.full(n, 4.0))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        In, Jn = I + di, Jj + dj
        nb_in = inside[In, Jn]
        rows.append(k[nb_in]); cols.append(idx[In[nb_in], Jn[nb_in]]); vals.append(np.full(nb_in.sum(), -1.0))
        bd = ~nb_in
        np.add.at(b, k[bd], exact(X[In[bd], Jn[bd]], Y[In[bd], Jn[bd]]))
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    t0 = time.perf_counter()
    u = spla.splu(A).solve(b)
    t = time.perf_counter() - t0
    ue = exact(X[inside], Y[inside])
    err = u - ue
    return dict(N=N, dof=n, max_err=float(np.abs(err).max()),
                l2_err=float(np.sqrt(np.mean(err ** 2))),
                rel_l2=float(np.linalg.norm(err) / np.linalg.norm(ue)),
                solve_s=t)


def order(rows, key):
    Ns = np.array([r['N'] for r in rows], float)
    e = np.array([r[key] for r in rows])
    return float(-np.polyfit(np.log(Ns[-3:]), np.log(e[-3:]), 1)[0])


def main():
    out = {}
    for name, ex in (('singular', exact_sing), ('smooth', exact_smooth)):
        rows = [solve(N, ex) for N in (32, 64, 128, 256, 512, 1024)]
        out[name] = dict(rows=rows, order_max=order(rows, 'max_err'), order_l2=order(rows, 'rel_l2'))
        # resolution needed for a target relative L2 error, extrapolated from the last two rows
        r1, r2 = rows[-2], rows[-1]
        p = np.log(r1['rel_l2'] / r2['rel_l2']) / np.log(2)
        out[name]['N_for_rel_l2'] = {str(t): float(r2['N'] * (r2['rel_l2'] / t) ** (1 / p)) for t in (1e-2, 2e-3, 1e-3, 1e-4)}
    out['note'] = 'five-point FD, sparse LU (scipy splu), single CPU thread timings, h = 2/N'
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1))
    for k in ('singular', 'smooth'):
        print(k, 'order max', round(out[k]['order_max'], 3), 'order relL2', round(out[k]['order_l2'], 3))
        for r in out[k]['rows']:
            print('  ', r)
        print('  N for target', out[k]['N_for_rel_l2'])


if __name__ == '__main__':
    main()
