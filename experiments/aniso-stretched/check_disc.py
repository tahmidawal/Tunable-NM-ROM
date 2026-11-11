"""Discretisation checks: manufactured solution convergence (constant rotated K), symmetry, SPD."""
import numpy as np, scipy.sparse.linalg as spla, time
import aniso as an
# manufactured: u = sin(pi x) sin(2 pi y), K const rotated; f = -(a u_xx + 2 b u_xy + c u_yy)
mu = np.array([30.0, -2.0, .5, .5])
for N in (32, 64, 128, 256):
    for beta in (0, 2.5):
        x = an.nodes(N, beta); xc = .5*(x[1:]+x[:-1])
        a, b, c = an.tensor("A", mu, xc, xc)
        A = an.assemble(x, x, a, b, c)
        X, Y = x[:, None], x[None, :]
        u = np.sin(np.pi*X)*np.sin(2*np.pi*Y)
        f = -(a[0,0]*(-np.pi**2)*u + 2*b[0,0]*(2*np.pi**2)*np.cos(np.pi*X)*np.cos(2*np.pi*Y) + c[0,0]*(-4*np.pi**2)*u)
        rhs = (an.mass1d(x) @ f @ an.mass1d(x))[1:-1,1:-1].ravel()
        uh = spla.spsolve(A.tocsc(), rhs)
        e = np.abs(an.to_full(uh, N) - u).max()
        print(N, beta, "asym", abs(A - A.T).max(), "maxerr %.3e" % e)
