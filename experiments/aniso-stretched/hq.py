"""Degree-p tensor-product Lagrange (GLL) elements for the same problem: the higher-order control (e)."""
import numpy as np, scipy.sparse as sp
import aniso as an


def gll(p):
    if p == 1:
        return np.array([-1.0, 1.0])
    c = np.polynomial.legendre.Legendre.basis(p).deriv()
    return np.r_[-1.0, np.sort(c.roots().real), 1.0]


def ref1d(p, pts):
    """Values and derivatives of the p+1 Lagrange basis functions (GLL nodes) at pts in [-1,1]."""
    xn = gll(p)
    V = np.ones((pts.size, p + 1)); D = np.zeros((pts.size, p + 1))
    for i in range(p + 1):
        others = [j for j in range(p + 1) if j != i]
        den = np.prod([xn[i] - xn[j] for j in others])
        V[:, i] = np.prod([pts - xn[j] for j in others], axis=0) / den
        D[:, i] = sum(np.prod([pts - xn[k] for k in others if k != j], axis=0) for j in others) / den
    return V, D


def problem(family, mu, N, bx, by, p):
    xv, yv = an.nodes(N, bx), an.nodes(N, by)
    hx, hy = np.diff(xv), np.diff(yv)
    q, w = np.polynomial.legendre.leggauss(p + 1)
    V, D = ref1d(p, q)                                   # (nq, p+1)
    xq = xv[:-1, None] + 0.5 * (q + 1) * hx[:, None]     # (N, nq)
    yq = yv[:-1, None] + 0.5 * (q + 1) * hy[:, None]
    a, b, c = an.tensor(family, mu, xq.ravel(), yq.ravel())
    nq = p + 1
    sh = (N, nq, N, nq)
    a, b, c = (t.reshape(sh).transpose(0, 2, 1, 3) for t in (a, b, c))   # (ex,ey,qx,qy)
    f = an.source(mu, xq.ravel(), yq.ravel()).reshape(sh).transpose(0, 2, 1, 3)
    W = w[:, None] * w[None, :]
    jx, jy = hx[:, None, None, None] / 2, hy[None, :, None, None] / 2
    # local basis index (i,k) = x-index i, y-index k ; arrays over (qx,qy)
    ke = (np.einsum("xyqr,qi,rk,qj,rl->xyikjl", a * W * jy / jx, D, V, D, V, optimize=True)
          + np.einsum("xyqr,qi,rk,qj,rl->xyikjl", c * W * jx / jy, V, D, V, D, optimize=True)
          + np.einsum("xyqr,qi,rk,qj,rl->xyikjl", b * W, D, V, V, D, optimize=True)
          + np.einsum("xyqr,qi,rk,qj,rl->xyikjl", b * W, V, D, D, V, optimize=True))
    fe = np.einsum("xyqr,qi,rk->xyik", f * W * jx * jy, V, V, optimize=True)
    n1 = N * p + 1
    gx = (np.arange(N)[:, None] * p + np.arange(p + 1)[None, :])          # (N,p+1)
    loc = (gx[:, None, :, None] * n1 + gx[None, :, None, :]).reshape(N, N, -1)   # (ex,ey,(i,k))
    m = (p + 1) ** 2
    ke = ke.reshape(N, N, m, m)
    rows = np.broadcast_to(loc[..., :, None], ke.shape).ravel()
    cols = np.broadcast_to(loc[..., None, :], ke.shape).ravel()
    A = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n1 * n1, n1 * n1)).tocsr()
    F = np.bincount(loc.ravel(), weights=fe.reshape(N, N, m).ravel(), minlength=n1 * n1)
    inter = np.zeros((n1, n1), bool); inter[1:-1, 1:-1] = True
    idx = np.flatnonzero(inter.ravel())
    return A[idx][:, idx].tocsr(), F[idx], n1


def interp1d(N, beta, p, xe):
    """Sparse matrix evaluating the degree-p piecewise polynomial (1D dofs) at points xe."""
    xv = an.nodes(N, beta)
    e = np.clip(np.searchsorted(xv, xe, side="right") - 1, 0, N - 1)
    t = 2 * (xe - xv[e]) / (xv[e + 1] - xv[e]) - 1
    V, _ = ref1d(p, t)
    r = np.repeat(np.arange(xe.size), p + 1)
    cidx = (e[:, None] * p + np.arange(p + 1)[None, :]).ravel()
    return sp.csr_matrix((V.ravel(), (r, cidx)), shape=(xe.size, N * p + 1))


def to_eval(u, n1, Px, Py):
    U = np.zeros((n1, n1)); U[1:-1, 1:-1] = u.reshape(n1 - 2, n1 - 2)
    return np.asarray(Px @ (Py @ U.T).T)
