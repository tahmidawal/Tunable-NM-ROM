"""imp-geometry C-SEM: nodal spectral elements on the macro-block meshes of geom.Mesh (DESIGN 5.1).

GLL nodes of order p per element, exact geometry (analytic macro maps incl. circular arcs), lumped (GLL) mass, matrix-free
stiffness, advection by GLL collocation (default) or 3/2 over-integration on Gauss points (dealias=True); IMEX SBDF1/2/3
(explicit advection, implicit diffusion); Helmholtz solves by Jacobi-preconditioned CG with extrapolated warm start.
Output: element-local Lagrange interpolation to the scoring lattice (precomputed weights).
"""
from __future__ import annotations

import math

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import geom as G


def gll(p):
    """GLL nodes (ascending), weights and differentiation matrix for order p (p+1 nodes)."""
    from numpy.polynomial import legendre as L
    c = np.zeros(p + 1)
    c[-1] = 1
    inner = L.legroots(L.legder(c)) if p > 1 else np.array([])
    x = np.concatenate([[-1.0], np.sort(inner), [1.0]])
    Pp = L.legval(x, c)
    w = 2.0 / (p * (p + 1) * Pp ** 2)
    n = p + 1
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = Pp[i] / (Pp[j] * (x[i] - x[j]))
    D[0, 0] = -p * (p + 1) / 4.0
    D[-1, -1] = p * (p + 1) / 4.0
    return x, w, D


def lagrange_matrix(nodes, xs):
    """L[k, i] = ell_i(xs[k]) for the Lagrange basis on `nodes` (barycentric)."""
    nodes = np.asarray(nodes)
    xs = np.asarray(xs)
    n = len(nodes)
    bw = np.array([1.0 / np.prod([nodes[i] - nodes[j] for j in range(n) if j != i]) for i in range(n)])
    diff = xs[:, None] - nodes[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-14)
    diff = np.where(exact, 1.0, diff)
    T = bw[None, :] / diff
    Lm = T / T.sum(1, keepdims=True)
    rows = np.any(exact, 1)
    Lm[rows] = exact[rows].astype(float)
    return Lm


def on_boundary(dom, X, Y, tol=1e-9):
    b = (np.abs(X) < tol) | (np.abs(X - G.SIDE) < tol) | (np.abs(Y) < tol) | (np.abs(Y - G.SIDE) < tol)
    if dom == 'L':
        b |= (np.abs(X - G.LC) < tol) & (Y >= G.LC - tol)
        b |= (np.abs(Y - G.LC) < tol) & (X >= G.LC - tol)
    if dom == 'H':
        b |= np.abs(np.hypot(X - G.HC[0], Y - G.HC[1]) - G.HR) < 1e-8
    return b


class SEMSpace:
    def __init__(self, dom, p, k, grade=None, chordal=False, dealias=False):
        self.dom, self.p, self.k, self.grade, self.chordal, self.dealias = dom, p, k, grade, chordal, dealias
        self.mesh = G.Mesh(dom, k, grade, chordal)
        x, w, D = gll(p)
        self.x, self.w, self.Dn = x, w, D
        n = p + 1
        XI, ET = np.meshgrid(x, x, indexing='ij')
        X, Dx, De = self.mesh.map_ref(XI, ET)                     # (E, n, n, 2)
        self.E = X.shape[0]
        key = np.round(X.reshape(-1, 2) * 1e9).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        self.Ng = len(uniq)
        self.conn = inv.reshape(self.E, n, n)
        Xg = np.zeros((self.Ng, 2))
        Xg[inv] = X.reshape(-1, 2)
        self.Xg = Xg
        bd = on_boundary(dom, Xg[:, 0], Xg[:, 1])
        if dom == 'H' and chordal:                                # chordal: the polygon vertices/edges are the boundary
            bd |= self._chord_boundary(Xg)
        self.dm_np = (~bd).astype(float)
        J = Dx[..., 0] * De[..., 1] - Dx[..., 1] * De[..., 0]
        self.minJ = float(np.min(np.abs(J)))
        rx, ry = De[..., 1] / J, -De[..., 0] / J                   # d xi / dx, d xi / dy
        sx, sy = -Dx[..., 1] / J, Dx[..., 0] / J                   # d eta / dx, d eta / dy
        W2 = np.outer(w, w)[None]
        aJ = np.abs(J) * W2
        self.G11 = aJ * (rx * rx + ry * ry)
        self.G12 = aJ * (rx * sx + ry * sy)
        self.G22 = aJ * (sx * sx + sy * sy)
        self.Bl = aJ
        self.rs = (rx + ry, sx + sy)                               # (d/dx + d/dy) = (rx+ry) d/dxi + (sx+sy) d/deta
        Bg = np.zeros(self.Ng)
        np.add.at(Bg, self.conn, self.Bl)
        self.Bg = Bg
        dg = (self.Dn ** 2)
        dK = np.einsum('ai,eaj->eij', dg, self.G11) + np.einsum('bj,eib->eij', dg, self.G22) \
            + 2 * np.diag(self.Dn)[None, :, None] * np.diag(self.Dn)[None, None, :] * self.G12
        Kd = np.zeros(self.Ng)
        np.add.at(Kd, self.conn, dK)
        self.Kd = Kd
        if dealias:
            M = int(math.ceil(1.5 * (p + 1)))
            gq, wq = np.polynomial.legendre.leggauss(M)
            self.Iq = lagrange_matrix(x, gq)                       # (M, n)
            XQ, EQ = np.meshgrid(gq, gq, indexing='ij')
            _, Dxq, Deq = self.mesh.map_ref(XQ, EQ)
            Jq = Dxq[..., 0] * Deq[..., 1] - Dxq[..., 1] * Deq[..., 0]
            self.rsq = ((Deq[..., 1] - Deq[..., 0]) / Jq, (-Dxq[..., 1] + Dxq[..., 0]) / Jq)
            self.Wq = np.abs(Jq) * np.outer(wq, wq)[None]
        self.j = {k_: jnp.asarray(v) for k_, v in dict(conn=self.conn, G11=self.G11, G12=self.G12, G22=self.G22,
                                                       Bl=self.Bl, r=self.rs[0], s=self.rs[1], Bg=self.Bg, Kd=self.Kd,
                                                       dm=self.dm_np, D=self.Dn, Xg=self.Xg).items()}
        if dealias:
            self.j.update(Iq=jnp.asarray(self.Iq), rq=jnp.asarray(self.rsq[0]), sq=jnp.asarray(self.rsq[1]),
                          Wq=jnp.asarray(self.Wq))

    def _chord_boundary(self, Xg):
        b = np.zeros(len(Xg), bool)
        # points on a chord: distance to the chord segment < 1e-9 for the four chords
        for (M, *_r) in self.mesh.elems:
            if getattr(M, 'curved', False):
                a0, _ = M.arc(np.array([0.0]))
                a1, _ = M.arc(np.array([1.0]))
                a0, a1 = a0[0], a1[0]
                d = a1 - a0
                t = np.clip(((Xg - a0) @ d) / (d @ d), 0, 1)
                dist = np.linalg.norm(Xg - (a0 + t[:, None] * d), axis=1)
                b |= dist < 1e-9
        return b

    # ------------------------------------------------------------------------------------------ operators -----
    def gather(self, u):
        return u[self.j['conn']]

    def dss(self, ul):
        return jnp.zeros(self.Ng, ul.dtype).at[self.j['conn']].add(ul)

    def Kloc(self, ul):
        j = self.j
        ur = jnp.einsum('ai,eij->eaj', j['D'], ul)
        us = jnp.einsum('bj,eij->eib', j['D'], ul)
        wr = j['G11'] * ur + j['G12'] * us
        ws = j['G12'] * ur + j['G22'] * us
        return jnp.einsum('ai,eaj->eij', j['D'], wr) + jnp.einsum('bj,eib->eij', j['D'], ws)

    def K(self, u):
        return self.dss(self.Kloc(self.gather(u)))

    def assembled_K(self):
        """Dense assembled stiffness on all global nodes (offline; for the 'eig' solver)."""
        n = self.p + 1
        E = self.E
        eye = np.eye(n * n).reshape(n * n, n, n)
        Kd = np.zeros((self.Ng, self.Ng))
        conn = self.conn.reshape(E, -1)
        for c0 in range(0, n * n, 32):
            basis = jnp.asarray(np.broadcast_to(eye[c0:c0 + 32][None], (E,) + eye[c0:c0 + 32].shape))
            cols = np.asarray(jax.vmap(self.Kloc, in_axes=1, out_axes=1)(basis)).reshape(E, -1, n * n)
            for t in range(cols.shape[1]):
                np.add.at(Kd, (conn, conn[:, c0 + t][:, None]), cols[:, t, :])
        return Kd

    def eig_factor(self):
        """Generalised eigendecomposition on the interior nodes: K v = lambda B v, V^T B V = I (nu-independent)."""
        ii = np.where(self.dm_np > 0)[0]
        Kd = self.assembled_K()[np.ix_(ii, ii)]
        Kd = 0.5 * (Kd + Kd.T)
        bh = 1.0 / np.sqrt(self.Bg[ii])
        lam, Q = np.linalg.eigh(bh[:, None] * Kd * bh[None, :])
        V = bh[:, None] * Q
        return ii, lam, V

    def N(self, u):
        """Lumped-mass advection u (u_x + u_y) as a nodal field (DSS-averaged)."""
        j = self.j
        ul = self.gather(u)
        ur = jnp.einsum('ai,eij->eaj', j['D'], ul)
        us = jnp.einsum('bj,eij->eib', j['D'], ul)
        if not self.dealias:
            Nl = j['Bl'] * ul * (j['r'] * ur + j['s'] * us)
        else:
            Iq = j['Iq']
            up = lambda a: jnp.einsum('pi,eij,qj->epq', Iq, a, Iq)
            fq = j['Wq'] * up(ul) * (j['rq'] * up(ur) + j['sq'] * up(us))
            Nl = jnp.einsum('pi,epq,qj->eij', Iq, fq, Iq)
        return self.dss(Nl) / j['Bg']

    def ic(self, d):
        X = self.j['Xg']
        return self.j['dm'] * G.ic(self.dom, d, X[:, 0], X[:, 1], jnp)

    # ------------------------------------------------------------------------------------------ lattice out ---
    def lattice_interp(self, pts):
        """(elem (P,), Lx (P, n), Ly (P, n)) for points pts (P, 2) inside the domain."""
        P = len(pts)
        elem = -np.ones(P, int)
        ref = np.zeros((P, 2))
        n = self.p + 1
        for e, (M, s0, s1, t0, t1) in enumerate(self.mesh.elems):
            corners = M.map(np.array([s0, s1, s0, s1]), np.array([t0, t0, t1, t1]))[0]
            if M.curved:
                ss = np.linspace(s0, s1, 9)
                arcpts = M.map(ss, np.full(9, t1))[0]
                corners = np.concatenate([corners, arcpts])
            lo, hi = corners.min(0) - 1e-9, corners.max(0) + 1e-9
            cand = np.where((elem < 0) & np.all((pts >= lo) & (pts <= hi), 1))[0]
            if len(cand) == 0:
                continue
            s = np.full(len(cand), 0.5 * (s0 + s1))
            t = np.full(len(cand), 0.5 * (t0 + t1))
            for _ in range(30):
                X, Ms, Mt = M.map(s, t)
                r = X - pts[cand]
                det = Ms[:, 0] * Mt[:, 1] - Ms[:, 1] * Mt[:, 0]
                ds = (Mt[:, 1] * r[:, 0] - Mt[:, 0] * r[:, 1]) / det
                dt_ = (-Ms[:, 1] * r[:, 0] + Ms[:, 0] * r[:, 1]) / det
                s, t = s - ds, t - dt_
            X, _, _ = M.map(s, t)
            ok = (np.linalg.norm(X - pts[cand], axis=1) < 1e-11) & (s >= s0 - 1e-10) & (s <= s1 + 1e-10) \
                & (t >= t0 - 1e-10) & (t <= t1 + 1e-10)
            idx = cand[ok]
            elem[idx] = e
            ref[idx, 0] = 2 * (s[ok] - s0) / (s1 - s0) - 1
            ref[idx, 1] = 2 * (t[ok] - t0) / (t1 - t0) - 1
        assert np.all(elem >= 0), f'{np.sum(elem < 0)} lattice points not located'
        return elem, lagrange_matrix(self.x, ref[:, 0]), lagrange_matrix(self.x, ref[:, 1])


SBDF = {1: (1.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        2: (1.5, (2.0, -0.5, 0.0), (2.0, -1.0, 0.0)),
        3: (11.0 / 6.0, (3.0, -1.5, 1.0 / 3.0), (3.0, -3.0, 1.0))}


def pcg(Aop, b, x0, Minv, tol, maxit):
    bn = jnp.sqrt(jnp.sum(b * b)) + 1e-300
    r = b - Aop(x0)
    z = Minv * r
    p = z
    rz = jnp.sum(r * z)

    def cond(s):
        x, r, p, rz, it = s
        return (jnp.sqrt(jnp.sum(r * r)) > tol * bn) & (it < maxit)

    def body(s):
        x, r, p, rz, it = s
        Ap = Aop(p)
        al = rz / jnp.sum(p * Ap)
        x = x + al * p
        r = r - al * Ap
        z = Minv * r
        rzn = jnp.sum(r * z)
        p = z + (rzn / rz) * p
        return x, r, p, rzn, it + 1
    x, r, p, rz, it = jax.lax.while_loop(cond, body, (x0, r, p, rz, jnp.int32(0)))
    return x, it, jnp.sqrt(jnp.sum(r * r)) <= tol * bn


EIG_MAX = 20000


def make_sem(dom, p, k, nsteps, tol, order=2, grade=None, dealias=False, chordal=False, maxit=500, pts=None,
             n_out=G.N_OUT, solver='pcg', S=None):
    """C-SEM query d -> (outputs (n_out+1, P) on the scoring lattice, stats (total CG its, max its, failed solves)).
    solver 'pcg' (Jacobi-PCG to tol) or 'eig' (exact solve by the offline generalised eigendecomposition)."""
    S = SEMSpace(dom, p, k, grade, chordal, dealias) if S is None else S
    if solver == 'eig':
        if int(S.dm_np.sum()) > EIG_MAX:
            raise ValueError(f'eig solver not admissible: {int(S.dm_np.sum())} interior nodes > {EIG_MAX}')
        ii, lam_e, Ve = S.eig_factor()
        iij, lamj, Vj = jnp.asarray(ii), jnp.asarray(lam_e), jnp.asarray(Ve)
    pts = G.lattice(dom)[0] if pts is None else pts
    elem, Lx, Ly = S.lattice_interp(pts)
    connP = jnp.asarray(S.conn[elem])
    Lx, Ly = jnp.asarray(Lx), jnp.asarray(Ly)
    dm, Bg, Kd = S.j['dm'], S.j['Bg'], S.j['Kd']
    dt = G.T_END / nsteps
    keep = nsteps // n_out
    assert keep * n_out == nsteps
    beta = jnp.array([SBDF[min(o, order)][0] for o in (1, 2, 3)])
    cu = jnp.array([SBDF[min(o, order)][1] for o in (1, 2, 3)])
    cn = jnp.array([SBDF[min(o, order)][2] for o in (1, 2, 3)])

    def out(u):
        return jnp.einsum('pi,pij,pj->p', Lx, u[connP], Ly)

    def q(d):
        nu = d[4]
        u0 = S.ic(d)
        N0 = S.N(u0)

        def step(i, c):
            u, u1, u2, Nn_1, Nn_2, tot, mx, bad = c   # Nn_1, Nn_2 = N at the two previous levels
            Nn = S.N(u)
            o = jnp.minimum(i, order - 1)             # step index i = 0, 1, 2, ... -> order 1, 2, 3 (capped)
            b_, a_, c_ = beta[o], cu[o], cn[o]
            rhs = (a_[0] * u + a_[1] * u1 + a_[2] * u2) / dt - (c_[0] * Nn + c_[1] * Nn_1 + c_[2] * Nn_2)
            b = dm * Bg * rhs
            al = b_ / dt
            Aop = lambda v: dm * (al * Bg * v + nu * S.K(v)) + (1 - dm) * v
            Minv = dm / (al * Bg + nu * Kd) + (1 - dm)
            if solver == 'eig':
                coef = 1.0 / (al + nu * lamj)
                un = jnp.zeros_like(u).at[iij].set(Vj @ (coef * (Vj.T @ b[iij])))
                it, ok = jnp.int32(0), jnp.asarray(True)
            else:
                x0 = jnp.where(i == 0, u, 2 * u - u1)
                un, it, ok = pcg(Aop, b, x0, Minv, tol, maxit)
            return (un, u, u1, Nn, Nn_1, tot + it, jnp.maximum(mx, it), bad + (~ok).astype(jnp.int32))

        def block(c, kb):
            c = jax.lax.fori_loop(kb * keep, (kb + 1) * keep, step, c)
            return c, out(c[0])
        c0 = (u0, u0, u0, N0, N0, jnp.int32(0), jnp.int32(0), jnp.int32(0))
        c, outs = jax.lax.scan(block, c0, jnp.arange(n_out))
        return jnp.concatenate([out(u0)[None], outs]), jnp.stack([c[5], c[6], c[7]])
    return jax.jit(q), S


def mms_helmholtz(dom, p, k, chordal=False, alpha=10.0, beta=1.0, grade=None):
    """MF-SEM: alpha M u + beta K u = M f (weak, GLL quadrature) for the fd.mms_fields solution; max error at the
    scoring lattice."""
    import fd
    S = SEMSpace(dom, p, k, grade, chordal)
    us, f = fd.mms_fields(dom, S.Xg[:, 0], S.Xg[:, 1], alpha, beta)
    dm = S.j['dm']
    b = dm * S.j['Bg'] * jnp.asarray(f)
    Aop = lambda v: dm * (alpha * S.j['Bg'] * v + beta * S.K(v)) + (1 - dm) * v
    Minv = dm / (alpha * S.j['Bg'] + beta * S.j['Kd']) + (1 - dm)
    u, it, ok = jax.jit(lambda b_: pcg(Aop, b_, jnp.zeros_like(b_), Minv, 1e-13, 20000))(b)
    pts = G.lattice(dom)[0]
    if chordal:
        pts = pts[np.hypot(pts[:, 0] - G.HC[0], pts[:, 1] - G.HC[1]) > G.HR + 0.02]   # outside the chordal polygon
    elem, Lx, Ly = S.lattice_interp(pts)
    uu = np.einsum('pi,pij,pj->p', Lx, np.asarray(u)[S.conn[elem]], Ly)
    ue, _ = fd.mms_fields(dom, pts[:, 0], pts[:, 1], alpha, beta)
    return float(np.max(np.abs(uu - ue))), int(it), bool(ok), S.Ng
