"""Wave 2D phase 3 — the latent-stepping ROM arms, the linear controls, and their energies.  numpy, f64.

Everything here is EXACT (zero quadrature): the tables are precomputed once from the bank G (M-orthonormal),
the test modes Phi (sine / cosine, unit M-norm) and the assembled operators.

  Petrov tables (arm A):   B = Phi^T M G,  A = Phi^T M L G (= -Lambda B, gate W1),  C = Phi^T M D_B G
  reduced operators (arm C): Mr = G^T M G (= I),  Kr = -G^T M L G (SPSD),  Dr = G^T M D_B G

Latent state carried: (z_{n-1}, z_n).  dt = DT_SNAP / RS.  s = c dt / 2, a = s^2.

ARM A  -- the incumbent: LSPG (Levenberg-Marquardt) on the exact damped-Newmark residual tested against Phi:
    r(z) = B (h(z) - 2 h_n + h_{n-1}) - a A (h(z) + 2 h_n + h_{n-1}) + s C (h(z) - h_{n-1})
    first step: r(z) = B (h - h_0) - a A (h + h_0) + s C (h - h_0) - dt Phi^T M v_0
ARM C  -- variational (forced Stormer-Verlet on the pulled-back Lagrangian), square Newton:
    F(z) = J_h(z_n)^T [ Mr (h(z) - 2 h_n + h_{n-1}) + c^2 dt^2 Kr h_n + s Dr (h(z) - h_{n-1}) ]
    F'(z) = J_h(z_n)^T (Mr + s Dr) J_h(z)
    first step from (z_0, zdot_0):
    F(z) = J_h(z_0)^T [ Mr (h(z) - h_0 - dt J_h(z_0) zdot_0) + (c^2 dt^2 / 2) Kr h_0 + s Dr (h(z) - h_0) ]
    (design r3, arm C; r2 audit item 2 CORRECT).  CFL: c dt sqrt(lambda_max(Mr^-1 Kr)) <= 2 on the linear
    span; reported, and the head's effective value is measured by W6.
CONTROLS -- POD-K / POD-R Galerkin Crank-Nicolson (three-level, exact projection since G is M-orthonormal),
    and an INDEPENDENT POD-K Stormer-Verlet (for W7: a linear head under arm C must reproduce it).
ENERGIES -- arm C: E_r^n = 1/2 zdot_c^T J^T Mr J zdot_c + c^2/2 h_n^T Kr h_n with the Verlet central-difference
    velocity (u-space: 1/2 |G J zdot|_M^2, exact because Mr = I).  Arm A: the CN-consistent DYNAMIC velocity on
    the decoded full-grid fields (never the kinematic recursion).  POD arms: exact reduced energies.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import wav2d_common as wc
from wav2d_common import Grid, precond


# ----------------------------- numpy head (timed paths never touch JAX) -----------------------------

def _silu(x):
    x = np.asarray(x, dtype=float); sg = np.empty_like(x); pos = x >= 0
    sg[pos] = 1.0 / (1.0 + np.exp(-x[pos])); e = np.exp(x[~pos]); sg[~pos] = e / (1.0 + e)
    return x * sg, sg


class HeadNP:
    """h(z) and J_h(z) in physical coefficient units, from a wav2d_head spec."""
    def __init__(self, spec):
        p = spec["params"]
        self.W = [np.asarray(w) for w, _ in p["mlp"]]; self.b = [np.asarray(b) for _, b in p["mlp"]]
        self.skip = np.asarray(p["skip"]); self.scale = float(spec["scale"])
        self.K = int(spec["K"]); self.R = int(spec["R"])

    def h(self, z):
        x = np.asarray(z, dtype=float).ravel(); z0 = x
        for i in range(len(self.W) - 1):
            x, _ = _silu(x @ self.W[i] + self.b[i])
        return self.scale * (x @ self.W[-1] + self.b[-1] + z0 @ self.skip)

    def hj(self, z):
        z = np.asarray(z, dtype=float).ravel(); x = z; J = np.eye(self.K)
        for i in range(len(self.W) - 1):
            a = x @ self.W[i] + self.b[i]; x, sg = _silu(a)
            J = (self.W[i].T @ J) * (sg * (1.0 + a * (1.0 - sg)))[:, None]
        return self.scale * (x @ self.W[-1] + self.b[-1] + z @ self.skip), self.scale * (self.W[-1].T @ J + self.skip.T)


class LinearHead:
    """h(z) = W z (W: R x K) -- for gate W7 (arm C on a linear head == POD-K Verlet)"""
    def __init__(self, Wm):
        self.Wm = np.asarray(Wm); self.R, self.K = self.Wm.shape

    def h(self, z):
        return self.Wm @ np.asarray(z, dtype=float).ravel()

    def hj(self, z):
        return self.Wm @ np.asarray(z, dtype=float).ravel(), self.Wm.copy()


# ----------------------------- tables -----------------------------

def build_tables(g: Grid, G, n_modes):
    """exact Petrov tables and reduced operators; Phi = the n_modes lowest test modes"""
    kmax = int(np.ceil(np.sqrt(n_modes))) + 2
    Phi, lam, kl = wc.mode_table(g, kmax)
    Phi, lam, kl = Phi[:, :n_modes], lam[:n_modes], kl[:n_modes]
    m = g.mass_diag(); d = g.damping_diag()
    L = wc.assemble_L_independent(g)
    LG = np.asarray(L @ G)
    PM = Phi.T * m[None, :]
    T = dict(Phi=Phi, lam=lam, kl=kl, B=PM @ G, A=PM @ LG, C=PM @ (d[:, None] * G),
             Mr=G.T @ (m[:, None] * G), Kr=-(G.T @ (m[:, None] * LG)), Dr=G.T @ (m[:, None] * (d[:, None] * G)),
             PhiM=PM, L=L.tocsr(), G=G, m=m, d=d)
    w = np.linalg.eigvalsh(T["Kr"])                        # Mr = I to 1e-12 (gate D0), so this is lambda(Mr^-1 Kr)
    T["lam_max_reduced"] = float(np.max(w))
    return T


# ----------------------------- arm A: LSPG on the Petrov residual -----------------------------

def lm_solve(resid, jac, z0, scale, max_iter=60, rtol=1e-12, xtol=1e-12, lam0=1e-6, gtol=1e-8):
    """damped Levenberg-Marquardt on ||r||; relative stop on scale (the Stokes cell's rule)"""
    z = np.asarray(z0, dtype=float).copy(); r = resid(z); val = float(np.linalg.norm(r)); J = jac(z)
    lam = lam0; it = 0; nres = 1
    stop = rtol * scale
    for it in range(1, max_iter + 1):
        if val <= stop:
            break
        H = J.T @ J; gr = J.T @ r
        if np.linalg.norm(gr) <= gtol * np.linalg.norm(J) * val:      # first-order optimality reached
            break
        dz = np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-30 * np.eye(len(z)), -gr)
        zn = z + dz; rn = resid(zn); nres += 1; vn = float(np.linalg.norm(rn))
        if np.isfinite(vn) and vn < val:
            z, r, val = zn, rn, vn; J = jac(z); lam = max(lam / 3.0, 1e-14)
            if np.linalg.norm(dz) <= xtol * (1.0 + np.linalg.norm(z)):
                break
        else:
            lam = min(lam * 10.0, 1e14)
            if lam >= 1e14:
                break
    # first-order optimality: relative gradient ||J^T r|| / (||J||_F ||r||) -- the residual of an overdetermined
    # LSPG problem is NOT small at the optimum; the gradient is
    gnorm = float(np.linalg.norm(J.T @ r) / max(np.linalg.norm(J) * val, 1e-300)) if val > 0 else 0.0
    return z, val, it, nres, gnorm


class ArmA:
    def __init__(self, T, head, c, dt):
        self.T, self.head, self.c, self.dt = T, head, float(c), float(dt)
        self.s = 0.5 * self.c * self.dt; self.a = self.s ** 2
        self.E = T["B"] - self.a * T["A"] + self.s * T["C"]           # multiplies h(z)

    def residual_gen(self, z, hn, hm):
        h = self.head.h(z)
        return (self.T["B"] @ (h - 2 * hn + hm) - self.a * (self.T["A"] @ (h + 2 * hn + hm)) + self.s * (self.T["C"] @ (h - hm)))

    def residual_first(self, z, h0, pv0):
        h = self.head.h(z)
        return (self.T["B"] @ (h - h0) - self.a * (self.T["A"] @ (h + h0)) + self.s * (self.T["C"] @ (h - h0)) - self.dt * pv0)

    def jac(self, z):
        _, J = self.head.hj(z)
        return self.E @ J

    def rollout(self, z0, pv0, n_steps, snap_every, max_iter=60, grad_tol=1e-4, rank_tol=1e-8):   # grad_tol: retraction 7
        """returns latents at the stored snapshots (n_snap+1, K), the full latent history (n_steps+1, K),
        per-step LM stats (iters, n_resid, ||r||/scale, cond J_h, relative gradient), and completion flag.  A step
        COMPLETES only if LM reached first-order optimality (relative gradient <= grad_tol, or ||r||/scale <= 1e-12),
        the latent is finite, and J_h keeps sigma_min/sigma_max >= rank_tol."""
        h0 = self.head.h(z0); scale = max(float(np.linalg.norm(self.T["B"] @ h0)), 1e-300)

        def cond(z):
            sv = np.linalg.svd(self.head.hj(z)[1], compute_uv=False); return float(sv[-1] / sv[0])
        z, val, it, nr, gn = lm_solve(lambda zz: self.residual_first(zz, h0, pv0), self.jac, z0, scale, max_iter=max_iter)
        c1 = cond(z)
        Zs = [np.asarray(z0).copy(), z.copy()]; stats = [(it, nr, val / scale, c1, gn)]
        zm, zn = np.asarray(z0).copy(), z
        hm, hn = h0, self.head.h(zn)
        ok = bool(np.all(np.isfinite(z)) and (gn <= grad_tol or val / scale <= 1e-12) and c1 >= rank_tol)
        for k in range(1, n_steps):
            if not ok:
                break
            z, val, it, nr, gn = lm_solve(lambda zz: self.residual_gen(zz, hn, hm), self.jac, 2 * zn - zm, scale, max_iter=max_iter)
            ck = cond(z) if np.all(np.isfinite(z)) else float("nan")
            ok = bool(np.all(np.isfinite(z)) and (gn <= grad_tol or val / scale <= 1e-12) and ck >= rank_tol)
            Zs.append(z.copy()); stats.append((it, nr, val / scale, ck, gn))
            zm, zn = zn, z; hm, hn = hn, self.head.h(zn)
        Zs = np.array(Zs)
        complete = bool(ok and len(Zs) == n_steps + 1)
        return Zs[::snap_every] if complete else None, Zs, np.array(stats), complete


# ----------------------------- arm C: forced variational Verlet -----------------------------

def newton_solve(F, J, z0, scale, tol=1e-12, accept=1e-10, max_iter=50):
    """Newton with backtracking on ||F||.  Returns (z, ||F||/scale, iters, ok).
    `scale` is a TERM-BASED residual scale (the size of the individual terms of F at the predictor, e.g.
    ||J^T Mr h_n|| + c^2 dt^2 ||J^T Kr h_n||), not ||F(z0)||, so the stop is a backward-error criterion.
    ok = converged to tol, or stalled (backtracking cannot reduce ||F||) at ||F||/scale <= accept."""
    z = np.asarray(z0, dtype=float).copy(); f = F(z); f0 = max(float(scale), 1e-300); it = 0
    for it in range(1, max_iter + 1):
        rel = np.linalg.norm(f) / f0
        if rel <= tol:
            return z, float(rel), it, True
        dz = np.linalg.solve(J(z), -f)
        lam = 1.0; improved = False
        for _ in range(20):
            zn = z + lam * dz; fn = F(zn)
            if np.isfinite(np.linalg.norm(fn)) and np.linalg.norm(fn) < np.linalg.norm(f):
                z, f = zn, fn; improved = True; break
            lam *= 0.5
        if not improved:
            rel = float(np.linalg.norm(f) / f0)
            return z, rel, it, bool(rel <= accept and np.all(np.isfinite(z)))
    rel = float(np.linalg.norm(f) / f0)
    return z, rel, it, bool(rel <= accept and np.all(np.isfinite(z)))


class ArmC:
    def __init__(self, T, head, c, dt, rank_tol=1e-8):
        self.T, self.head, self.c, self.dt = T, head, float(c), float(dt)
        self.s = 0.5 * self.c * self.dt
        self.Mr, self.Kr, self.Dr = T["Mr"], T["Kr"], T["Dr"]
        self.MD = self.Mr + self.s * self.Dr
        self.rank_tol = rank_tol

    def cfl(self):
        return self.c * self.dt * np.sqrt(self.T["lam_max_reduced"])

    def _cond(self, J):
        sv = np.linalg.svd(J, compute_uv=False); return sv[-1] / sv[0]

    def _scale(self, Jn, hn):
        return float(np.linalg.norm(Jn.T @ (self.Mr @ hn)) + self.c ** 2 * self.dt ** 2 * np.linalg.norm(Jn.T @ (self.Kr @ hn)))

    def step(self, zm, zn, hm, hn, Jn):
        f_n = self.c ** 2 * self.dt ** 2 * (self.Kr @ hn)
        F = lambda z: Jn.T @ (self.Mr @ (self.head.h(z) - 2 * hn + hm) + f_n + self.s * (self.Dr @ (self.head.h(z) - hm)))
        J = lambda z: Jn.T @ (self.MD @ self.head.hj(z)[1])
        return newton_solve(F, J, 2 * zn - zm, self._scale(Jn, hn))

    def first_step(self, z0, zdot0):
        h0, J0 = self.head.hj(z0)
        kick = self.dt * (J0 @ zdot0)
        f0 = 0.5 * self.c ** 2 * self.dt ** 2 * (self.Kr @ h0)
        F = lambda z: J0.T @ (self.Mr @ (self.head.h(z) - h0 - kick) + f0 + self.s * (self.Dr @ (self.head.h(z) - h0)))
        J = lambda z: J0.T @ (self.MD @ self.head.hj(z)[1])
        return newton_solve(F, J, z0 + self.dt * zdot0, self._scale(J0, h0))

    def zdot_from_velocity(self, z0, cv0):
        """least-squares zdot_0 from the velocity coefficients: min || J_h(z0) zdot - cv0 ||  (Mr = I)"""
        _, J0 = self.head.hj(z0)
        return np.linalg.lstsq(J0, cv0, rcond=None)[0]

    def rollout(self, z0, zdot0, n_steps, snap_every):
        z1, res, it, ok = self.first_step(np.asarray(z0, dtype=float), np.asarray(zdot0, dtype=float))
        Zs = [np.asarray(z0, dtype=float).copy(), z1.copy()]; stats = [(it, res)]; conds = []
        zm, zn = np.asarray(z0, dtype=float).copy(), z1
        hm = self.head.h(zm); hn, Jn = self.head.hj(zn)
        conds.append(self._cond(Jn)); ok = ok and conds[-1] >= self.rank_tol
        for k in range(1, n_steps):
            if not ok:
                break
            z, res, it, ok = self.step(zm, zn, hm, hn, Jn)
            Zs.append(z.copy()); stats.append((it, res))
            zm, zn = zn, z; hm = hn; hn, Jn = self.head.hj(zn)
            conds.append(self._cond(Jn)); ok = ok and conds[-1] >= self.rank_tol and np.all(np.isfinite(z))
        Zs = np.array(Zs)
        complete = bool(ok and len(Zs) == n_steps + 1)
        return Zs[::snap_every] if complete else None, Zs, np.array(stats), np.array(conds), complete

    def energy_reduced(self, Zs, zdot0=None):
        """E_r at EVERY latent step (len(Zs) values): central-difference velocity inside, the given zdot0 (or a
        forward difference) at k=0 and a backward difference at the end -- so E[0] is E(0) and E[-1] is E(4T)."""
        E = []
        n = len(Zs)
        for k in range(n):
            h, J = self.head.hj(Zs[k])
            if k == 0:
                zd = np.asarray(zdot0, dtype=float) if zdot0 is not None else (Zs[1] - Zs[0]) / self.dt
            elif k == n - 1:
                zd = (Zs[k] - Zs[k - 1]) / self.dt
            else:
                zd = (Zs[k + 1] - Zs[k - 1]) / (2 * self.dt)
            v = J @ zd
            E.append(0.5 * v @ (self.Mr @ v) + 0.5 * self.c ** 2 * h @ (self.Kr @ h))
        return np.array(E)

    def coefficient_history(self, Zs):
        return np.array([self.head.h(z) for z in Zs])


def armC_backward_euler(T, head, c, dt, z0, zdot0, n_steps, snap_every, fixed_point_iters=3):
    """CONTROL integrator (never a result): reduced backward Euler on the manifold,
        z_{n+1} = z_n + dt zdot_{n+1},   J(z_{n+1})^T [ Mr J(z_{n+1}) zdot_{n+1} - Mr J(z_n) zdot_n + dt c^2 Kr h(z_{n+1}) + c dt Dr J(z_{n+1}) zdot_{n+1} ] = 0
    first order in dt and dissipative: its energy must drift (W4 control) and its error must depend on RS (W6 control)."""
    z = np.asarray(z0, dtype=float).copy(); zd = np.asarray(zdot0, dtype=float).copy()
    Zs = [z.copy()]; Er = []
    for k in range(n_steps):
        hz, Jz = head.hj(z); mom = T["Mr"] @ (Jz @ zd)
        zdn = zd.copy()
        for _ in range(fixed_point_iters):
            zn_ = z + dt * zdn; hn_, Jn_ = head.hj(zn_)
            lhs = Jn_.T @ ((T["Mr"] + c * dt * T["Dr"]) @ Jn_)
            rhs = Jn_.T @ (mom - dt * c ** 2 * (T["Kr"] @ hn_))
            zdn = np.linalg.solve(lhs, rhs)
        z = z + dt * zdn; zd = zdn; Zs.append(z.copy())
        hE, JE = head.hj(z); v = JE @ zd
        Er.append(0.5 * v @ (T["Mr"] @ v) + 0.5 * c ** 2 * hE @ (T["Kr"] @ hE))
    Zs = np.array(Zs)
    return Zs[::snap_every], Zs, np.array(Er)


# ----------------------------- linear controls -----------------------------

class PodCN:
    """POD-K Galerkin Crank-Nicolson (three-level, exact projection): q in R^K, first K columns of G"""
    def __init__(self, T, K, c, dt):
        self.K, self.c, self.dt = K, float(c), float(dt)
        s = 0.5 * self.c * self.dt; a = s * s
        Kr, Dr, Mr = T["Kr"][:K, :K], T["Dr"][:K, :K], T["Mr"][:K, :K]
        self.Mr, self.Kr, self.Dr, self.s, self.a = Mr, Kr, Dr, s, a
        self.Aop = Mr + s * Dr + a * Kr                     # (I + sD - aL) with L = -K
        self.lu = np.linalg.inv(self.Aop)                   # K x K, dense

    def rollout(self, q0, qv0, n_steps, snap_every):
        s, a = self.s, self.a
        qm = q0.copy()
        q = self.lu @ ((self.Mr + s * self.Dr) @ qm + self.dt * (self.Mr @ qv0) - a * (self.Kr @ qm))
        Q = [qm.copy(), q.copy()]
        for k in range(1, n_steps):
            qp = self.lu @ (2 * (self.Mr @ q) - self.Mr @ qm - a * (self.Kr @ (2 * q + qm)) + s * (self.Dr @ qm))
            qm, q = q, qp; Q.append(q.copy())
        Q = np.array(Q)
        return Q[::snap_every], Q

    def energy(self, Q):
        """exact reduced energy with the CN-consistent dynamic velocity  qv_{k} = (I+sD)^{-1}[(I-sD) qv_{k-1} - (dt c^2/2) K (q_{k-1}+q_k)]"""
        s = self.s; Pinv = np.linalg.inv(self.Mr + s * self.Dr)
        qv = np.zeros(self.K); E = [0.5 * self.c ** 2 * Q[0] @ (self.Kr @ Q[0])]
        for k in range(1, len(Q)):
            qv = Pinv @ ((self.Mr - s * self.Dr) @ qv - 0.5 * self.dt * self.c ** 2 * (self.Kr @ (Q[k - 1] + Q[k])))
            E.append(0.5 * qv @ (self.Mr @ qv) + 0.5 * self.c ** 2 * Q[k] @ (self.Kr @ Q[k]))
        return np.array(E)


def pod_verlet(T, K, c, dt, q0, qv0, n_steps, snap_every):
    """INDEPENDENT POD-K damped Stormer-Verlet with the half-kick first step (for gate W7):
       Mr (q_{n+1} - 2 q_n + q_{n-1}) + c^2 dt^2 Kr q_n + (c dt/2) Dr (q_{n+1} - q_{n-1}) = 0
       first: Mr (q_1 - q_0 - dt qv0) + (c^2 dt^2/2) Kr q_0 + (c dt/2) Dr (q_1 - q_0) = 0"""
    Mr, Kr, Dr = T["Mr"][:K, :K], T["Kr"][:K, :K], T["Dr"][:K, :K]
    s = 0.5 * c * dt
    Ainv = np.linalg.inv(Mr + s * Dr)
    q1 = Ainv @ (Mr @ (q0 + dt * qv0) - 0.5 * c ** 2 * dt ** 2 * (Kr @ q0) + s * (Dr @ q0))
    Q = [q0.copy(), q1.copy()]; qm, q = q0.copy(), q1
    for k in range(1, n_steps):
        qp = Ainv @ (Mr @ (2 * q - qm) - c ** 2 * dt ** 2 * (Kr @ q) + s * (Dr @ qm))
        qm, q = q, qp; Q.append(q.copy())
    Q = np.array(Q)
    return Q[::snap_every], Q


# ----------------------------- full-grid diagnostics (never timed) -----------------------------

def decode(G, H):
    """coefficient history (T, R) -> fields (T, n)"""
    return np.asarray(H) @ G.T


def dynamic_velocity_energy(g: Grid, T, U_hist, c, dt):
    """arm A / any u-history: CN-consistent dynamic velocity on the decoded FULL-GRID fields and the energy
    E = 1/2 v^T M v - c^2/2 u^T M L u at every step (U_hist (T, n) at the ROM's dt, v_0 = 0 assumed unless given)"""
    m, d, L = T["m"], T["d"], T["L"]
    s = 0.5 * c * dt
    v = np.zeros(U_hist.shape[1]); E = []
    Lu_prev = np.asarray(L @ U_hist[0])
    E.append(0.5 * np.sum(m * v * v) - 0.5 * c ** 2 * np.sum(m * U_hist[0] * Lu_prev))
    for k in range(1, len(U_hist)):
        Lu = np.asarray(L @ U_hist[k])
        v = ((1 - s * d) * v + 0.5 * dt * c ** 2 * (Lu_prev + Lu)) / (1 + s * d)
        E.append(0.5 * np.sum(m * v * v) - 0.5 * c ** 2 * np.sum(m * U_hist[k] * Lu))
        Lu_prev = Lu
    return np.array(E)


def full_grid_residual(g: Grid, T, c, dt, u, un, um, first=False, v0=None):
    """INDEPENDENT full-grid path for gate W0: decode -> the SOLVER'S STENCIL (wav2d_common.lap_fn, i.e. not the
    assembled L the tables were built from; the two operators are certified equal by F0) + damping rows -> project
    with Phi^T M.   r = Phi^T M [ (u - 2 un + um) - a L (u + 2 un + um) + s D_B (u - um) ]
    (first: (u-u0) - aL(u+u0) + sD(u-u0) - dt v0)."""
    import jax.numpy as jnp
    s = 0.5 * c * dt; a = s * s
    lap = wc.lap_fn(g); d = T["d"]
    Lf = lambda x: np.asarray(lap(jnp.asarray(x)))
    if first:
        rf = (u - un) - a * Lf(u + un) + s * d * (u - un) - dt * v0
    else:
        rf = (u - 2 * un + um) - a * Lf(u + 2 * un + um) + s * d * (u - um)
    return T["PhiM"] @ rf


def lspg_optimality_independent(g: Grid, T, head, c, dt, Zf, sample=None):
    """Gate W5 (restated, arm A): at every step of the ROM's latent history the decoded state must be a
    first-order-optimal LSPG solution of the residual formed through the INDEPENDENT full-grid path (solver stencil,
    not the tables):  || J^T r_full || / (||J||_F ||r_full||) with J = (B - aA + sC) J_h(z) built from the Petrov
    projection of the stencil applied to the decoded Jacobian columns.  Returns the per-step relative gradients."""
    import jax.numpy as jnp
    s = 0.5 * c * dt; a = s * s
    lap = wc.lap_fn(g); d = T["d"]; PM = T["PhiM"]; G = T["G"]
    Lf = lambda x: np.asarray(lap(jnp.asarray(x)))
    out = []
    idx = range(1, len(Zf) - 1) if sample is None else [n for n in sample if 1 <= n <= len(Zf) - 2]
    for n in idx:                                             # n+1 is the solved step; n, n-1 its TRUE predecessors
        z = Zf[n + 1]; hz, Jh = head.hj(z); un = G @ head.h(Zf[n]); um = G @ head.h(Zf[n - 1]); u = G @ hz
        rf = PM @ ((u - 2 * un + um) - a * Lf(u + 2 * un + um) + s * d * (u - um))
        # Jacobian through the same independent path, column by column (K columns)
        GJ = G @ Jh
        Jcols = np.stack([PM @ (GJ[:, k] - a * Lf(GJ[:, k]) + s * d * GJ[:, k]) for k in range(Jh.shape[1])], axis=1)
        out.append(np.linalg.norm(Jcols.T @ rf) / max(np.linalg.norm(Jcols) * np.linalg.norm(rf), 1e-300))
    return np.array(out)


def projected_momentum_residual(g: Grid, T, c, dt, U_hist, Phi_M=None):
    """Gate W5 (restated): the ROM's decoded history must satisfy the Petrov-projected damped-Newmark equations
    on the full grid through the solver's stencil: r_n = Phi^T M [ (u^{n+1} - 2u^n + u^{n-1}) - aL(u^{n+1}+2u^n+u^{n-1})
    + sD(u^{n+1}-u^{n-1}) ] for n >= 1, normalised by the term scale ||Phi^T M (u^{n+1}+2u^n+u^{n-1})|| a ||L||-ish:
    we use ||Phi^T M u^n|| (the mass term) as the scale.  Returns the per-step normalised residual norms."""
    import jax.numpy as jnp
    s = 0.5 * c * dt; a = s * s
    lap = wc.lap_fn(g); d = T["d"]; PM = T["PhiM"] if Phi_M is None else Phi_M
    out = []
    for n in range(1, len(U_hist) - 1):
        u, un, um = U_hist[n + 1], U_hist[n], U_hist[n - 1]
        rf = (u - 2 * un + um) - a * np.asarray(lap(jnp.asarray(u + 2 * un + um))) + s * d * (u - um)
        out.append(np.linalg.norm(PM @ rf) / max(np.linalg.norm(PM @ un), 1e-300))
    return np.array(out)


def momentum_balance(g: Grid, T, c, dt, U_hist):
    """gate W5 (arm A, absorbing): with vbar_n = (u^{n+1}-u^n)/dt, v^{n+1} = 2 vbar_n - v^n, u_bar = (u^{n+1}+u^n)/2:
       R_m = M (v^{n+1}-v^n) + c dt M D_B vbar + c^2 dt K_full u_bar,  E^{n+1}-E^n + c dt vbar^T M D_B vbar - vbar^T R_m = 0
    returns the per-step closure residual relative to E^0, and the same with the flux term dropped (control)."""
    m, d, L = T["m"], T["d"], T["L"]
    v = np.zeros(U_hist.shape[1]); out = []; ctrl = []
    def energy(u, vv):
        return 0.5 * np.sum(m * vv * vv) - 0.5 * c ** 2 * np.sum(m * u * np.asarray(L @ u))
    E_prev = energy(U_hist[0], v)
    for k in range(1, len(U_hist)):
        vbar = (U_hist[k] - U_hist[k - 1]) / dt; v_new = 2 * vbar - v; ubar = 0.5 * (U_hist[k] + U_hist[k - 1])
        Rm = m * (v_new - v) + c * dt * m * d * vbar - c ** 2 * dt * m * np.asarray(L @ ubar)
        E_new = energy(U_hist[k], v_new)
        flux = c * dt * np.sum(m * d * vbar * vbar)
        out.append(E_new - E_prev + flux - vbar @ Rm); ctrl.append(E_new - E_prev - vbar @ Rm)
        v, E_prev = v_new, E_new
    E0 = max(abs(energy(U_hist[0], np.zeros_like(v))), 1e-300)
    return np.array(out) / E0, np.array(ctrl) / E0
