"""imp-faststep core: precontracted Galerkin steppers for the off-mesh bank ROM in diffusion coordinates
(DESIGN.md sections 3.2-3.4 and amendment A1).

The t2core GAL semi-discretisation  Q^T [A c' + N(c) + nu Lambda A c] = 0,  A = Q Rm,  N(c) = Psi^T[(Gq c)(Gs c)],
is rewritten offline in diffusion coordinates

    Q^T Lambda Q = V D V^T,   z = V^T Rm c,   c = Rm^{-1} V z,
    z' = -g(z),   g(z) = Pz^T[(Bz z)(Dz z)] + nu D z,   Bz = Gq Rm^{-1} V,  Dz = Gs Rm^{-1} V,  Pz = Psi Q V,

so ||z|| = ||A c|| and one advection evaluation costs three (m, r) products. Three steppers are built from the same
right-hand side (generic in the advection provider, so the un-precontracted "twin" runs the same code):

    'cn'   Crank-Nicolson, Newton with the exact Jacobian and LU, predictor 2 z_n - z_{n-1}
    'etd'  ETDRK4 (Cox-Matthews; psjax2d recurrence) with phi coefficients by the 32-point contour mean
    'ros'  one-pass linearised CN: (I + dt/2 J(z_n)) delta = -dt g(z_n)

Certificates (A1.3) use only the ORIGINAL tables (A, Q, Rm, lambda, tested_value) and independently evaluated
coefficients. Every large array is a jit argument; f64 throughout.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / 'jcp-p1-scaling'
if str(P1) not in sys.path:
    sys.path.insert(0, str(P1))
import p1core as T2  # noqa: E402  (sets the vendored paths; enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.scipy.linalg import solve_triangular  # noqa: E402

NMAX = 401                       # state buffer (longest schedule: 400 steps at dt0 / 4 ... dt0/8 needs 400 steps)
MUTATIONS = (None, 'neg_nu', 'f2_as_f1', 'f3_scale', 'newton0', 'jzero', 'half_update', 'late_out', 'wrong_nu')
TOL_CN = {'prod': 1e-10, 'tight': 1e-12, 'tighter': 1e-13}        # arm 2 levels (A1.3)
CN_BUDGET = 30


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


# ------------------------------------------------------------------ ETDRK4 coefficients ----

def etd_coefs(Lk, h, Mc=32):
    """Kassam-Trefethen contour means: text of jcp-fom2d/pscore.etdrk4_coefs (copied; same arithmetic)."""
    Lk = jnp.asarray(Lk)
    acc = [jnp.zeros_like(Lk) for _ in range(4)]
    for j in range(1, Mc + 1):
        r = np.exp(1j * np.pi * (j - 0.5) / Mc)
        LR = h * Lk + r
        eLR = jnp.exp(LR)
        acc[0] = acc[0] + ((jnp.exp(LR / 2) - 1) / LR).real
        acc[1] = acc[1] + ((-4 - LR + eLR * (4 - 3 * LR + LR ** 2)) / LR ** 3).real
        acc[2] = acc[2] + ((2 + LR + eLR * (-2 + LR)) / LR ** 3).real
        acc[3] = acc[3] + ((-4 - 3 * LR - LR ** 2 + eLR * (4 - LR)) / LR ** 3).real
    Q, f1, f2, f3 = (h * a / Mc for a in acc)
    return dict(E=jnp.exp(h * Lk), E2=jnp.exp(h * Lk / 2), Q=Q, f1=f1, f2=f2, f3=f3)


def phi_np(x, k):
    """Independent phi_k(x) = sum_j x^j / (j + k)! (k = 1, 2, 3) in NumPy f64: 40-term Taylor series for |x| < 0.5,
    closed forms otherwise. Used only by the certificates (A1.3), never by the steppers."""
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 0.5
    xs = x[small]
    s = np.zeros_like(xs)
    fact = float(np.prod(np.arange(1, k + 1)))
    term = np.ones_like(xs) / fact
    for j in range(40):
        s = s + term
        term = term * xs / (j + k + 1)
    out[small] = s
    xl = x[~small]
    ex = np.exp(xl)
    if k == 1:
        out[~small] = (ex - 1) / xl
    elif k == 2:
        out[~small] = (ex - 1 - xl) / xl ** 2
    else:
        out[~small] = (ex - 1 - xl - xl ** 2 / 2) / xl ** 3
    return out


def etd_coefs_indep(Lk, h):
    """The same ETDRK4 weights from the closed phi forms (Cox-Matthews: f1 = h(phi1 - 3 phi2 + 4 phi3),
    f2 = h(phi2 - 2 phi3), f3 = h(4 phi3 - phi2), Q = (h/2) phi1(h L / 2))."""
    x = h * np.asarray(Lk, np.float64)
    p1, p2, p3 = (phi_np(x, k) for k in (1, 2, 3))
    return dict(E=np.exp(x), E2=np.exp(x / 2), Q=.5 * h * phi_np(x / 2, 1), f1=h * (p1 - 3 * p2 + 4 * p3),
                f2=h * (p2 - 2 * p3), f3=h * (4 * p3 - p2))


# ------------------------------------------------------------------ precontraction ----

def precontract(dg, mutation=None):
    """dg: the t2core GAL data dict of one (mesh, r, rule) (A, lam, Qm, Rm, Gq, Gs, Psi, ...). Returns the diffusion-
    coordinate tables and the A1.1 checks. mutation 'wrong_congruence' (control C1m): Pz built with Q Rm^{-T}."""
    Qm, Rm, lam = dg['Qm'], dg['Rm'], dg['lam']
    K = Qm.T @ (lam[:, None] * Qm)
    K = .5 * (K + K.T)
    D, V = jnp.linalg.eigh(K)
    RinvV = solve_triangular(Rm, V, lower=False)
    VtRm = V.T @ Rm
    QV = Qm @ V
    if mutation == 'wrong_congruence':
        QV = solve_triangular(Rm, Qm.T, lower=False, trans='T').T @ V
    tab = dict(Bz=dg['Gq'] @ RinvV, Dz=dg['Gs'] @ RinvV, Pz=dg['Psi'] @ QV, D=D, V=V, RinvV=RinvV, VtRm=VtRm)
    tab = jax.block_until_ready(tab)
    r = Rm.shape[0]
    chk = dict(D_min=float(jnp.min(D)), D_max=float(jnp.max(D)),
               qr_orth=float(jnp.max(jnp.abs(Qm.T @ Qm - jnp.eye(r)))),
               eig_resid=float(jnp.linalg.norm(K @ V - V * D[None, :]) / jnp.linalg.norm(D)),
               v_orth=float(jnp.max(jnp.abs(V.T @ V - jnp.eye(r)))))
    s = np.linalg.svd(np.asarray(dg['A']), compute_uv=False)
    chk.update(A_rank=int(np.sum(s > s[0] * 1e-13)), A_cond=float(s[0] / s[-1]))
    assert chk['D_min'] > 0, chk
    return tab, chk


def table_fingerprint(tab):
    """Identity of a table set (C12): shape and three device reductions of every array (p1common.fingerprint idea)."""
    parts = []
    for k in sorted(tab):
        v = jnp.ravel(tab[k])
        st = max(1, v.size // 4096)
        smp = v[::st]
        parts.append(np.asarray([v.size, float(jnp.sum(v)), float(jnp.sum(jnp.abs(v))),
                                 float(jnp.sum(smp * jnp.arange(1, smp.size + 1)))]))
    return sha(np.concatenate(parts))


# ------------------------------------------------------------------ advection providers ----

def adv_pre(z, tab):
    """Precontracted N_z(z) = Pz^T[(Bz z)(Dz z)] and its Jacobian J_N (r, r)."""
    u, s = tab['Bz'] @ z, tab['Dz'] @ z
    return tab['Pz'].T @ (u * s)


def jac_pre(z, tab):
    u, s = tab['Bz'] @ z, tab['Dz'] @ z
    return tab['Pz'].T @ (s[:, None] * tab['Bz'] + u[:, None] * tab['Dz'])


def make_adv_orig(nl):
    """Un-precontracted N_z(z) = V^T Q^T N(c), c = Rm^{-1} V z, through the original tables only (tested_value)."""
    def adv(z, tab):
        dg = tab['orig']
        c = solve_triangular(dg['Rm'], tab['V'] @ z, lower=False)
        return tab['V'].T @ (dg['Qm'].T @ nl(c, dg))
    return adv


# ------------------------------------------------------------------ schedules ----

def sched(dt, horizon=.25, spacing=.05, level='prod'):
    steps, keep = int(round(horizon / dt)), int(round(spacing / dt))
    assert abs(steps * dt - horizon) < 1e-12 and abs(keep * dt - spacing) < 1e-12, dt
    assert steps < NMAX, (steps, NMAX)
    return dict(dt=jnp.float64(dt), steps=jnp.int32(steps), keep=jnp.int32(keep), tolf=jnp.float64(TOL_CN[level]),
                budget=jnp.int32(CN_BUDGET))


# ------------------------------------------------------------------ steppers ----

def make_evolve(kind, adv, jac=None, mutation=None, budget_override=None):
    """evolve(c0, nu, tab, sch) -> (W (6, r) output coefficients c, stats). Generic in the advection provider
    adv(z, tab) (and jac(z, tab) for 'cn'/'ros'; None -> forward-mode autodiff of adv). stats: states Z (NMAX, r)
    (row k = z after k steps), per-step iterations / residual ratio / reason (cn), stage maximum |z| (etd), monitor."""
    assert kind in ('cn', 'etd', 'ros') and mutation in MUTATIONS
    J_N = jac if jac is not None else (lambda z, tab: jax.jacfwd(lambda y: adv(y, tab))(z))

    def evolve(c0, nu, tab, sch):
        dt = sch['dt']
        nu_e = -nu if mutation == 'neg_nu' else nu
        if mutation == 'wrong_nu':
            nu_e = nu * 1.37
        D = tab['D']
        r = D.shape[0]
        z0 = tab['VtRm'] @ c0
        n0 = jnp.linalg.norm(z0) + 1e-300
        if kind == 'etd':
            co = etd_coefs(-nu_e * D, dt)
            if mutation == 'f2_as_f1':
                co = dict(co, f2=co['f1'])
            if mutation == 'f3_scale':
                co = dict(co, f3=co['f3'] * 1.01)
        g = lambda z: adv(z, tab) + nu_e * D * z
        eye = jnp.eye(r)

        def step(k, zn, zp):
            if kind == 'etd':
                Nf = lambda z: -adv(z, tab)
                Nv = Nf(zn)
                a = co['E2'] * zn + co['Q'] * Nv
                Na = Nf(a)
                b = co['E2'] * zn + co['Q'] * Na
                Nb = Nf(b)
                cc = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
                Nc = Nf(cc)
                zn1 = co['E'] * zn + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3']
                smax = jnp.max(jnp.abs(jnp.stack((a, b, cc, zn1))))
                return zn1, jnp.int32(0), jnp.float64(0.), jnp.int32(1), smax
            if kind == 'ros':
                Jm = eye if mutation == 'jzero' else eye + .5 * dt * (J_N(zn, tab) + jnp.diag(nu_e * D))
                delta = jnp.linalg.solve(Jm, -dt * g(zn))
                zn1 = zn + (.5 * delta if mutation == 'half_update' else delta)
                return zn1, jnp.int32(1), jnp.float64(0.), jnp.int32(1), jnp.max(jnp.abs(zn1))
            # cn: Newton on rho(z) = z - zn + dt/2 (g(z) + g(zn))
            const = zn - .5 * dt * g(zn)
            rho = lambda z: z - const + .5 * dt * g(z)
            tol = sch['tolf'] * jnp.linalg.norm(zn)
            zi = jnp.where(k >= 1, 2. * zn - zp, zn)
            bud = 0 if mutation == 'newton0' else (sch['budget'] if budget_override is None else budget_override)

            def cond(s):
                z, rn, it = s
                return (it < bud) & (rn > tol) & jnp.isfinite(rn)

            def body(s):
                z, rn, it = s
                Jm = eye + .5 * dt * (J_N(z, tab) + jnp.diag(nu_e * D))
                z2 = z - jnp.linalg.solve(Jm, rho(z))
                return z2, jnp.linalg.norm(rho(z2)), it + 1
            z1, rn, it = jax.lax.while_loop(cond, body, (zi, jnp.linalg.norm(rho(zi)), jnp.int32(0)))
            reason = jnp.where(~jnp.isfinite(rn), 3, jnp.where(rn <= tol, 1, 0)).astype(jnp.int32)
            return z1, it, rn / (tol + 1e-300), reason, jnp.max(jnp.abs(z1))

        Z0 = jnp.full((NMAX, r), jnp.nan).at[0].set(z0)
        st0 = dict(Z=Z0, it=jnp.full(NMAX, -1, jnp.int32), tolratio=jnp.full(NMAX, jnp.nan),
                   reason=jnp.full(NMAX, -1, jnp.int32), stage_max=jnp.float64(0.), growth=jnp.float64(1.))
        out0 = jnp.zeros((6, r), jnp.float64).at[0].set(z0)

        def body(c):
            k, zn, zp, out, st = c
            z1, it, tr, reason, smax = step(k, zn, zp)
            kk = k + 1
            st = dict(Z=st['Z'].at[kk].set(z1), it=st['it'].at[k].set(it), tolratio=st['tolratio'].at[k].set(tr),
                      reason=st['reason'].at[k].set(reason),
                      stage_max=jnp.maximum(st['stage_max'], jnp.where(jnp.isfinite(smax), smax, jnp.inf)),
                      growth=jnp.maximum(st['growth'], jnp.where(jnp.isfinite(jnp.linalg.norm(z1)),
                                                                 jnp.linalg.norm(z1) / n0, jnp.inf)))
            store = (kk % sch['keep']) == 0
            idx = jnp.clip(kk // sch['keep'], 0, 5)
            val = zn if mutation == 'late_out' else z1
            out = jnp.where(store, out.at[idx].set(val), out)
            return kk, z1, zn, out, st

        k, _, _, outz, st = jax.lax.while_loop(lambda c: c[0] < sch['steps'], body,
                                               (jnp.int32(0), z0, z0, out0, st0))
        W = outz @ tab['RinvV'].T                     # c = Rm^{-1} V z for the six outputs
        return W, dict(st, steps=k, z_out=outz)
    return evolve


# ------------------------------------------------------------------ certificates (original tables) ----

def make_orig_rhs(nl):
    """g_orig(z) = V^T Q^T [N(c) + nu Lambda A c], c = Rm^{-1} V z, through the original tables (A1.3)."""
    def g(z, nu, dg, V):
        c = solve_triangular(dg['Rm'], V @ z, lower=False)
        return V.T @ (dg['Qm'].T @ (nl(c, dg) + nu * dg['lam'] * (dg['A'] @ c)))
    return g


def make_cert_cn(nl):
    """Fresh GAL-CN residual of every saved step in c coordinates (t2core GAL residual, CN coefficients):
    r_k = Q^T [A (c_{k+1} - c_k) + dt/2 (F(c_{k+1}) + F(c_k))], F(c) = N(c) + nu lam A c; returns
    (||r_k||, ||A c_k||, gamma_k of the fresh autodiff Jacobian, recorded only)."""
    def F(c, nu, dg):
        return nl(c, dg) + nu * dg['lam'] * (dg['A'] @ c)

    def one(k, Z, nu, dg, V, dt):
        ck = solve_triangular(dg['Rm'], V @ Z[k], lower=False)
        c1 = solve_triangular(dg['Rm'], V @ Z[k + 1], lower=False)
        res = lambda c: dg['Qm'].T @ (dg['A'] @ (c - ck) + .5 * dt * (F(c, nu, dg) + F(ck, nu, dg)))
        rv = res(c1)
        J = jax.jacfwd(res)(c1)
        rn = jnp.linalg.norm(rv)
        return rn, jnp.linalg.norm(dg['A'] @ ck), jnp.linalg.norm(J.T @ rv) / (jnp.linalg.norm(J) * rn + 1e-300)

    def cert(Z, nu, dg, V, dt, steps):
        return jax.lax.map(lambda k: one(k, Z, nu, dg, V, dt), jnp.arange(steps))
    return jax.jit(cert, static_argnums=(5,))


def make_cert_etd(nl):
    """Per-step reconstruction (A1.3): from saved z_k, one ETDRK4 step with the advection through the ORIGINAL tables
    (make_adv_orig) and the independent coefficients `co` (etd_coefs_indep on the host); returns
    ||z_{k+1}^rec - z_{k+1}|| / ||z_{k+1}|| for every step."""
    adv = make_adv_orig(nl)

    def cert(Z, dg, V, co, steps):
        tb = dict(orig=dg, V=V)
        Nf = lambda z: -adv(z, tb)

        def one_(k):
            zn = Z[k]
            Nv = Nf(zn)
            a = co['E2'] * zn + co['Q'] * Nv
            Na = Nf(a)
            b = co['E2'] * zn + co['Q'] * Na
            Nb = Nf(b)
            cc = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
            Nc = Nf(cc)
            z1 = co['E'] * zn + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3']
            return jnp.linalg.norm(z1 - Z[k + 1]) / (jnp.linalg.norm(Z[k + 1]) + 1e-300)
        return jax.lax.map(one_, jnp.arange(steps))
    return jax.jit(cert, static_argnums=(4,))


def make_cert_ros(nl, adv_tab_jac=jac_pre):
    """Per step: fresh autodiff Jacobian J^f of g_orig at saved z_k; (i) ||J_used - J^f||_F / ||J^f||_F with J_used
    from the precontracted tables; (ii) the saved increment's linear residual
    ||(I + dt/2 J^f) delta_k + dt g^f(z_k)|| / ||dt g^f(z_k)||."""
    g = make_orig_rhs(nl)

    def cert(Z, nu, dg, V, tab, dt, steps):
        r = Z.shape[1]

        def one_(k):
            zk = Z[k]
            gf = g(zk, nu, dg, V)
            Jf = jax.jacfwd(lambda y: g(y, nu, dg, V))(zk)
            Ju = adv_tab_jac(zk, tab) + jnp.diag(nu * tab['D'])
            dk = Z[k + 1] - zk
            lin = (jnp.eye(r) + .5 * dt * Jf) @ dk + dt * gf
            return (jnp.linalg.norm(Ju - Jf) / (jnp.linalg.norm(Jf) + 1e-300),
                    jnp.linalg.norm(lin) / (jnp.linalg.norm(dt * gf) + 1e-300))
        return jax.lax.map(one_, jnp.arange(steps))
    return jax.jit(cert, static_argnums=(6,))


def rhs_agreement(Zs, nu, dg, tab, nl):
    """C1 / fresh-RHS check: max_k ||g_pre(z_k) - g_orig(z_k)|| / ||g_orig(z_k)|| over the rows of Zs."""
    g = make_orig_rhs(nl)
    f = jax.jit(lambda z, nu, dg, tab: (adv_pre(z, tab) + nu * tab['D'] * z, g(z, nu, dg, tab['V'])))
    worst = 0.
    for z in Zs:
        a, b = f(jnp.asarray(z), nu, dg, tab)
        worst = max(worst, float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300)))
    return worst
