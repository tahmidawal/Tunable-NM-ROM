"""imp-geometry reduced engine for a linear span with Dirichlet-exact functions (DESIGN 5.2, A1).

A `Space` holds, for the full ordered width R_max:
  P  (m, R)   basis values at the arm's evaluation points (bank: composite Gauss rule; POD / ECSW: FD nodes)
  S  (m, R)   (d/dx + d/dy) of the basis there (bank: exact derivative; POD: FD central / Shortley-Weller rows)
  w  (m,)     weights (rule weights; FD: h^2; ECSW: fitted node weights)
  M  (R, R)   mass   (bank: P^T W P on the frozen reference rule of this space; POD: identity)
  K  (R, R)   stiffness (bank: int grad psi_i . grad psi_j; POD: Phi^T (-A_h) Phi h^2, non-symmetric on H)
  Pi (m_i, R), wi (m_i,)  initial-condition projection points / weights (z0 = M^-1 Pi^T (wi u0(x)))
  Pl (P_lat, R) lattice rows (decoder)
The model ODE is  M z' = -nu K z - N(z),  N_k(z) = sum_q w_q theta_k(x_q) u_q s_q,  u = P z, s = S z.

Steppers: 'lspg_cnr'  CN-R residual tested on the nested prefix Theta = first Mt columns (Mt = min(4r, R_max)),
                       Levenberg-Marquardt, tolerances (gtol, tolf); certificate gamma per step;
          'gal_cnr'   Galerkin CN-R, Newton to tolf;
          'gal_cnr_lin' one Newton pass per step (linearised CN);
          'gal_etdrk4' Galerkin ETDRK4 in the eigenbasis of M^-1 K (real or complex).
CN-R = two backward-Euler start steps then Crank-Nicolson (jcp-robust SCHEMES['CNR']).
Nonlinear term in 'point' form (6 m r flops) or 'tensor' form (precontracted, Mt r^2).
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import geom as G

STEPPERS = ('lspg_cnr', 'gal_cnr', 'gal_cnr_lin', 'gal_etdrk4')
TOLS = {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.0), 'tighter': (1e-8, 0.0)}      # LSPG (gtol, tolf), incumbent
GAL_TOLS = {'prod': 1e-10, 'tight': 1e-13, 'tighter': 1e-15}


class Space:
    def __init__(self, P, S, w, M, K, Pi, wi, Pl, name='', meta=None):
        self.P, self.S, self.w = np.asarray(P), np.asarray(S), np.asarray(w)
        self.M, self.K = np.asarray(M), np.asarray(K)
        self.Pi, self.wi, self.Pl = np.asarray(Pi), np.asarray(wi), np.asarray(Pl)
        self.R = self.P.shape[1]
        self.name = name
        self.meta = meta or {}

    def with_weights(self, idx, w):
        """ECSW sub-space: same basis on the node subset idx with weights w (mass / stiffness / IC / decoder kept)."""
        return Space(self.P[idx], self.S[idx], w, self.M, self.K, self.Pi, self.wi, self.Pl,
                     self.name + '-ecsw', dict(self.meta, ecsw_m=len(idx)))


def tensor(Th, Ps, Ss, w, chunk=1024):
    """T[k, i, j] = sum_q w_q Th_qk Ps_qi Ss_qj (precontracted; exact contraction of the discrete rule), accumulated
    over point chunks as (Th w)^T (Ps (x) Ss) so no (q, k, i) intermediate is formed."""
    Th, Ps, Ss, w = (jnp.asarray(x) for x in (Th, Ps, Ss, w))
    m, K = Th.shape
    r1, r2 = Ps.shape[1], Ss.shape[1]

    @jax.jit
    def part(A, B, C):
        BC = (B[:, :, None] * C[:, None, :]).reshape(B.shape[0], -1)
        return jnp.matmul(A.T, BC, precision='highest')
    T = jnp.zeros((K, r1 * r2))
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        A = Th[s:e] * w[s:e, None]
        if e - s < chunk:                        # pad the last chunk to the compiled shape
            pad = chunk - (e - s)
            A = jnp.pad(A, ((0, pad), (0, 0)))
            B, C = jnp.pad(Ps[s:e], ((0, pad), (0, 0))), jnp.pad(Ss[s:e], ((0, pad), (0, 0)))
        else:
            B, C = Ps[s:e], Ss[s:e]
        T = T + part(A, B, C)
    return T.reshape(K, r1, r2)


def etdrk4_coefs(Lam, dt, nc=32):
    """Kassam-Trefethen contour coefficients for diagonal L = Lam (complex ok): returns E, E2, Q, f1, f2, f3."""
    Lam = Lam.astype(jnp.complex128)
    hL = dt * Lam
    rts = jnp.exp(1j * jnp.pi * (jnp.arange(1, 2 * nc + 1) - 0.5) / nc)     # full circle (complex L allowed)
    LR = hL[:, None] + rts[None, :]
    E = jnp.exp(hL)
    E2 = jnp.exp(hL / 2)
    Q = dt * jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=1)
    f1 = dt * jnp.mean((-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1)
    f2 = dt * jnp.mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR ** 3, axis=1)
    f3 = dt * jnp.mean((-4 - 3 * LR - LR ** 2 + jnp.exp(LR) * (4 - LR)) / LR ** 3, axis=1)
    return E, E2, Q, f1, f2, f3


def _phi_series(z, k, terms=24):
    """phi_k(z) = sum_m z^m / (m+k)! (series; used for |z| < 1)."""
    import math as _m
    out = jnp.zeros_like(z)
    for m in range(terms - 1, -1, -1):
        out = out * z + 1.0 / _m.factorial(m + k)
    return out


def _phi_closed(z, k):
    e = jnp.exp(z)
    if k == 1:
        return (e - 1) / z
    if k == 2:
        return (e - 1 - z) / z ** 2
    return (e - 1 - z - z * z / 2) / z ** 3


def etd_coefs_direct(Lam, dt):
    """Independent closed-form / series ETDRK4 coefficients (Cox-Matthews phi functions) for the A3.5 certificate."""
    Lam = Lam.astype(jnp.complex128)
    z = dt * Lam
    small = jnp.abs(z) < 1.0
    zs = jnp.where(small, z, 0.5)
    zl = jnp.where(small, 2.0, z)
    phi = {k: jnp.where(small, _phi_series(zs, k), _phi_closed(zl, k)) for k in (1, 2, 3)}
    zh = z / 2
    smallh = jnp.abs(zh) < 1.0
    phi1h = jnp.where(smallh, _phi_series(jnp.where(smallh, zh, 0.5), 1), _phi_closed(jnp.where(smallh, 2.0, zh), 1))
    E, E2 = jnp.exp(z), jnp.exp(zh)
    Q = dt / 2 * phi1h
    f1 = dt * (phi[1] - 3 * phi[2] + 4 * phi[3])
    f2 = dt * (phi[2] - 2 * phi[3])
    f3 = dt * (-phi[2] + 4 * phi[3])
    return E, E2, Q, f1, f2, f3


def make_query(sp: Space, r, stepper, nsteps, form='point', tol='prod', n_out=G.N_OUT, Mt=None, cert=False,
               maxit=60, perturb_cert=0.0, zero_slice=None, out='lattice', zero_test=None, trace=False):
    """Reduced query d -> (lattice outputs (n_out+1, P_lat), stats). stats (float vector):
    [n_nonstationary, max_gamma, n_steps, n_nonfinite, max_iters, n_damping_saturated, n_unconverged].
    cert=True computes gamma with a fresh Jacobian at every step (LSPG) / the Newton residual ratio (Galerkin).
    perturb_cert: MF-Q1 fault injection (iterate perturbed by this relative size before the certificate).
    zero_slice: MF-TENS fault injection (tensor slice k zeroed)."""
    assert stepper in STEPPERS
    R = sp.R
    assert r <= R
    if stepper == 'lspg_cnr':
        Mt = min(4 * r, R) if Mt is None else Mt
    else:
        Mt = r
    P = jnp.asarray(sp.P[:, :r])
    Sx = jnp.asarray(sp.S[:, :r])
    Th = jnp.asarray(sp.P[:, :Mt])
    w = jnp.asarray(sp.w)
    Mtr = jnp.asarray(sp.M[:Mt, :r])
    Ktr = jnp.asarray(sp.K[:Mt, :r])
    Mrr = np.asarray(sp.M[:r, :r])
    Pi = jnp.asarray(sp.Pi[:, :r])
    wi = jnp.asarray(sp.wi)
    Pl = jnp.asarray(sp.Pl[:, :r])
    Minv_ic = jnp.asarray(np.linalg.inv(Mrr))
    Xi = jnp.asarray(sp.meta['ic_points'])
    dt = G.T_END / nsteps
    keep = nsteps // n_out
    assert keep * n_out == nsteps
    if form == 'tensor':
        T = tensor(Th, P, Sx, w)
        if zero_slice is not None:
            T = T.at[zero_slice].set(0.0)

        def Nt(z):
            return jnp.einsum('kij,i,j->k', T, z, z)

        def dNt(z):
            return jnp.einsum('kij,j->ki', T, z) + jnp.einsum('kij,i->kj', T, z)
    else:
        if zero_test is not None:                       # MF-TENS independent fault: k-th test function removed
            Th = Th.at[:, zero_test].set(0.0)

        def Nt(z):
            return Th.T @ (w * (P @ z) * (Sx @ z))

        def dNt(z):
            u, s = P @ z, Sx @ z
            return Th.T @ ((w * s)[:, None] * P + (w * u)[:, None] * Sx)

    dom = sp.meta['dom']

    def z0_of(d):
        u0 = G.ic(dom, d, Xi[:, 0], Xi[:, 1], jnp)
        return Minv_ic @ (Pi.T @ (wi * u0))

    def decode(z):
        return Pl @ z if out == 'lattice' else z

    if stepper == 'gal_etdrk4':
        Kr = np.asarray(sp.K[:r, :r])
        A = np.linalg.solve(Mrr, Kr)
        lam, V = np.linalg.eig(A)
        cplx = bool(np.max(np.abs(lam.imag)) > 1e-10 * np.max(np.abs(lam)))
        condV = float(np.linalg.cond(V))
        if condV > 1e8:
            raise ValueError(f'ETDRK4 not admissible: cond(V) = {condV:.3e} > 1e8 (A1.6)')
        if not cplx:
            lam, V = lam.real, V.real
        Vi = np.linalg.inv(V)
        Vj, Vij = jnp.asarray(V), jnp.asarray(Vi @ np.linalg.inv(Mrr))
        lamj = jnp.asarray(lam)

        def q(d):
            nu = d[4]
            E, E2, Q, f1, f2, f3 = etdrk4_coefs(-nu * lamj, dt)
            if cert:                                   # certificates only in the untimed replay (A6)
                dco = etd_coefs_direct(-nu * lamj, dt)
                coef_cert = jnp.max(jnp.stack([jnp.max(jnp.abs(c_ - d_) / (jnp.abs(d_) + 1e-300))
                                               for c_, d_ in zip((E, E2, Q, f1, f2, f3), dco)]))
            else:
                dco = (E, E2, Q, f1, f2, f3)
                coef_cert = jnp.asarray(0.0)
            if not cplx:
                E, E2, Q, f1, f2, f3 = (c_.real for c_ in (E, E2, Q, f1, f2, f3))
            z0 = z0_of(d)
            y0 = jnp.linalg.solve(Vj, z0.astype(Vj.dtype)) if cplx else jnp.asarray(Vi) @ z0

            def Ny(y):
                z = (Vj @ y).real if cplx else Vj @ y
                return -(Vij @ Nt(z))

            def st_stages(y):
                Nv = Ny(y)
                a_ = E2 * y + Q * Nv
                Na = Ny(a_)
                b_ = E2 * y + Q * Na
                Nb = Ny(b_)
                c_ = E2 * a_ + Q * (2 * Nb - Nv)
                Nc = Ny(c_)
                return E * y + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3, (a_, b_, c_)

            def st(i, y):
                return st_stages(y)[0]

            nz0 = jnp.linalg.norm(z0)

            dE, dE2, dQ, df1, df2, df3 = (c_.real if not cplx else c_ for c_ in dco)

            def st_indep(y):
                # independent transcription of the Cox-Matthews ETDRK4 update with the closed-form coefficients
                k1 = Ny(y)
                s2 = dE2 * y + dQ * k1
                k2 = Ny(s2)
                s3 = dE2 * y + dQ * k2
                k3 = Ny(s3)
                s4 = dE2 * s2 + dQ * (2 * k3 - k1)
                k4 = Ny(s4)
                return dE * y + df1 * k1 + df2 * (2 * k2 + 2 * k3) + df3 * k4, (s2, s3, s4)

            def defect(x, xi, y):
                return jnp.linalg.norm(x - xi) / (jnp.linalg.norm(xi) + 1e-14 * jnp.linalg.norm(y) + 1e-300)

            def st2(i, c):
                y, bad, blow, dfx = c
                yn, stg = st_stages(y)
                bad = bad + (~jnp.all(jnp.stack([jnp.all(jnp.isfinite(x_)) for x_ in stg]))).astype(jnp.float64)
                if cert:
                    yi, stgi = st_indep(y)
                    d_ = defect(yn, yi, y)
                    for x_, xi_ in zip(stg, stgi):                  # every stage, not only the update (A5)
                        d_ = jnp.maximum(d_, defect(x_, xi_, y))
                    dfx = jnp.maximum(dfx, d_)
                zz = (Vj @ yn).real if cplx else Vj @ yn
                zp = (Vj @ y).real if cplx else Vj @ y
                bad = bad + (~jnp.all(jnp.isfinite(zz))).astype(jnp.float64)
                blow = blow + (jnp.linalg.norm(zz) > 10 * jnp.maximum(nz0, jnp.linalg.norm(zp))).astype(jnp.float64)
                return yn, bad, blow, dfx

            def block(c, _):
                c = jax.lax.fori_loop(0, keep, st2, c)
                z = (Vj @ c[0]).real if cplx else Vj @ c[0]
                return c, decode(z)
            y0c = y0.astype(jnp.complex128) if cplx else y0
            c, outs = jax.lax.scan(block, (y0c, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0)), None,
                                   length=n_out)
            O = jnp.concatenate([decode(z0)[None], outs])
            # stats: [n_nonstationary(=blow-ups), max_gamma, n_steps, n_nonfinite, max_iters, n_sat, n_unconverged]
            # stats: blow-ups (counted as non-stationary), coefficient certificate, steps, non-finite, -, -, -
            # stats[4] = max per-step relative defect between the update and its independent recomputation
            return O, jnp.stack([c[2], coef_cert, jnp.asarray(float(nsteps)), c[1], c[3],
                                 jnp.asarray(0.0), jnp.asarray(0.0)])
        q_ = jax.jit(q)
        q_.info = dict(complex=cplx, condV=condV, Mt=Mt)
        return q_

    gtol, tolf = TOLS[tol] if stepper == 'lspg_cnr' else (0.0, GAL_TOLS[tol])
    one_pass = stepper == 'gal_cnr_lin'

    def resid(z, zn, Nn, nu, th):
        """theta-scheme residual: th = 1 (BE) or 0.5 (CN)."""
        return Mtr @ (z - zn) / dt + th * (nu * (Ktr @ z) + Nt(z)) + (1 - th) * (nu * (Ktr @ zn) + Nn)

    def jac(z, nu, th):
        return Mtr / dt + th * (nu * Ktr + dNt(z))

    def gamma(Jm, rv):
        return jnp.linalg.norm(Jm.T @ rv) / (jnp.linalg.norm(Jm) * jnp.linalg.norm(rv) + 1e-300)

    def solve(zn, nu, th, z_init):
        Nn = Nt(zn)
        r0n = jnp.linalg.norm(resid(z_init, zn, Nn, nu, th))

        if stepper == 'lspg_cnr':
            def cond(s):
                z, lamb, it, done, sat = s
                return (~done) & (it < maxit)

            def body(s):
                z, lamb, it, done, sat = s
                rv = resid(z, zn, Nn, nu, th)
                Jm = jac(z, nu, th)
                g = Jm.T @ rv
                Hm = Jm.T @ Jm
                dz = -jnp.linalg.solve(Hm + lamb * jnp.diag(jnp.diag(Hm)) + 1e-300 * jnp.eye(r), g)
                zt = z + dz
                ok = jnp.linalg.norm(resid(zt, zn, Nn, nu, th)) < jnp.linalg.norm(rv)
                z = jnp.where(ok, zt, z)
                lamb = jnp.where(ok, jnp.maximum(lamb / 3, 1e-12), lamb * 4)
                sat = sat | (lamb > 1e8)
                rv2 = resid(z, zn, Nn, nu, th)
                Jm2 = jac(z, nu, th)
                done = (gamma(Jm2, rv2) <= gtol) | (jnp.linalg.norm(rv2) <= tolf * (1 + r0n))
                return z, lamb, it + 1, done, sat
            z, lamb, it, done, sat = jax.lax.while_loop(
                cond, body, (z_init, jnp.asarray(1e-6), jnp.int32(0), jnp.asarray(False), jnp.asarray(False)))
            return z, it, done, sat, Nn

        def cond(s):
            z, it, done = s
            return (~done) & (it < (1 if one_pass else maxit))

        def body(s):
            z, it, done = s
            rv = resid(z, zn, Nn, nu, th)
            z = z - jnp.linalg.solve(jac(z, nu, th), rv)
            rv2 = resid(z, zn, Nn, nu, th)
            done = jnp.linalg.norm(rv2) <= tolf * (1 + r0n)
            return z, it + 1, done
        z, it, done = jax.lax.while_loop(cond, body, (z_init, jnp.int32(0), jnp.asarray(False)))
        return z, it, done | one_pass, jnp.asarray(False), Nn

    def q(d):
        nu = d[4]
        z0 = z0_of(d)

        def st(i, c):
            z, zp, ns, gmax, nst, nf, itmax, nsat, nunc, _gl = c
            th = jnp.where(i < 2, 1.0, 0.5)
            guess = jnp.where(i == 0, z, 2 * z - zp)
            zn, it, ok, sat, Nn = solve(z, nu, th, guess)
            if cert:
                zc = zn
                if perturb_cert:
                    zc = zn + perturb_cert * jnp.linalg.norm(zn) * jnp.cos(jnp.arange(r) * 1.7 + 0.3) / jnp.sqrt(r)
                rv = resid(zc, z, Nn, nu, th)
                Jm = jac(zc, nu, th)
                if stepper == 'lspg_cnr':
                    gm = gamma(Jm, rv)
                    bad = (gm > 1e-3) | ~jnp.isfinite(gm)
                elif stepper == 'gal_cnr':
                    gm = jnp.linalg.norm(rv) / (jnp.linalg.norm(Mtr @ (zc - z)) / dt
                                                + jnp.linalg.norm(Mtr @ zc) / dt * 1e-14 + 1e-300)
                    bad = (gm > 1e-8) | ~jnp.isfinite(gm)
                else:                                   # linearised CN: residual of the one linear solve
                    Rg = resid(guess, z, Nn, nu, th)
                    Jg = jac(guess, nu, th)
                    gm = jnp.linalg.norm(Jg @ (zc - guess) + Rg) / (jnp.linalg.norm(Rg)
                                                                     + jnp.linalg.norm(Mtr @ zc) / dt * 1e-14 + 1e-300)
                    bad = (gm > 1e-10) | ~jnp.isfinite(gm)
                ns = ns + bad.astype(jnp.float64)
                gmax = jnp.maximum(gmax, jnp.where(jnp.isfinite(gm), gm, jnp.inf))
                glast = gm
            else:
                glast = jnp.asarray(0.0)
            nf = nf + (~jnp.all(jnp.isfinite(zn))).astype(jnp.float64)
            return (zn, z, ns, gmax, nst + 1, nf, jnp.maximum(itmax, it.astype(jnp.float64)),
                    nsat + sat.astype(jnp.float64), nunc + (~ok).astype(jnp.float64), glast)

        def block(c, k):
            c = jax.lax.fori_loop(k * keep, (k + 1) * keep, st, c)
            return c, (decode(c[0]), c[9])
        c0 = (z0, z0) + tuple(jnp.asarray(0.0) for _ in range(8))
        c, (outs, gtr) = jax.lax.scan(block, c0, jnp.arange(n_out))
        O = jnp.concatenate([decode(z0)[None], outs])
        if trace:
            return O, jnp.stack(list(c[2:9])), gtr
        return O, jnp.stack(list(c[2:9]))
    q_ = jax.jit(q)
    q_.info = dict(Mt=Mt)
    return q_


def floor_err(sp: Space, r, Ulat):
    """G-FLOOR: best lattice-norm approximation error of reference lattice fields Ulat (6, P) in span of the sampled
    basis rows Pl[:, :r]: max_k ||U_k - proj U_k|| / ||U_0||."""
    A = sp.Pl[:, :r]
    Q, _ = np.linalg.qr(A)
    E = Ulat[1:] - (Ulat[1:] @ Q) @ Q.T
    return float(np.max(np.linalg.norm(E, axis=1)) / np.linalg.norm(Ulat[0]))
