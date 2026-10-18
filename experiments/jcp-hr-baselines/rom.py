"""jcp-hr-baselines: POD-LSPG arms (none / ecsw / qdeim), the native damped Gauss-Newton and the common (ours') LM
steppers, the end-to-end query, and the untimed Q1 certification pass (DESIGN.md 3.4, 4, 5, 7; amendment A1.1, A1.2,
A1.7, A1.15).

Arm interface (all f64, shapes static per configuration):
    const = step_const(cn, cm, nu, co, dt, data)          # history terms, once per step
    r, J  = rJ(c, const, nu, co, dt, data)                # reduced residual (k,) and Jacobian (k, r)

  none   r = R_n(Phi c) (all n rows);   J = [a0 I + dt b0 (diag(s(u)) + diag(u) D_h - nu Lap_h)] Phi    (A1.1)
  ecsw   r = sqrt(xi) * R_n(Phi c)|_E;   rows of the same J by stencil algebra on Pst = Phi[stencil(E)]
  qdeim  r = R_B z(c),  z = [a0 c + a1 cn + a2 cm ; -dt nu (b0 c + b1 cn) ; dt (b0 Nt(c) + b1 Nt(cn))],
         B = [Phi, Lap_h Phi, U_p] = Q_B R_B (Householder, A1.2),  Nt(c) = (P^T U_p)^{-1} N_h(Phi c)|_P.
         Stored: T1 = R_B[:, :r], T2 = R_B[:, r:2r], T3E = R_B[:, 2r:] (P^T U_p)^{-1}.

Exit reasons (as t2core): 0 budget, 1 residual tolerance, 2 tiny step, 3 non-finite / damping limit (GN: a stalled backtracking exits with 2),
4 stationary.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import hrcore as H
import t2core as T2          # byte-identical copy of jcp-combined t2core.py (ours' LM: make_lm)

GN_BUDGET = 30
GN_HALVINGS = 8
LM_BUDGET = 600
STAT_TOL = 1e-3              # Q1 stationarity bar (V4-2)
LEVELS = {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.),    # c2common.tol_of('LSPG', .)
          'recover': (1e-12, 1e-14)}                                             # G-full only (A2.11)
ARMS = ('none', 'ecsw', 'qdeim')


def ratio(J, r):
    return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)


# ------------------------------------------------------------------------------------------------- arms -----

def make_arm(kind, d, L, mutation=None):
    """-> (step_const, rJ). mutation 'drop_adv_jac' removes the advection part of J (controls N-fd, N-grad)."""
    sh = H.shape_of(d, L)
    assert kind in ARMS and mutation in (None, 'drop_adv_jac')

    if kind == 'none':
        def Ffull(u, nu):
            g = u.reshape(sh)
            return (H.adv_central(g, L) - nu * H.lap(g, L)).reshape(-1)

        def step_const(cn, cm, nu, co, dt, data):
            un, um = data['Phi'] @ cn, data['Phi'] @ cm
            return co[1] * un + co[2] * um + dt * co[4] * Ffull(un, nu)

        def rJ(c, const, nu, co, dt, data):
            Phi = data['Phi']
            u = Phi @ c
            g = u.reshape(sh)
            s = H.dsum(g, L).reshape(-1)
            r = co[0] * u + const + dt * co[3] * (u * s - nu * H.lap(g, L).reshape(-1))
            adv = s[:, None] * Phi + u[:, None] * data['DPhi']
            if mutation == 'drop_adv_jac':
                adv = 0. * adv
            J = co[0] * Phi + dt * co[3] * (adv - nu * data['LPhi'])
            return r, J

    elif kind == 'ecsw':
        def node_F(c, nu, data):
            us = jnp.einsum('esr,r->es', data['Pst'], c)
            F, _ = H.stencil_F(us, nu, L, d)
            return us, F

        def step_const(cn, cm, nu, co, dt, data):
            usn, Fn = node_F(cn, nu, data)
            um = data['Pst'][:, 0, :] @ cm
            return co[1] * usn[:, 0] + co[2] * um + dt * co[4] * Fn

        def rJ(c, const, nu, co, dt, data):
            us, F = node_F(c, nu, data)
            sw = data['sw']
            r = sw * (co[0] * us[:, 0] + const + dt * co[3] * F)
            dF, dN = H.stencil_dF(us, nu, L, d)
            if mutation == 'drop_adv_jac':
                dF = dF - dN
            J = sw[:, None] * (co[0] * data['Pst'][:, 0, :] + dt * co[3] * jnp.einsum('es,esr->er', dF, data['Pst']))
            return r, J

    else:
        def NP(c, data):
            us = jnp.einsum('psr,r->ps', data['Pst'], c)
            _, N = H.stencil_F(us, 0., L, d)
            return us, N

        def step_const(cn, cm, nu, co, dt, data):
            _, Nn = NP(cn, data)
            return (data['T1'] @ (co[1] * cn + co[2] * cm) - dt * nu * co[4] * (data['T2'] @ cn)
                    + dt * co[4] * (data['T3E'] @ Nn))

        def rJ(c, const, nu, co, dt, data):
            us, N = NP(c, data)
            r = co[0] * (data['T1'] @ c) - dt * nu * co[3] * (data['T2'] @ c) + dt * co[3] * (data['T3E'] @ N) + const
            _, dN = H.stencil_dF(us, 0., L, d)
            if mutation == 'drop_adv_jac':
                dN = 0. * dN
            JN = jnp.einsum('ps,psr->pr', dN, data['Pst'])
            J = co[0] * data['T1'] - dt * nu * co[3] * data['T2'] + dt * co[3] * (data['T3E'] @ JN)
            return r, J

    return step_const, rJ


# ---------------------------------------------------------------------------------------------- solvers ----

def ls_step(J, r):
    """Least-squares Gauss-Newton step: argmin ||J d + r||. Overdetermined (k >= r): Householder QR of J.
    Underdetermined (k < r, e.g. an ECSW support smaller than the rank): the minimum-norm solution via QR of J^T.
    Rank deficiency shows up as a non-finite step (exit reason 3)."""
    k, n = J.shape
    if k >= n:
        Q, Rm = jnp.linalg.qr(J, mode='reduced')
        return -jax.scipy.linalg.solve_triangular(Rm, Q.T @ r, lower=False)
    Q, Rm = jnp.linalg.qr(J.T, mode='reduced')                   # J = Rm^T Q^T
    y = jax.scipy.linalg.solve_triangular(Rm.T, r, lower=True)
    return -Q @ y


def make_gn(rJ, budget=GN_BUDGET, halvings=GN_HALVINGS):
    """Native damped Gauss-Newton (A1.15): QR least-squares step, backtracking by halving until ||r|| decreases."""
    def gn(c0, args, tol, gtol):
        r, J = rJ(c0, *args)
        rn = jnp.linalg.norm(r)
        reason = jnp.where(jnp.isfinite(rn), jnp.where(ratio(J, r) <= gtol, 4, jnp.where(rn <= tol, 1, 0)),
                           3).astype(jnp.int32)

        def body(s):
            c, r, J, rn, it, reason = s
            step = ls_step(J, r)
            ok = jnp.all(jnp.isfinite(step))
            step = jnp.where(ok, step, 0.)

            res_ = lambda cc: rJ(cc, *args)[0]

            def ls_body(t):
                h, _, _, k = t
                cn = c + h * step
                return h * 0.5, cn, jnp.linalg.norm(res_(cn)), k + 1

            def ls_cond(t):
                h, cn, rn2, k = t
                return (k < halvings) & ~(jnp.isfinite(rn2) & (rn2 < rn))
            # trials: the full step, then halvings (at most `halvings`); only residuals are evaluated here
            cn0 = c + step
            _, cn, rn2, k = jax.lax.while_loop(ls_cond, ls_body, (jnp.float64(0.5), cn0, jnp.linalg.norm(res_(cn0)),
                                                                   jnp.int32(0)))
            r2, J2 = rJ(cn, *args)
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            c3 = jnp.where(accept, cn, c)
            r3 = jnp.where(accept, r2, r)
            J3 = jnp.where(accept, J2, J)
            rn3 = jnp.where(accept, rn2, rn)
            tiny = accept & (jnp.linalg.norm(cn - c) <= 1e-14 * (1 + jnp.linalg.norm(c)))
            # no decrease after every halving: stalled (2) if the step was finite, non-finite (3) otherwise
            reason = jnp.where(~ok | ~jnp.isfinite(rn2), 3, jnp.where(~accept, 2, jnp.where(ratio(J3, r3) <= gtol, 4,
                                                     jnp.where(rn3 <= tol, 1, jnp.where(tiny, 2, 0)))))
            return c3, r3, J3, rn3, it + 1, reason.astype(jnp.int32)

        c, r, J, rn, it, reason = jax.lax.while_loop(lambda s: (s[4] < budget) & (s[5] == 0), body,
                                                     (c0, r, J, rn, jnp.int32(0), reason))
        return c, it, reason, jnp.float64(0.)
    return gn


def make_common(rJ, trust, budget=LM_BUDGET):
    """Ours' LM (t2core.make_lm, unchanged) on this arm's residual."""
    def evalJ(w, args):
        return rJ(w, *args)
    lm = T2.make_lm(evalJ, budget, trust, 1e-10)

    def solve(c0, args, tol, gtol, lam0):
        w, rn, it, reason, gn, rej, lam = lm(c0, args, tol, gtol, lam0)
        return w, it, reason, lam
    return solve


# ----------------------------------------------------------------------------------------------- stepper ---

def make_evolve(kind, d, L, solver, nmax, trust=None, mutation=None):
    """evolve(c0, nu, scale, data, sch) -> (W (6, r), C (nmax+1, r) the whole trajectory, stats).
    sch: hrcore.sched(..., nmax=nmax) plus gtol, tolf (traced). mutation 'drop_adv_jac' -> the SOLVER's Jacobian
    lacks its advection part (N-grad); certification (certify()) always uses the correct arm."""
    step_const, rJ_true = make_arm(kind, d, L)
    _, rJ_solve = make_arm(kind, d, L, mutation=mutation)
    res = lambda c, *a: rJ_true(c, *a)[0]
    if solver == 'native':
        solve = make_gn(rJ_solve)
    else:
        assert trust is not None
        solve = make_common(rJ_solve, trust)
    be = jnp.asarray(H.BE_CO, jnp.float64)

    def evolve(c0, nu, scale, data, sch):
        dt = sch['dt']
        r_ = c0.shape[0]
        out0 = jnp.zeros((6, r_), jnp.float64).at[0].set(c0)
        C0 = jnp.zeros((nmax + 1, r_), jnp.float64).at[0].set(c0)
        st0 = dict(it=jnp.full(nmax, -1, jnp.int32), reason=jnp.full(nmax, -1, jnp.int32),
                   lam=jnp.full(nmax, jnp.nan))

        def body(cr):
            k, cv, cp, cpp, lam0, out, C, st = cr
            co = jnp.where(k < sch['nstart'], be, sch['co'])
            const = step_const(cv, cp, nu, co, dt, data)
            args = (const, nu, co, dt, data)
            tol = sch['tolf'] * scale
            if solver == 'native':
                c1, it, reason, _ = solve(cv, args, tol, sch['gtol'])
                lamn = lam0
            else:
                ce = cv + (cv - cp)
                cq = jnp.where(k >= 2, 3. * cv - 3. * cp + cpp, ce)
                cand = jnp.stack((cv, ce, cq))
                rs = jax.vmap(lambda w: jnp.linalg.norm(res(w, *args)))(cand)
                rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
                c1, it, reason, lamn = solve(cand[jnp.argmin(rs)], args, tol, sch['gtol'], lam0)
            st = dict(it=st['it'].at[k].set(it), reason=st['reason'].at[k].set(reason), lam=st['lam'].at[k].set(lamn))
            kk = k + 1
            sl = sch['slot'][kk]
            out = jax.lax.cond(sl >= 0, lambda o: o.at[sl].set(c1), lambda o: o, out)
            return kk, c1, cv, cp, lamn, out, C.at[kk].set(c1), st

        c0_ = (jnp.int32(0), c0, c0, c0, jnp.asarray(1e-6, jnp.float64), out0, C0, st0)
        _, _, _, _, _, out, C, st = jax.lax.while_loop(lambda cr: cr[0] < sch['steps'], body, c0_)
        return out, C, st
    return evolve


def make_tuple_gamma(kind, d, L):
    """gamma of ONE isolated step tuple (history c_n, c_{n-1}; returned state c_{n+1}; the step's coefficients) with a
    fresh Jacobian. Shared by the trajectory certifier and the N-grad control (A4.1)."""
    step_const, rJ = make_arm(kind, d, L)

    def tg(cn, cm, c1, nu, co, dt, data):
        const = step_const(cn, cm, nu, co, dt, data)
        r, J = rJ(c1, const, nu, co, dt, data)
        return ratio(J, r)
    return tg


def make_certifier(kind, d, L, nmax):
    """Untimed Q1 pass (A2.7): for every step n, gamma_n at c_{n+1} with step n's history and coefficients, fresh J."""
    tg = make_tuple_gamma(kind, d, L)
    be = jnp.asarray(H.BE_CO, jnp.float64)

    def cert(C, nu, data, sch):
        dt = sch['dt']

        def one(k):
            kc = jnp.minimum(k, nmax - 1)
            co = jnp.where(kc < sch['nstart'], be, sch['co'])
            return tg(C[kc], C[jnp.maximum(kc - 1, 0)], C[kc + 1], nu, co, dt, data)
        g = jax.lax.map(one, jnp.arange(nmax))
        live = jnp.arange(nmax) < sch['steps']
        return jnp.where(live, g, jnp.nan)
    return jax.jit(cert)


def step_coefficients(sch, n):
    """The coefficients step n (0-based) uses: BE during start-up, else the scheme's (as the stepper)."""
    return jnp.asarray(H.BE_CO, jnp.float64) if n < int(sch['nstart']) else sch['co']


def make_query(kind, d, L, solver, trust=None, mutation=None):
    """End-to-end query: c0 = Phi^T u0 (full field), stepping, decode of the six full-field outputs.
    q(u0, nu, data, sch) -> dict(fields (6, n), W (6, r), C, c0, scale, stats). schedule(scheme, dt, level)."""
    nmax = H.NMAX[d]
    ev = make_evolve(kind, d, L, solver, nmax, trust=trust, mutation=mutation)

    def query(u0, nu, data, sch):
        c0 = data['Phi'].T @ u0
        scale = jnp.linalg.norm(c0)
        W, C, st = ev(c0, nu, scale, data, sch)
        return dict(fields=W @ data['Phi'].T, W=W, C=C, c0=c0, scale=scale, stats=st)
    return jax.jit(query)


def schedule(d, scheme, dt, level):
    g, t_ = LEVELS[level]
    s = H.sched(scheme, dt, nmax=H.NMAX[d])
    s = {k: v for k, v in s.items() if k != 'nsave'}
    return dict(s, gtol=jnp.float64(g), tolf=jnp.float64(t_))


def certify(out, gam, sch, level='prod'):
    """Per-run Q1 evidence (A2.7, COMMON-BRIEF 3): a step is non-stationary iff gamma > 1e-3 or non-finite (no
    residual-tolerance exception). valid = finite trajectory, every step exited (reason >= 0), no reason-3 step, no
    damping saturation (LM lambda finite and below 1e14 at every step).
    For anchors (level tight / tighter) `anchor_ok` additionally needs gamma <= that level's gtol at every step."""
    steps = int(sch['steps'])
    g = np.asarray(gam)[:steps]
    rs = np.asarray(out['stats']['reason'])[:steps]
    lam = np.asarray(out['stats']['lam'])[:steps]
    stationary = np.isfinite(g) & (g <= STAT_TOL)
    C = np.asarray(out['C'])[:steps + 1]
    saturated = int(np.sum(~np.isfinite(lam) | (lam >= 1e14)))          # LM damping at its limit (V4-2 validity)
    valid = bool(np.isfinite(C).all() and np.all(rs >= 0) and not np.any(rs == 3) and saturated == 0)
    gl = LEVELS[level][0]
    return dict(valid=valid, nonstat=int(np.sum(~stationary)), steps=steps, damping_saturated=saturated,
                worst_gamma=float(np.max(np.where(np.isfinite(g), g, np.inf))) if steps else float('nan'),
                anchor_ok=bool(valid and np.all(np.isfinite(g) & (g <= gl))),
                it_sum=int(np.sum(np.asarray(out['stats']['it'])[:steps])),
                reasons=np.bincount(rs[rs >= 0], minlength=5).tolist())
