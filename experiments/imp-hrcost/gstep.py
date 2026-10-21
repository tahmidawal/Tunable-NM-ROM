"""imp-hrcost precontracted Galerkin Crank-Nicolson stepper (DESIGN A1.3), generic in the tables of hrtab.py.

    f(c)   = -P^T [(Gq c) * (Gs c)] + nu L0 c
    rho(c) = c - c_n - dt/2 [f(c) + f(c_n)]        solved by Newton with the exact Jacobian I - dt/2 J_f(c)
    eta    = ||U rho|| / ||U c_n||                 U^T U = Gram (A^T A = Rm^T Rm for sine tests, Mass for bank tests)

This is the lane's own stepper: R and every arm run through it with only the tables swapped. If imp-faststep's
HANDOFF-E6.md stepper is adopted (DESIGN A1.8 / A2), this module is the P2 comparison or the fallback.

Query = cold-start initial fit (p1core contract) + evolve + full-field decode on the run mesh. f64 throughout.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import hrtab as HT
from hrtab import Q  # noqa: F401

NSTEP_MAX = 400
REASONS = {0: 'budget', 1: 'tol', 2: 'nonfinite'}


def sched(dt, tol, budget=50, horizon=.25, spacing=.05):
    steps = int(round(horizon / dt))
    keep = int(round(spacing / dt))
    assert abs(steps * dt - horizon) < 1e-12 and abs(keep * dt - spacing) < 1e-12, dt
    assert steps <= NSTEP_MAX
    return dict(dt=jnp.float64(dt), tol=jnp.float64(tol), steps=jnp.int32(steps), keep=jnp.int32(keep),
                budget=jnp.int32(budget))


def rhs(c, nu, d):
    return -d['P'].T @ ((d['Gq'] @ c) * (d['Gs'] @ c)) + nu * (d['L0'] @ c)


def rhs_jac(c, nu, d):
    u, s = d['Gq'] @ c, d['Gs'] @ c
    return -d['P'].T @ (s[:, None] * d['Gq'] + u[:, None] * d['Gs']) + nu * d['L0']


def gnorm(U, v):
    return jnp.linalg.norm(U @ v)


def make_evolve(nrec=NSTEP_MAX + 1, mutation=None):
    """evolve(c0, nu, d, U, sch) -> (W (6, r), stats incl. states (nrec, r)). mutation: None | 'jzero' (Jacobian without
    the advection part: a chord method that still converges, slower) for probes only."""

    def evolve(c0, nu, d, U, sch):
        r = c0.shape[0]
        dt = sch['dt']
        eye = jnp.eye(r)

        def newton(cn, fn, cprev, k):
            res = lambda c: c - cn - .5 * dt * (rhs(c, nu, d) + fn)
            scale = gnorm(U, cn)
            tol = sch['tol'] * scale
            c0_ = jnp.where(k >= 1, 2. * cn - cprev, cn)

            def body(s):
                c, rv, it, done = s
                Jf = rhs_jac(c, nu, d) if mutation != 'jzero' else nu * d['L0']
                dc = jnp.linalg.solve(eye - .5 * dt * Jf, -rv)
                c2 = c + dc
                r2 = res(c2)
                return c2, r2, it + 1, gnorm(U, r2) <= tol

            rv0 = res(c0_)
            s0 = (c0_, rv0, jnp.int32(0), gnorm(U, rv0) <= tol)
            c, rv, it, done = jax.lax.while_loop(
                lambda s: (~s[3]) & (s[2] < sch['budget']) & jnp.all(jnp.isfinite(s[1])), body, s0)
            rn = gnorm(U, rv)
            fin = jnp.all(jnp.isfinite(c)) & jnp.isfinite(rn)
            reason = jnp.where(~fin, 2, jnp.where(rn <= tol, 1, 0))
            return c, it, reason, rn / (scale + 1e-300)

        out0 = jnp.zeros((6, r)).at[0].set(c0)
        st0 = dict(states=jnp.full((nrec, r), jnp.nan).at[0].set(c0), it=jnp.full(NSTEP_MAX, -1, jnp.int32),
                   reason=jnp.full(NSTEP_MAX, -1, jnp.int32), eta_rec=jnp.full(NSTEP_MAX, jnp.nan))

        def body(carry):
            k, cn, cprev, out, st = carry
            fn = rhs(cn, nu, d)
            c, it, reason, eta = newton(cn, fn, cprev, k)
            st = dict(states=st['states'].at[jnp.minimum(k + 1, nrec - 1)].set(c), it=st['it'].at[k].set(it),
                      reason=st['reason'].at[k].set(reason), eta_rec=st['eta_rec'].at[k].set(eta))
            kk = k + 1
            out = jnp.where(kk % sch['keep'] == 0, out.at[jnp.clip(kk // sch['keep'], 0, 5)].set(c), out)
            return kk, c, cn, out, st

        k, *_r, out, st = jax.lax.while_loop(lambda c: c[0] < sch['steps'], body,
                                             (jnp.int32(0), c0, c0, out0, st0))
        return out, dict(st, steps=k)

    return evolve


def make_query(L, nrec=NSTEP_MAX + 1, mutation=None):
    """Split + fused end-to-end queries (same text as p1core.make_split_query's init / decode)."""
    import engines as e
    import hops as H
    evolve = make_evolve(nrec, mutation)

    def initialize(u0, cold):
        xy, wq, Qc, Rc, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        return jax.scipy.linalg.solve_triangular(Rc, Qc.T @ ui, lower=False)

    def decode(W, G):
        U_ = H.bank_apply(G, W.T).T.reshape(len(W), L - 1, L - 1)
        return jnp.pad(U_, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, d, U, G, cold, sch):
        w0 = initialize(u0, cold)
        W, st = evolve(w0, nu, d, U, sch)
        return dict(fields=decode(W, G), W=W, w0=w0, stats=st)

    return dict(init=jax.jit(initialize), evolve=jax.jit(evolve), decode=jax.jit(decode), fused=jax.jit(query))


def make_certifier():
    """In-job certificate (DESIGN A1.3): from the saved states, every step's CN residual is rebuilt and its Jacobian
    formed by forward-mode AD (jax.jacfwd), not the solver's analytic rhs_jac. Returns per step eta, literal gamma,
    cond(J), ||U rho|| (n = steps)."""

    def one(k, states, nu, d, U, dt):
        cn, c1 = states[k], states[k + 1]
        fn = rhs(cn, nu, d)
        res = lambda c: c - cn - .5 * dt * (rhs(c, nu, d) + fn)
        rv = res(c1)
        J = jax.jacfwd(res)(c1)
        rn = jnp.linalg.norm(rv)
        sv = jnp.linalg.svd(J, compute_uv=False)
        eta = gnorm(U, rv) / (gnorm(U, cn) + 1e-300)
        eta = jnp.where(gnorm(U, cn) > 0, eta, jnp.inf)
        gam = jnp.linalg.norm(J.T @ rv) / (jnp.linalg.norm(J) * rn + 1e-300)
        return eta, gam, sv[0] / sv[-1]

    def cert(states, nu, d, U, dt, nsteps):
        return jax.lax.map(lambda k: one(k, states, nu, d, U, dt), jnp.arange(nsteps))

    return jax.jit(cert, static_argnums=(5,))


def gram_factor(proj, base):
    """U with U^T U = Gram (A1.3): Rm for sine tests, chol(Mass)^T for bank tests."""
    if proj == 'sine':
        return base.Rm
    Mm = base.bank_ops()[0]
    return jnp.linalg.cholesky(Mm).T


def run_valid(stats, nsteps):
    """Validity (DESIGN 3.1 item 1): every step converged to tolerance (reason 1), finite states."""
    rs = np.asarray(stats['reason'])[:nsteps]
    st = np.asarray(stats['states'])[:nsteps + 1]
    return bool(np.all(rs == 1) and np.isfinite(st).all())
