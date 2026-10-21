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
            eta = jnp.where(scale > 0, rn / jnp.where(scale > 0, scale, 1.), jnp.where(rn == 0, 0., jnp.inf))
            return c, it, reason, eta

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
        sc, rg = gnorm(U, cn), gnorm(U, rv)
        # A2.4: zero scale with zero residual -> certified (0); zero scale with non-zero residual -> inf
        eta = jnp.where(sc > 0, rg / jnp.where(sc > 0, sc, 1.), jnp.where(rg == 0, 0., jnp.inf))
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


def run_valid(out, nsteps):
    """Validity (DESIGN 3.1 item 1): finite outputs, fields and states, no non-finite step (reason 2). Budget exits
    (reason 0) are not invalid by themselves; they are judged by the certificate (non-converged steps <= 1 %)."""
    st = out['stats']
    rs = np.asarray(st['reason'])[:nsteps]
    ok = int(st['steps']) == nsteps and bool(np.all(rs >= 0) and not np.any(rs == 2))
    for k in ('W', 'w0', 'fields'):
        if k in out:
            ok = ok and bool(np.isfinite(np.asarray(out[k])).all())
    return ok and bool(np.isfinite(np.asarray(st['states'])[:nsteps + 1]).all())


# =============================================================== E5 stepper adapter (DESIGN E5R) ====
# imp-faststep's fscore.py is copied byte-identically into e5/ (PROVENANCE.json); only the tables are built here.

def _e5():
    import sys
    from pathlib import Path
    p = str(Path(__file__).resolve().parent / 'e5')
    if p not in sys.path:
        sys.path.insert(0, p)
    import fscore as F
    return F


def e5_basis_sine(base):
    """(D, V) of Q^T Lambda Q, computed once per base (E5 precontract convention) and reused for every sine table."""
    K = base.Qm.T @ (base.lam[:, None] * base.Qm)
    return jnp.linalg.eigh(.5 * (K + K.T))


def e5_tables(proj, d, aux, base, basis):
    """Diffusion-coordinate tables (Bz, Dz, Pz, D, V, RinvV, VtRm) of an arm for fscore.make_evolve.
    sine: fscore.precontract on the arm's (Gq, Gs, Psi = weighted tested rows) with the shared basis;
    bank: generalised eigenproblem Stiff V = Mass V D with V^T Mass V = I, z = V^T Mass c, c = V z,
          Bz = Gq V, Dz = Gs V, Pz = (w Gq) V  (so dz/dt = -Pz^T[(Bz z)(Dz z)] - nu D z)."""
    F = _e5()
    if proj == 'sine':
        dg = dict(A=base.A, lam=base.lam, Qm=base.Qm, Rm=base.Rm, Gq=d['Gq'], Gs=d['Gs'], Psi=aux['T'])
        tab, chk = F.precontract(dg, basis=basis)
        return tab, chk
    Mm, Km = basis
    Lc = jnp.linalg.cholesky(Mm)
    Linv = jax.scipy.linalg.solve_triangular(Lc, jnp.eye(Mm.shape[0]), lower=True)
    S = Linv @ Km @ Linv.T
    Dg, Wv = jnp.linalg.eigh(.5 * (S + S.T))
    V = Linv.T @ Wv
    tab = dict(Bz=d['Gq'] @ V, Dz=d['Gs'] @ V, Pz=aux['T'] @ V, D=Dg, V=V, RinvV=V, VtRm=V.T @ Mm)
    r = Mm.shape[0]
    chk = dict(D_min=float(Dg.min()), D_max=float(Dg.max()),
               v_orth=float(jnp.max(jnp.abs(V.T @ Mm @ V - jnp.eye(r)))),
               eig_resid=float(jnp.linalg.norm(Km @ V - Mm @ V * Dg[None]) / jnp.linalg.norm(Dg)), qr_orth=0.)
    return jax.block_until_ready(tab), chk


def make_query_e5(L, nrec):
    """Same interface as make_query: init / evolve / decode / fused, with fscore's 'cn' stepper. stats are converted to
    this lane's convention: states (c coordinates, rows 0..S), reason (1 tol, 0 budget, 2 non-finite), eta_rec."""
    import engines as e
    import hops as H
    F = _e5()
    ev = F.make_evolve('cn', F.adv_pre, F.jac_pre)

    def initialize(u0, cold):
        xy, wq, Qc, Rc, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        return jax.scipy.linalg.solve_triangular(Rc, Qc.T @ ui, lower=False)

    def decode(W, G):
        U_ = H.bank_apply(G, W.T).T.reshape(len(W), L - 1, L - 1)
        return jnp.pad(U_, ((0, 0), (1, 1), (1, 1)))

    def evolve(c0, nu, tab, sch):
        W, st = ev(c0, nu, tab, sch)
        Z = st['Z'][:nrec]
        rs = st['reason']
        out = dict(states=Z @ tab['RinvV'].T, it=st['it'], reason=jnp.where(rs == 3, 2, rs),
                   eta_rec=st['tolratio'] * sch['tolf'], steps=st['steps'])
        return W, out

    def query(u0, nu, tab, G, cold, sch):
        w0 = initialize(u0, cold)
        W, st = evolve(w0, nu, tab, sch)
        return dict(fields=decode(W, G), W=W, w0=w0, stats=st)

    return dict(init=jax.jit(initialize), evolve=jax.jit(evolve), decode=jax.jit(decode), fused=jax.jit(query))
