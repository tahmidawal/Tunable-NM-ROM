"""jcp-p1-scaling component kernels (DESIGN 4.2, A1.2, A1.10).

Two families, each component a separately jitted function timed as its own synchronised dispatch:
  * solver sub-computations (A1.2 accounting), text replicated from p1core.make_evolve / make_lm (LSPG, no mutation):
      hist  A c_k, A c_{k-1} and, on CN steps, F(c_k)
      pred  the batched residual norms of the three candidates and the argmin
      init  evalJ at the start point + gradient J^T r + stationarity ratio
      iter  one LM body (J^T J, damped Cholesky solve, clip, evalJ at the trial point, gradient, ratio, selects)
  * kernel components (H-cost-r): N, J, JTJ, chol (verdict) and res, evalJ, g (descriptive).
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

import p1core as T2
import qstudy as QS


def _time(fn, args, reps, burn, warm=3):
    """0.1 s burn (outside the interval), then warm calls, then `reps` synchronised calls (A1.10); all raw times."""
    QS.burn(burn)
    for _ in range(warm):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t)
    return ts


def lspg_parts(nl, nlJ):
    def inner(w, Aw, val, hist, nu, lam, co, dt):
        a0, a1, a2, b0, b1 = co[0], co[1], co[2], co[3], co[4]
        Acn, Acm, Fn = hist
        return a0 * Aw + a1 * Acn + a2 * Acm + dt * (b0 * (val + nu * lam * Aw) + b1 * Fn)

    def res(w, hist, nu, data, co, dt):
        Aw = data['A'] @ w
        return inner(w, Aw, nl(w, data), hist, nu, data['lam'], co, dt) / (co[0] + dt * co[3] * nu * data['lam'])

    def evalJ(w, hist, nu, data, co, dt):
        Aw = data['A'] @ w
        val, dN = nlJ(w, data, None)
        R = inner(w, Aw, val, hist, nu, data['lam'], co, dt)
        S = 1. / (co[0] + dt * co[3] * nu * data['lam'])
        return R * S, data['A'] + (dt * co[3] * S)[:, None] * dN
    return res, evalJ


def make_solver_components(nl, nlJ, trust):
    res, evalJ = lspg_parts(nl, nlJ)

    def ratio(g, J, rn):
        return jnp.linalg.norm(g) / (jnp.linalg.norm(J) * rn + 1e-300)

    @jax.jit
    def hist_be(ck, cm, nu, data):
        return data['A'] @ ck, data['A'] @ cm

    @jax.jit
    def hist_cn(ck, cm, nu, data):
        return data['A'] @ ck, data['A'] @ cm, nl(ck, data) + nu * data['lam'] * (data['A'] @ ck)

    @jax.jit
    def pred(wv, wprev, wprev2, hist, nu, data, co, dt):
        we = wv + (wv - wprev)
        wqd = 3. * wv - 3. * wprev + wprev2
        cand = jnp.stack((wv, we, wqd))
        rs = jax.vmap(lambda ww: jnp.linalg.norm(res(ww, hist, nu, data, co, dt)))(cand)
        rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
        return cand[jnp.argmin(rs)]

    @jax.jit
    def init(w0, hist, nu, data, co, dt):
        r, J = evalJ(w0, hist, nu, data, co, dt)
        rn = jnp.linalg.norm(r)
        g = J.T @ r
        return r, J, g, rn, ratio(g, J, rn)

    @jax.jit
    def iteration(w, r, J, g, rn, lam, hist, nu, data, co, dt):
        Hm = J.T @ J
        d = jnp.diag(Hm) + 1e-30
        c = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
        step = jax.scipy.linalg.cho_solve(c, -g)
        nz = jnp.linalg.norm(step)
        step = step * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
        ok = jnp.all(jnp.isfinite(step))
        wn = w + jnp.where(ok, step, 0.)
        r2, J2 = evalJ(wn, hist, nu, data, co, dt)
        rn2 = jnp.linalg.norm(r2)
        accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
        w3 = jnp.where(accept, wn, w)
        r3 = jnp.where(accept, r2, r)
        J3 = jnp.where(accept, J2, J)
        rn3 = jnp.where(accept, rn2, rn)
        g3 = J3.T @ r3
        lam2 = jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
        return w3, r3, J3, g3, rn3, lam2, ratio(g3, J3, rn3)
    return dict(hist_be=hist_be, hist_cn=hist_cn, pred=pred, init=init, iter=iteration)


def component_times(st, scheme, nu, w0, scale, reps=30, burn=.1):
    """Median times of the solver sub-computations of one setting (state w0; cost is value-independent)."""
    import p1common as P
    nl, nlJ = st.q0['LSPG']['nl'], P.Q.tested_jac('point', st.L)
    comps = make_solver_components(nl, nlJ, st.trust)
    d = st.data[('main', 'LSPG')]
    co = jnp.asarray(T2.SCHEMES[scheme][:5], jnp.float64)
    dt = jnp.asarray(P.DT0, jnp.float64)
    w = jnp.asarray(w0)
    hist = comps['hist_cn'](w, w, nu, d)
    raw = {}
    raw['hist_be'] = _time(comps['hist_be'], (w, w, nu, d), reps, burn)
    raw['hist_cn'] = _time(comps['hist_cn'], (w, w, nu, d), reps, burn)
    raw['pred'] = _time(comps['pred'], (w, w, w, hist, nu, d, co, dt), reps, burn)
    raw['init'] = _time(comps['init'], (w, hist, nu, d, co, dt), reps, burn)
    r, J, g, rn, _ = comps['init'](w, hist, nu, d, co, dt)
    lam = jnp.asarray(1e-6, jnp.float64)
    raw['iter'] = _time(comps['iter'], (w, r, J, g, rn, lam, hist, nu, d, co, dt), reps, burn)
    out = {k: float(np.median(v)) for k, v in raw.items()}
    out['raw'] = raw
    return out


# ------------------------------------------------------------------ kernel components (H-cost-r) ----

@jax.jit
def k_N(c, Gq, Gs, Psi):
    return Psi.T @ ((Gq @ c) * (Gs @ c))


@jax.jit
def k_J(c, Gq, Gs, Psi):
    u, s = Gq @ c, Gs @ c
    return Psi.T @ (s[:, None] * Gq + u[:, None] * Gs)


@jax.jit
def k_JTJ(J):
    return J.T @ J


@jax.jit
def k_chol(Hm, g, lam):
    d = jnp.diag(Hm) + 1e-30
    c = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
    return jax.scipy.linalg.cho_solve(c, -g)


@jax.jit
def k_res(c, A, lam_, Gq, Gs, Psi, nu, dt):
    Aw = A @ c                                  # BE residual with previous projection p = 0 (cost is value-independent)
    return (Aw + dt * (Psi.T @ ((Gq @ c) * (Gs @ c)) + nu * lam_ * Aw)) / (1 + dt * nu * lam_)


@jax.jit
def k_evalJ(c, A, lam_, Gq, Gs, Psi, nu, dt):
    u, s = Gq @ c, Gs @ c
    S = 1. / (1 + dt * nu * lam_)
    Aw = A @ c
    r = (Aw + dt * (Psi.T @ (u * s) + nu * lam_ * Aw)) * S
    return r, A + (dt * S)[:, None] * (Psi.T @ (s[:, None] * Gq + u[:, None] * Gs))


@jax.jit
def k_g(J, r):
    return J.T @ r


def flops_bytes(comp, m, M, r):
    """Operation model of DESIGN 4.2a (f64)."""
    if comp == 'N':
        return 4 * m * r + m + 2 * m * M, 8 * (2 * m * r + m * M)
    if comp == 'J':
        return 2 * m * M * r + 4 * m * r, 8 * (2 * m * r + m * M + M * r)
    if comp == 'JTJ':
        return 2 * M * r * r, 8 * (M * r + r * r)
    if comp == 'chol':
        return r ** 3 / 3 + 2 * r * r, 8 * r * r
    raise KeyError(comp)


def kernel_panel(A, lam_, tabs, fixtures, ranks, nu, dt, reps=30, burn=.1, seed=0, comps=('N', 'J', 'JTJ', 'chol', 'res', 'evalJ', 'g')):
    """Time every component at every rank on every fixture. tabs(r) -> (A_r, lam, Gq_r, Gs_r, Psi) for that rank.
    fixtures: (n_fix, R_max) states truncated per rank. Order of ranks randomised per fixture. Returns raw times."""
    rng = np.random.default_rng(seed)
    out = []
    for fi, cfull in enumerate(fixtures):
        order = list(ranks)
        rng.shuffle(order)
        for r in order:
            Ar, lr, Gq, Gs, Psi = tabs(r)
            c = jnp.asarray(cfull[:r])
            J = k_J(c, Gq, Gs, Psi)
            Jf = Ar + (dt / (1 + dt * nu * lr))[:, None] * J
            Hm = k_JTJ(Jf)
            rv = k_res(c, Ar, lr, Gq, Gs, Psi, nu, dt)
            g = k_g(Jf, rv)
            lam = jnp.asarray(1e-6, jnp.float64)
            args = dict(N=(k_N, (c, Gq, Gs, Psi)), J=(k_J, (c, Gq, Gs, Psi)), JTJ=(k_JTJ, (Jf,)), chol=(k_chol, (Hm, g, lam)),
                        res=(k_res, (c, Ar, lr, Gq, Gs, Psi, nu, dt)), evalJ=(k_evalJ, (c, Ar, lr, Gq, Gs, Psi, nu, dt)),
                        g=(k_g, (Jf, rv)))
            for comp in comps:
                fn, a_ = args[comp]
                ts = _time(fn, a_, reps, burn)
                out.append(dict(fixture=fi, r=int(r), comp=comp, times=ts, median=float(np.median(ts))))
    return out


def _settle(fn, args, burn, warm=3):
    """The kernel measurement protocol (A1.10): burn, then warm calls, then the timed interval (A6.1)."""
    QS.burn(burn)
    for _ in range(warm):
        jax.block_until_ready(fn(*args))


def dispatch_sync(fn, args, reps=10, burn=.1):
    """Kernel dispatch-sync control (DESIGN 6, A6.1): two separately synchronised dispatches vs one, single-double-single,
    each interval preceded by the same burn + 3 warm calls as the timed kernels it governs."""
    jax.block_until_ready(fn(*args))
    res = []
    for _ in range(reps):
        _settle(fn, args, burn)
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        t1 = time.perf_counter() - t
        _settle(fn, args, burn)
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        jax.block_until_ready(fn(*args))
        tb = time.perf_counter() - t
        _settle(fn, args, burn)
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        t2 = time.perf_counter() - t
        res.append(dict(t1=t1, t2=t2, tb=tb, ratio=tb / (.5 * (t1 + t2))))
    return dict(rows=res, ratio_median=float(np.median([x['ratio'] for x in res])))
